//===- LTLToCore.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts LTL and Verif operations to Core operations
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LTLToCore.h"
#include "circt/Conversion/HWToSV.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/SymbolTable.h"
namespace circt {
#define GEN_PASS_DEF_LOWERLTLTOCORE
#define GEN_PASS_DEF_LOWERVERIF
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
  llvm::DenseMap<Value, Value> clockMap;
  llvm::DenseMap<Value, Value> delayDependcies;
  //Implication specific
  Value leftMost;
  Value rhsStart;

  //Assume Implication 
  void findLeftMost(Value v){
    if (!v)
      return;
    if (auto *defOp = v.getDefiningOp()) {
      if (auto impOp = llvm::dyn_cast<ltl::ImplicationOp>(defOp)){
        findLeftMost(impOp.getAntecedent());
      }
      else if (auto concatOp = llvm::dyn_cast<ltl::ConcatOp>(defOp)){
        leftMost = v;
      }
      else if (auto delayOp = llvm::dyn_cast<ltl::DelayOp>(defOp)){
        leftMost = v;
        return;
      }
    }
    return;
  }
  
  void propagateClock(Value v, Value clockVal) {
    if (!v || !clockVal) return;
    // Already propagated
    if (clockMap.count(v)) return;
    clockMap[v] = clockVal;
    // Propagate forward to all users
    for (auto *user : v.getUsers()) {
      for (auto result : user->getResults()) {
        propagateClock(result, clockVal);
      }
    }
    // Propagate backward to operands
    if (auto *defOp = v.getDefiningOp()) {
      for (auto operand : defOp->getOperands()) {
        propagateClock(operand, clockVal);
      }
    }
    return;
  }

  int64_t getTotalDelay(Value val) {
    if (!val)
      return 0;
    if (auto *defOp = val.getDefiningOp()) {
      if (auto delayOp = llvm::dyn_cast<ltl::DelayOp>(defOp))
        return delayOp.getDelay() + getTotalDelay(delayOp.getOperand());

    int64_t sum = 0;
    for (auto operand : defOp->getOperands())
      sum += getTotalDelay(operand);
    return sum;
    }
    return 0;
  }

  Value getStartOfChain(Value val) {
    if (!val)
      return Value();
    if (isa<hw::InOutType>(val.getType()))
      return val;
    if (auto *defOp = val.getDefiningOp()) {
      if(!isa<hw::InOutType>(defOp->getOperand(0).getType()))
        return getStartOfChain(defOp->getOperand(0));
    }
    
    if(!isa<IntegerType>(val.getType())){
      val.setType(IntegerType::get(val.getContext(), 1));
    }

    return val;
  }

struct HasBeenResetOpConversion : OpConversionPattern<verif::HasBeenResetOp> {
  using OpConversionPattern<verif::HasBeenResetOp>::OpConversionPattern;

  // HasBeenReset generates a 1 bit register that is set to one once the reset
  // has been raised and lowered at at least once.
  LogicalResult
  matchAndRewrite(verif::HasBeenResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i1 = rewriter.getI1Type();
    // Generate the constant used to set the register value
    Value constZero = seq::createConstantInitialValue(
        rewriter, op->getLoc(), rewriter.getIntegerAttr(i1, 0));

    // Generate the constant used to negate the reset value
    Value constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 1);

    // Create a backedge for the register to be used in the OrOp
    circt::BackedgeBuilder bb(rewriter, op.getLoc());
    circt::Backedge reg = bb.get(rewriter.getI1Type());

    // Generate an or between the reset and the register's value to store
    // whether or not the reset has been active at least once
    Value orReset =
        rewriter.create<comb::OrOp>(op.getLoc(), adaptor.getReset(), reg);

    // This register should not be reset, so we give it dummy reset and resetval
    // operands to fit the build signature
    Value reset, resetval;

    // Finally generate the register to set the backedge
    reg.setValue(rewriter.create<seq::CompRegOp>(
        op.getLoc(), orReset,
        rewriter.createOrFold<seq::ToClockOp>(op.getLoc(), adaptor.getClock()),
        rewriter.getStringAttr("hbr"), reset, resetval, constZero,
        InnerSymAttr{} // inner_sym
        ));

    // We also need to consider the case where we are currently in a reset cycle
    // in which case our hbr register should be down-
    // Practically this means converting it to (and hbr (not reset))
    Value notReset =
        rewriter.create<comb::XorOp>(op.getLoc(), adaptor.getReset(), constOne);
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, reg, notReset);

    return success();
    }
  };
  struct LTLAndOpConversion : OpConversionPattern<ltl::AndOp> {
  using OpConversionPattern<ltl::AndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the LTL 'and' with a comb.and
    auto result = rewriter.create<comb::AndOp>(
        op.getLoc(), adaptor.getOperands()[0], adaptor.getOperands()[1]);
    rewriter.replaceOp(op, result.getResult());
    return success();
    }
  };
  struct LTLOrOpConversion : OpConversionPattern<ltl::OrOp> {
    using OpConversionPattern<ltl::OrOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ltl::OrOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
    // Replace the LTL 'or' with a comb.or
    auto result = rewriter.create<comb::OrOp>(
        op.getLoc(), adaptor.getOperands()[0], adaptor.getOperands()[1]);
    rewriter.replaceOp(op, result.getResult());
    return success();
    }
  };
  struct LTLNotOpConversion : OpConversionPattern<ltl::NotOp> {
    using OpConversionPattern<ltl::NotOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ltl::NotOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      Value operand = adaptor.getOperands()[0];
      auto i1 = rewriter.getI1Type();
      auto constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 1);
      // Replace the LTL 'not' with a comb.xor
      auto result = rewriter.create<comb::XorOp>(
          op.getLoc(), operand, constOne);
      rewriter.replaceOp(op, result.getResult());
      return success();
    }
  };
  struct LTLImplication : OpConversionPattern<ltl::ImplicationOp> {
    using OpConversionPattern<ltl::ImplicationOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ltl::ImplicationOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
      Value lhs = adaptor.getAntecedent();
      Value rhs = adaptor.getConsequent();
      //Get RHS recursive tree search for all delay ops
      int64_t rhsDelay = getTotalDelay(rhs);
      Value rhsStart = getStartOfChain(rhs);
      auto i1 = rewriter.getI1Type();
      Value constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 1);
      Value constZero = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 0);

      //TODO: CLEAN UP THE CLOCK LOOKUP
      Value ImplicationClock = clockMap[op];
      Value lhsClock = clockMap[lhs];
      Value rhsClock = clockMap[rhs];
      Value clock = lhsClock ? lhsClock : rhsClock;
      clock = clock ? clock : ImplicationClock;
      if (!clock){
        llvm::errs() << "No clock found for signal in LTL Implication\n" << clock << "\n";
        return failure();
      }
      findLeftMost(op.getAntecedent());

      Value notLhs = rewriter.create<comb::XorOp>(op.getLoc(),lhs,constOne);
      Value rhsTrigger = rewriter.create<comb::OrOp>(op.getLoc(),notLhs,rhsStart);
      rhsStart = rhsTrigger;
      //ShiftReg can't be zero in length
      if (rhsDelay == 0){
        rewriter.replaceOp(op,rhsTrigger);
        return success();
      }
      Value curr = lhs;
      //Create the shift reg op
      for (int64_t i = 0; i < rhsDelay; i++){
        curr = rewriter.create<seq::CompRegOp>(
          op.getLoc(),
          curr,                // input
          clock               // clk
        );
      }
      if(!isa<IntegerType>(rhs.getType()))
        rhs.setType(IntegerType::get(rhs.getContext(), 1));

      auto andOp = rewriter.create<comb::AndOp>(op.getLoc(), curr, rhs);
      rewriter.replaceOp(op, andOp.getResult());
      return success();
    }
  };
  struct LTLClockOp : OpConversionPattern<ltl::ClockOp> {
    using OpConversionPattern<ltl::ClockOp>::OpConversionPattern;
    // In your OpConversionPattern<ltl::ClockOp>:
    LogicalResult
    matchAndRewrite(ltl::ClockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Save the value that represents the LTL op's single result
    auto result = rewriter.create<seq::ToClockOp>(op.getLoc(), op.getOperands()[1]);
    Value clock = result.getResult();

    if (op.getEdge() == ltl::ClockEdge::Neg) {
      // Invert the clock for negative edge
      clock = rewriter.create<seq::ClockInverterOp>(
          op.getLoc(), clock);
    } else if (op.getEdge() == ltl::ClockEdge::Both) {
      llvm::errs() << "Both edges not supported\n";
      return failure();
    } 


    rewriter.replaceOp(op, clock);
    propagateClock(op.getResult(),clock);
    propagateClock(op.getInput(),clock);

    if (!clock) {
      llvm::errs() << "No clock generated for LTL ClockOp\n";
      return failure();
    }
    return success();
    }
  };

  struct LTLDelay : OpConversionPattern<ltl::DelayOp>{
      using OpConversionPattern<ltl::DelayOp>::OpConversionPattern;

    explicit LTLDelay(mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ltl::DelayOp>{typeConverter, ctx, 1} {
    // Enable rewrite recursion to make sure nested `loop` directives are
    // handled.
    this->setHasBoundedRewriteRecursion(true);
    }
    LogicalResult
    matchAndRewrite(ltl::DelayOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {  

    Value signal = adaptor.getOperands()[0];
    int64_t delay = op.getDelay();
    std::optional<uint64_t> length = op.getLength();
    if (!length.has_value()){
      llvm::errs() << "Infinite length not supported\n";
      return failure();
    }
    // Is the m in ##[n:m], can be zero
    int64_t lengthValue = length.value();
    // if length and delay is zero, no register needed it just a sequence
    if (lengthValue == 0 && delay == 0){
      rewriter.replaceOp(op, signal);
      return success();
    }
    
    //Get clock from struct
    Value clock = clockMap[op];
    if (!clock) {
      llvm::errs() << "No clock attribute found on Delay op\n" << clock << "\n";
      return failure();
    }
    Value curr = signal;
    //Create the shift reg op
    for (int64_t i = 0; i < delay; i++){
        curr = rewriter.create<seq::CompRegOp>(
          op.getLoc(),
          curr,                // input
          clock               // clk
        );
    }
    llvm::errs() << "shift reg made \n";
    
    if (lengthValue != 0){
      llvm::errs() << "Variable delay not yet supported";
      return failure();
    }

    rewriter.replaceOp(op, curr);
    return success();
    }
  };
  struct LTLConcatOpConversion : OpConversionPattern<ltl::ConcatOp> {
    using OpConversionPattern<ltl::ConcatOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ltl::ConcatOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
    
    auto inputs = adaptor.getInputs();
    int n = inputs.size();

    // const true
    Value constOne = rewriter.create<hw::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(), 1);

    // --- backward pass ---
    // Successor condition that the RIGHTMOST operand must satisfy
    Value nextCond = constOne;

    // Walk left to right
    for (int i = 0; i < n; i++) {

      Value in = inputs[i];

      // Remove compreg layers so we get the *pure* signal
      Value pure = in;
      llvm::errs() << "Input before peel" << pure << "\n";
      while(auto cast = pure.getDefiningOp<UnrealizedConversionCastOp>()){
        pure = cast.getOperand(0);
      }

      if (i != 0){
        Value lookback = inputs[i-1];
        while(auto cast = lookback.getDefiningOp<UnrealizedConversionCastOp>()){
          lookback = cast.getOperand(0);
        }
        nextCond = lookback.getDefiningOp<seq::CompRegOp>();
      }
      seq::CompRegOp firstCompReg = nullptr;
      while (auto cr = pure.getDefiningOp<seq::CompRegOp>()){
        pure = cr.getInput();
        firstCompReg = cr;
      }
      // Each operand’s output condition is: pure AND nextCond
      Value andOp = rewriter.create<comb::AndOp>(op.getLoc(), pure, nextCond);
      firstCompReg->setOperand(0, andOp);
    }

      // The whole concat op returns lhs condition
      rewriter.replaceOp(op, nextCond);
      return success();
    }
  };
}
//===----------------------------------------------------------------------===//
// Lower LTL To Core pass
//===----------------------------------------------------------------------===//
namespace {
struct LowerVerifPass
    : public circt::impl::LowerVerifBase<LowerVerifPass> {
  LowerVerifPass() = default;
  void runOnOperation() override;
};
} // namespace

namespace {
struct LowerLTLToCorePass
    : public circt::impl::LowerLTLToCoreBase<LowerLTLToCorePass> {
  LowerLTLToCorePass() = default;
  void runOnOperation() override;
};
} // namespace

void LowerVerifPass::runOnOperation(){
  // Create type converters, mostly just to convert an ltl property to a bool
  mlir::TypeConverter converter;

  // Convert the ltl property type to a built-in type
  converter.addConversion([](IntegerType type) { return type; });
  converter.addConversion([](ltl::PropertyType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([](ltl::SequenceType type) {
    return IntegerType::get(type.getContext(), 1);
  });

  // Basic materializations
  converter.addTargetMaterialization(
    [&](mlir::OpBuilder &builder, mlir::Type resultType,
      mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
    if (inputs.size() != 1)
      return Value();
    return builder
      .create<UnrealizedConversionCastOp>(loc, resultType, inputs[0])
      ->getResult(0);
  });

  converter.addSourceMaterialization(
    [&](mlir::OpBuilder &builder, mlir::Type resultType,
      mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
    if (inputs.size() != 1)
      return Value();
    return builder
      .create<UnrealizedConversionCastOp>(loc, resultType, inputs[0])
      ->getResult(0);
  });

  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addLegalDialect<verif::VerifDialect>();
}

// Simply applies the conversion patterns defined above
void LowerLTLToCorePass::runOnOperation() {

  // Create type converters, mostly just to convert an ltl property to a bool
  mlir::TypeConverter converter;

  // Convert the ltl property type to a built-in type
  converter.addConversion([](IntegerType type) { return type; });
  converter.addConversion([](ltl::PropertyType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([](ltl::SequenceType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  // Convert Moore types from lowered System Verilog  
  converter.addConversion([](moore::IntType type){
    return IntegerType::get(type.getContext(), 1);
  });
  //Inout convertered to i1
  converter.addConversion([](hw::InOutType type){
    return IntegerType::get(type.getContext(),1);
  });

  // Basic materializations
  converter.addTargetMaterialization(
    [&](mlir::OpBuilder &builder, mlir::Type resultType,
      mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
    if (inputs.size() != 1)
      return Value();
    return builder
      .create<UnrealizedConversionCastOp>(loc, resultType, inputs[0])
      ->getResult(0);
  });

  converter.addSourceMaterialization(
    [&](mlir::OpBuilder &builder, mlir::Type resultType,
      mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
    if (inputs.size() != 1)
      return Value();
    return builder
      .create<UnrealizedConversionCastOp>(loc, resultType, inputs[0])
      ->getResult(0);
  });

  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addLegalDialect<verif::VerifDialect>();
  target.addIllegalOp<ltl::ClockOp>();

  RewritePatternSet clockpattern(&getContext());
  clockpattern.add<LTLClockOp>(converter, clockpattern.getContext());


  if (failed(
      applyPartialConversion(getOperation(), target, std::move(clockpattern))))
    return signalPassFailure();
  

  RewritePatternSet earlypatterns(&getContext());
  target.addIllegalOp<ltl::ImplicationOp>();
  earlypatterns.add<LTLImplication>(converter, earlypatterns.getContext());

  if (failed(
      applyPartialConversion(getOperation(), target, std::move(earlypatterns))))
    return signalPassFailure();

  target.addIllegalOp<ltl::DelayOp>();

  RewritePatternSet delayPattern(&getContext());
  delayPattern.add<LTLDelay>(converter, delayPattern.getContext());


  if (failed(
      applyPartialConversion(getOperation(), target, std::move(delayPattern))))
    return signalPassFailure();

  target.addIllegalDialect<ltl::LTLDialect>();
  target.addIllegalOp<verif::HasBeenResetOp>();

  RewritePatternSet patterns(&getContext());

  patterns.add<LTLAndOpConversion>(converter, patterns.getContext());
  patterns.add<LTLOrOpConversion>(converter, patterns.getContext());
  patterns.add<LTLNotOpConversion>(converter, patterns.getContext());
  patterns.add<LTLConcatOpConversion>(converter, patterns.getContext());
  patterns.add<HasBeenResetOpConversion>(converter, patterns.getContext());
  // Apply the conversions
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
  }
// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerVerifPass() {
  return std::make_unique<LowerVerifPass>();
}


std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}