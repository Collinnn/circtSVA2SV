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
  Value constTrue;
  //Implication specific
  Value leftMost;
  Value rhsStart;

  //Implication, find leftmost value and save it globally
  //Only one implication is possible
  Value findLeftMost(Value v){
    while(auto *defOp = v.getDefiningOp()) {
      if (auto impOp = llvm::dyn_cast<ltl::ImplicationOp>(defOp)){
        v = (impOp.getAntecedent());
      }
      else if (auto concatOp = llvm::dyn_cast<ltl::ConcatOp>(defOp)){
        v = concatOp.getInputs()[0];
      }
      else if (auto delayOp = llvm::dyn_cast<ltl::DelayOp>(defOp)){
        return v;
      }
      else if (auto andOp = llvm::dyn_cast<ltl::AndOp>(defOp)){
        v = andOp.getOperands()[0];
      }
      else if (auto orOp = llvm::dyn_cast<ltl::OrOp>(defOp)){
        v = orOp.getOperands()[0];
      }
      else if (auto unrealizedConversionCastOp = llvm::dyn_cast<UnrealizedConversionCastOp>(defOp)){
        v = unrealizedConversionCastOp.getOperand(0);
      } else{
        return v;
      }
    }    
    return v;
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
      if (auto delayOp = llvm::dyn_cast<ltl::DelayOp>(defOp)){
        return delayOp.getDelay() + getTotalDelay(delayOp.getOperand());
      }
      int64_t sum = 0;
      for (auto operand : defOp->getOperands())
        sum += getTotalDelay(operand);
      return sum;
    }
    return 0;
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
      Value constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 1);
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
      auto i1 = rewriter.getI1Type();
      Value constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 1);

      Value clock = clockMap[op];
      if (!clock){
        llvm::errs() << "No clock found for signal in LTL Implication\n" << clock << "\n";
        return failure();
      }

      //Handles if only one delay is shown
      Value v = rhs;
      while (auto cast = v.getDefiningOp<UnrealizedConversionCastOp>()) {
        v = cast.getOperand(0);
      }
      // Now inspect the real defining op
      if (auto delayOp = v.getDefiningOp<ltl::DelayOp>()) {
        delayOp.getInputMutable().assign(lhs);
      }
      
      Value curr = lhs;
      //If Rhs has zero delays
      if (rhsDelay == 0){
        auto notOp = rewriter.create<comb::XorOp>(op.getLoc(),rhs, constOne);
        auto orOp = rewriter.create<comb::OrOp>(op.getLoc(), curr, notOp);
        rewriter.replaceOp(op, orOp.getResult());
        return success();
      }
      for (int64_t i = 0; i < rhsDelay; i++){
        curr = rewriter.create<seq::CompRegOp>(
          op.getLoc(),
          curr, // input
          clock // clk
        );
      }
      if(!isa<IntegerType>(rhs.getType())){
        rhs.setType(IntegerType::get(rhs.getContext(), 1));
      }
      auto notOp = rewriter.create<comb::XorOp>(op.getLoc(),curr, constOne);
      auto orOp = rewriter.create<comb::OrOp>(op.getLoc(), rhs, notOp);
      rhsStart = rhs;
      rewriter.replaceOp(op, orOp.getResult());
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
    auto result = rewriter.create<seq::ToClockOp>(op.getLoc(), adaptor.getClock());
    Value signal = adaptor.getInput();
    Value clock = result.getResult();

    if (op.getEdge() == ltl::ClockEdge::Neg) {
      // Invert the clock for negative edge
      clock = rewriter.create<seq::ClockInverterOp>(
        op.getLoc(), clock);
    } else if (op.getEdge() == ltl::ClockEdge::Both) {
      llvm::errs() << "Both edges not supported\n";
      return failure();
    } 
    //Find the leftmostValue
    leftMost = findLeftMost(signal);
    llvm::errs() << "Leftmost value is: " << leftMost << "\n";

    rewriter.replaceOp(op, signal);
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

    if (lengthValue != 0){
      llvm::errs() << "Variable delay not yet supported";
      return failure();
    }
    
    //Get clock from struct
    Value clock = clockMap[op];
    if (!clock) {
      llvm::errs() << "No clock attribute found on Delay op\n" << clock << "\n";
      return failure();
    }
    Value curr;
    //Check if ltl.concat exists
    if (auto concat = op.getResult().getDefiningOp<ltl::ConcatOp>()){
      //Find this value in the concat op
      for (size_t i = 0; i < concat.getInputs().size(); i++){
        if (concat.getInputs()[i] == op.getResult()){
          if (i == 0){
            if (leftMost == op.getResult()){
              curr = rewriter.create<hw::ConstantOp>(op.getLoc(), rewriter.getI1Type(), 1);
            }else{
              llvm::errs( ) << "Leftmost value from other side of implication is" << rhsStart << "\n";
              curr = rhsStart;
            }
          }else{
            curr = concat.getInputs()[i-1];
          }
        }
      }
    }    
    else{
      curr = signal;

      if (leftMost == op.getResult()){
        auto constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), rewriter.getI1Type(), 1); 
        curr = constOne;
      }else{
        curr = rhsStart;
      }
      
    }
    // if length and delay is zero, no register needed it just a sequence
    if (lengthValue == 0 && delay == 0){
      auto andOp = rewriter.create<comb::AndOp>(op.getLoc(), curr, signal);
      rewriter.replaceOp(op, andOp.getResult());
      return success();
    }
    //Create the shift reg op
    for (int64_t i = 0; i < delay; i++){
      curr = rewriter.create<seq::CompRegOp>(
        op.getLoc(),
        curr, // input
        clock // clk
      );
    }
    auto andOp = rewriter.create<comb::AndOp>(op.getLoc(), curr, signal);

    rewriter.replaceOp(op, andOp.getResult());
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
    
    auto i1 = rewriter.getI1Type();
    Value constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 1);

    // Successor condition that the LEFTMOST operand must satisfy
    Value nextCond = constOne;
    llvm::SmallVector<Value, 8> valueVec;
    valueVec.resize(n);
    // Walk left to right
    for (int i = 0; i < n; i++) {
      Value in = inputs[i];
      
      Value signal = in;
      //Current DelayOp
      while(auto cast = signal.getDefiningOp<UnrealizedConversionCastOp>()){
        signal = cast.getOperand(0);
      }
      valueVec[i] = signal;
      while(auto andOp = signal.getDefiningOp<comb::AndOp>()){
        signal = andOp.getInputs()[0];
      }
      Value compreg;
      while(auto compRegOp = signal.getDefiningOp<seq::CompRegOp>()){
        compreg = compRegOp;
        signal = compRegOp.getInput();
      }
      if (i == 0){
        if (leftMost == op.getInputs()[0]){
          nextCond = constOne;
        }
        
      }else{
        if (auto compregOp = compreg.getDefiningOp<seq::CompRegOp>()){
          compregOp->setOperand(0, valueVec[i-1]);
        }else if(auto andOp = signal.getDefiningOp<comb::AndOp>()){
          andOp->setOperand(0, valueVec[i-1]);
        }
      }
    }
    rewriter.replaceOp(op, valueVec[n-1]);
    return success();
    }
  };

  struct VerifAssertOpConversion : OpConversionPattern<verif::AssertOp> {
    using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      // Change the type from ltl.property to i1
      auto newop = rewriter.create<verif::AssertOp>(
          op.getLoc(), adaptor.getProperty(),
          adaptor.getEnable(), adaptor.getLabelAttr());
      rewriter.replaceOp(op, newop);
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
    return IntegerType::get(type.getContext(), type.getWidth());
  });
  //Inout type conversion
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
  target.addIllegalOp<verif::AssertOp>();
  
  target.addDynamicallyLegalOp<verif::AssertOp>(
  [&](verif::AssertOp op) {
    //If not I1, then illegal
    if (!isa<IntegerType>(op.getProperty().getType()))
      return false;
    return true;
  });

  RewritePatternSet patterns(&getContext());

  patterns.add<LTLAndOpConversion>(converter, patterns.getContext());
  patterns.add<LTLOrOpConversion>(converter, patterns.getContext());
  patterns.add<LTLNotOpConversion>(converter, patterns.getContext());
  patterns.add<LTLConcatOpConversion>(converter, patterns.getContext());
  patterns.add<HasBeenResetOpConversion>(converter, patterns.getContext());
  patterns.add<VerifAssertOpConversion>(converter, patterns.getContext());

  // Apply the conversions
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();

  //Unrealized Conversion cleanup
  SmallVector<UnrealizedConversionCastOp> casts;
  getOperation().walk([&] (UnrealizedConversionCastOp cast){
    casts.push_back(cast);
  });
  
  for (auto cast : casts){
    //If no users just erase
    if(cast->getUsers().empty()){
      cast.erase();
      continue;
    }
    //Recursive find the starting signal before unrealized chain and replace all uses with it
    Value orignalSignal = cast.getResult(0);
    while(auto internalcast = orignalSignal.getDefiningOp<UnrealizedConversionCastOp>()){
        orignalSignal = internalcast.getOperand(0);
    }
    //Replace the signal connected to it
    // Replace all uses of the outermost cast with the original signal
    cast.getResult(0).replaceAllUsesWith(orignalSignal);
    cast.erase(); 
  }
}
// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerVerifPass() {
  return std::make_unique<LowerVerifPass>();
}

std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}