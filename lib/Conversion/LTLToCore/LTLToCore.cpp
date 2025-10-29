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
#include <iostream>

namespace circt {
#define GEN_PASS_DEF_LOWERLTLTOCORE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

  llvm::DenseMap<Operation*, Value> opClockMap; 
  
  void createLocalClockMap(Value dataVal, Value clockVal) {
    if (auto op = dataVal.getDefiningOp()) {
      if (!opClockMap.count(op)) {
        opClockMap[op] = clockVal; // Map the op once
      }
      // Recurse through operands
      for (auto operand : op->getOperands()) {
        createLocalClockMap(operand, clockVal);
      }
    }
  }

  int64_t getTotalDelay(mlir::Value val) {
    if (!val)
      return 0;
    int total = 0;
    if (auto *defOp = val.getDefiningOp()) {
      if (auto delayOp = llvm::dyn_cast<circt::ltl::DelayOp>(defOp)) {
        total += delayOp.getDelay() + getTotalDelay(delayOp.getOperand());
      } else {
        for (auto operand : defOp->getOperands()) {
          total += getTotalDelay(operand);
        }
      }
    }
    return total;
  }
  Value getStartOfChain(mlir::Value val) {
    if (auto *defOp = val.getDefiningOp()) {
      return getStartOfChain(defOp->getOperand(0));
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
      auto type = operand.getType();
      auto constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), type, 1);
      // Replace the LTL 'not' with a comb.xor
      auto result = rewriter.create<comb::XorOp>(
          op.getLoc(), operand, constOne);
      rewriter.replaceOp(op, result.getResult());
      return success();
    }
  };
  struct verifClockedAssert : public OpConversionPattern<verif::ClockedAssertOp> {
  using OpConversionPattern<verif::ClockedAssertOp>::OpConversionPattern;
    LogicalResult
    matchAndRewrite(verif::ClockedAssertOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      // Get the clock signal from the adaptor
      Value clock = adaptor.getClock();
      // Ensure clock is seq::ClockType
      if (!isa<seq::ClockType>(clock.getType())) {
        llvm::errs() << "Clock provided to ClockedAssert is not of seq::ClockType\n";
        return failure();
      }

      switch (op.getEdge()) {
        case verif::ClockEdge::Pos:
          // No change needed, clock is already positive edge
          break;
        case verif::ClockEdge::Neg:
          // Invert the clock for negative edge
          clock = rewriter.create<seq::ClockInverterOp>(
              op.getLoc(), clock);
          break;
        case verif::ClockEdge::Both:
          llvm::errs() << "Both edges not supported\n";
          return failure();
      }

      Value property = adaptor.getOperands()[0];
      llvm::errs() << "Property is of the type:" << property.getType() << "\n";
      // Add the signal to the map, which allows every function to look up the clock later
      createLocalClockMap(property, clock);
      auto constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), rewriter.getI1Type(), 1);

      rewriter.replaceOpWithNewOp<verif::AssertOp>(
        op,
        TypeRange{},        // no result types
        property,           // property value
        constOne,           // enable signal
        op.getLabelAttr()  // optional label attr
      );
      return success();
    }
  };
  struct LTLImplication : OpConversionPattern<ltl::ImplicationOp> {
    using OpConversionPattern<ltl::ImplicationOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ltl::ImplicationOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
      
      //Get RHS recursive tree search for all delay ops
      int64_t rhsDelay = getTotalDelay(adaptor.getOperands()[1]);
      Value rhsStart = getStartOfChain(adaptor.getOperands()[1]);
      Value lhs = adaptor.getOperands()[0];
      Value rhs = adaptor.getOperands()[1];
      auto constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), lhs.getType(), 1);
      // Replace the LTL 'not' with a comb.xor
      auto notOperator = rewriter.create<comb::XorOp>(op.getLoc(), lhs, constOne);

      // TODO: DIRECT THIS TO THE START OF THE RHS CHAIN
      auto result = rewriter.create<comb::OrOp>(op.getLoc(), notOperator, rhsStart);
      Value constZero = rewriter.create<hw::ConstantOp>(op.getLoc(), lhs.getType(), 0);
      if (rhsDelay == 0){
        rewriter.replaceOp(op, result.getResult());
        return success();
      }
      
      Value clock = opClockMap[op.getOperation()];
      if (!clock){
        llvm::errs() << "No clock found for signal in LTL Implication\n";
        return failure();
      }
      
      auto shiftReg = rewriter.create<seq::ShiftRegOp>(
          op.getLoc(),
          rhs.getType(),                      // result type
          rewriter.getI64IntegerAttr(rhsDelay),  // numElements (I64Attr)
          lhs,                              // input
          clock,                           // clk
          constOne,                        // clk en (always enabled)
          rewriter.getStringAttr("implication_reg"), // optional name
          Value(),                          // reset (none)
          Value(),                          // resetValue (none)
          constZero, // powerOnValue (constant)
          hw::InnerSymAttr{}                // inner_sym
      );
      auto andOp = rewriter.create<comb::AndOp>(op.getLoc(), shiftReg, rhs);
      rewriter.replaceOp(op, andOp.getResult());
      return success();
    }
  };
  struct LTLClock : OpConversionPattern<ltl::ClockOp> {
    using OpConversionPattern<ltl::ClockOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ltl::ClockOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      Value clock = op.getOperands()[0];
      Value seqClock;
      if (!isa<seq::ClockType>(clock.getType())) {
        // Replace the LTL 'clock' with a seq.to_clock
        auto result = rewriter.create<seq::ToClockOp>(op.getLoc(), clock); 
        rewriter.replaceOp(op, result.getResult());
        seqClock = result.getResult();
      }else{
        seqClock = clock;
        // remove the ltl.clock op
        rewriter.replaceOp(op, clock); 
      }
      // Store the edge information
      switch (op.getEdge()) {
      case ltl::ClockEdge::Pos:
        createLocalClockMap(op.getResult(), seqClock);
        break;
      case ltl::ClockEdge::Neg:
        seqClock = rewriter.create<seq::ClockInverterOp>(
            op.getLoc(), seqClock);
        createLocalClockMap(op.getResult(), seqClock);
        break;
      case ltl::ClockEdge::Both:
        llvm::errs() << "Both edges not supported";
        break;
      }
      return success();
    }
  };

  struct LTLDelay : OpConversionPattern<ltl::DelayOp>{
      using OpConversionPattern<ltl::DelayOp>::OpConversionPattern;

      LogicalResult
      matchAndRewrite(ltl::DelayOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override {
      //Get clock from struct
      Value clock = opClockMap[op.getOperation()];

      // Ensure clock is seq::ClockType
      if (!isa<seq::ClockType>(clock.getType())) {
        llvm::errs() << "Clock provided to LTL Delay is not of seq::ClockType\n";
        return failure();
      }
      rewriter.setInsertionPoint(op);

      Value signal = adaptor.getOperands()[0];
      int64_t delay = op.getDelay();
      std::optional<uint64_t> length = op.getLength();
      if (!length.has_value()){
        llvm::errs() << "Infinite length not supported\n";
        return failure();
      }else{
        // Is the m in ##[n:m], can be zero
        int64_t lengthValue = length.value();
        // if length and delay is zero, no register needed it just a sequence
        if (lengthValue == 0 && delay == 0){
          rewriter.replaceOp(op, signal);
          return success();
        }

        //Reset signal for now is constantly 0
        Value constZero = rewriter.create<hw::ConstantOp>(
            op.getLoc(), signal.getType(), rewriter.getIntegerAttr(signal.getType(), 0));
        Value constOne = rewriter.create<hw::ConstantOp>(
            op.getLoc(), signal.getType(), rewriter.getIntegerAttr(signal.getType(), 1));

        auto users = op.getResult().getUsers();
        Value userResult;
        if (std::distance(users.begin(), users.end()) > 3){
          llvm::errs() << "Delay op with multiple users not supported\n";
          llvm::errs() << "Number of users: " << std::distance(users.begin(), users.end()) << "\n";
          llvm::errs() << "Users:\n";
          for (auto user : users){
            llvm::errs() << " - " << *user << "\n";
          }
          return failure();
        } else if (std::distance(users.begin(), users.end()) == 0){
          // no users, always true
          userResult = constOne;          
          return success();
        }else{
          Operation *userOp = *(users.begin());
          userResult = userOp->getResult(0);
        }
        //Create the shift reg op
        auto shiftReg = rewriter.create<seq::ShiftRegOp>(
          op.getLoc(),
          rewriter.getI1Type(),                 // result type
          rewriter.getI64IntegerAttr(delay),    // numElements (I64Attr)
          userResult,                           // input
          clock,                            // clk
          constOne,                         // clk en (always enabled)
          rewriter.getStringAttr("delay_reg"), // optional name
          Value(),                          // reset (none)
          Value(),                          // resetValue (none)
          constZero,                        // powerOnValue (constant)
          hw::InnerSymAttr{}                // inner_sym
        );
        if (lengthValue != 0){
          llvm::errs() << "Variable delay not yet supported";
          return failure();
        }
        //and Of orignal signal with value from shift reg
        auto andOp = rewriter.create<comb::AndOp>(
            op.getLoc(), shiftReg, signal);

        rewriter.replaceOp(op, andOp.getResult());
      }

      return success();
    }
  };
  struct LTLConcatOpConversion : OpConversionPattern<ltl::ConcatOp> {
    using OpConversionPattern<ltl::ConcatOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ltl::ConcatOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      // Replace the LTL 'concat' with a comb.concat
      auto result = rewriter.create<comb::ConcatOp>(
          op.getLoc(), adaptor.getOperands());
      rewriter.replaceOp(op, result.getResult());
      return success();
    }
  };
}
//===----------------------------------------------------------------------===//
// Lower LTL To Core pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerLTLToCorePass
    : public circt::impl::LowerLTLToCoreBase<LowerLTLToCorePass> {
  LowerLTLToCorePass() = default;
  void runOnOperation() override;
};
} // namespace

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
  target.addIllegalOp<verif::ClockedAssertOp>();
  target.addIllegalOp<ltl::ClockOp>();


  RewritePatternSet clockpattern(&getContext());
  clockpattern.add<verifClockedAssert>(converter, clockpattern.getContext());
  clockpattern.add<LTLClock>(converter, clockpattern.getContext());
  if (failed(
      applyPartialConversion(getOperation(), target, std::move(clockpattern))))
    return signalPassFailure();


  RewritePatternSet earlypatterns(&getContext());
  target.addIllegalOp<ltl::ImplicationOp>();
  earlypatterns.add<LTLImplication>(converter, earlypatterns.getContext());

  if (failed(
      applyPartialConversion(getOperation(), target, std::move(earlypatterns))))
    return signalPassFailure();


  target.addIllegalDialect<ltl::LTLDialect>();
  target.addIllegalOp<verif::HasBeenResetOp>();
  
  RewritePatternSet patterns(&getContext());
  patterns.add<HasBeenResetOpConversion>(converter, patterns.getContext());
  patterns.add<LTLAndOpConversion>(converter, patterns.getContext());
  patterns.add<LTLOrOpConversion>(converter, patterns.getContext());
  patterns.add<LTLNotOpConversion>(converter, patterns.getContext());
  patterns.add<LTLDelay>(converter, patterns.getContext());
  patterns.add<LTLConcatOpConversion>(converter, patterns.getContext());
  // Apply the conversions
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}