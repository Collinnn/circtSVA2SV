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
  enum ClockEdge : uint8_t {
    Pos,
    Neg,
    Both
  };

  struct ClockInfo {
    Value seqClock;
    ClockEdge edge;
  };
  ClockInfo clockInfo;


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
  struct LTLImplication : OpConversionPattern<ltl::ImplicationOp> {
    using OpConversionPattern<ltl::ImplicationOp>::OpConversionPattern;
    LogicalResult
    matchAndRewrite(ltl::ImplicationOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
      Value lhs = adaptor.getOperands()[0];
      Value rhs = adaptor.getOperands()[1];
      auto type = lhs.getType();
      auto constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), type, 1);
      // Replace the LTL 'not' with a comb.xor
      auto notOperator = rewriter.create<comb::XorOp>(op.getLoc(), lhs, constOne);
      auto result = rewriter.create<comb::OrOp>(op.getLoc(), notOperator, rhs);
      rewriter.replaceOp(op, result.getResult());
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
        auto result = rewriter.create<seq::ToClockOp>(
            op.getLoc(), clock); 
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
        clockInfo.seqClock = seqClock;
        clockInfo.edge = ClockEdge::Pos;
        break;
      case ltl::ClockEdge::Neg:
        clockInfo.seqClock = rewriter.create<seq::ClockInverterOp>(
            op.getLoc(), clock);
        clockInfo.edge = ClockEdge::Neg;
        break;
      case ltl::ClockEdge::Both:
        llvm::errs() << "Both edges not supported yet";
        clockInfo.edge = ClockEdge::Both;
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
      Value clock = clockInfo.seqClock;

      // Ensure clock is seq::ClockType
      if (!isa<seq::ClockType>(clock.getType())) {
        llvm::errs() << "Clock provided to LTL Delay is not of seq::ClockType\n";
        return failure();
      }
      
      Value signal = adaptor.getOperands()[0];
      Value data;
      int64_t delay = op.getDelay();
      std::optional<uint64_t> length = op.getLength();
      if (!length.has_value()){
        llvm::errs() << "Infinite length not supported yet\n";
        return failure();
      }else{
        // Is the m in ##[n:m], can be zero
        int64_t lengthValue = length.value();
        auto depthAttr = rewriter.getI64IntegerAttr(lengthValue + delay);

        //Reset signal for now is constantly 0
        Value constZero = rewriter.create<hw::ConstantOp>(
            op.getLoc(), signal.getType(), rewriter.getIntegerAttr(signal.getType(), 0));
        Value constOne = rewriter.create<hw::ConstantOp>(
            op.getLoc(), signal.getType(), rewriter.getIntegerAttr(signal.getType(), 1));¨
          
        // Control mask, sets bits from delay+1 to length+delay+1
        uint64_t maskBits = ((1ULL << (lengthValue + delay + 1)) - 1) & ~((1ULL << delay) - 1);
        auto shiftWidth = lengthValue + delay; // matches your ShiftReg depth
        auto maskType = rewriter.getIntegerType(shiftWidth);
        auto maskAttr = rewriter.getIntegerAttr(maskType, maskBits);
        Value mask = rewriter.create<hw::ConstantOp>(op.getLoc(), maskAttr);

        //Create the shift reg op
        auto shiftReg = rewriter.create<seq::ShiftRegOp>(
          op.getLoc(),
          signal.getType(),                 // result type
          depthAttr,                        // numElements (I64Attr)
          signal,                           // input
          clock,                            // clk
          constOne,                         // clk en (always enabled)
          rewriter.getStringAttr("ltl_delay_"), // optional name
          Value(),                          // reset (none)
          Value(),                          // resetValue (none)
          constZero,                        // powerOnValue (constant)
          hw::InnerSymAttr{}                // inner_sym
        );
        //Shift reg size, vs mask size
        llvm::errs() << "ShiftReg size: " << shiftReg.getType() << "\n";
        llvm::errs() << "Mask size: " << mask.getType() << "\n";
        // Apply the mask on the output
        Value masked = rewriter.create<comb::AndOp>(op.getLoc(), shiftReg, mask);

        //Starts at zero
        Value trigger = rewriter.create<hw::ConstantOp>(
          op.getLoc(), rewriter.getIntegerType(1), rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0));
        
        for (int i = delay; i < lengthValue + delay; i++){
          Value bit = rewriter.create<comb::ExtractOp>(
          op.getLoc(), rewriter.getIntegerType(1), masked, static_cast<uint32_t>(i));
          trigger = rewriter.create<comb::OrOp>(op.getLoc() , trigger, bit);
        }
        rewriter.replaceOp(op, trigger);
      }

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

  // Set target dialects: We don't want to see any ltl or verif that might
  // come from an AssertProperty left in the result
  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addIllegalDialect<ltl::LTLDialect>();
  target.addLegalDialect<verif::VerifDialect>();
  target.addIllegalOp<verif::HasBeenResetOp>();


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

  // Create the operation rewrite patters
  RewritePatternSet patterns(&getContext());
  patterns.add<HasBeenResetOpConversion>(converter, patterns.getContext());
  patterns.add<LTLAndOpConversion>(converter, patterns.getContext());
  patterns.add<LTLOrOpConversion>(converter, patterns.getContext());
  patterns.add<LTLNotOpConversion>(converter, patterns.getContext());
  patterns.add<LTLImplication>(converter, patterns.getContext());
  patterns.add<LTLClock>(converter, patterns.getContext());
  patterns.add<LTLDelay>(converter, patterns.getContext());
  // Apply the conversions
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}
