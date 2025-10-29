// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s
// Tests that an Assert Property high level statement can be converted correctly

module {
  //CHECK:  hw.module @test(in %clock : !seq.clock, in %reset : i1, in %a : i1)
  hw.module @test(in %clock : !seq.clock, in %reset : i1, in %a : i1) {
    //CHECK:  [[CLK:%.+]] = seq.from_clock %clock
    %0 = seq.from_clock %clock 
    // CHECK-NEXT: %[[INIT:.+]] = seq.initial() {
    // CHECK-NEXT:   %false = hw.constant false
    // CHECK-NEXT:   seq.yield %false : i1
    // CHECK-NEXT: } : () -> !seq.immutable<i1>

    //CHECK:  %true = hw.constant true
    //CHECK:  [[TMP:%.+]] = comb.or %reset, %hbr : i1
    //CHECK:  %hbr = seq.compreg [[TMP]], %clock initial %[[INIT]] : i1
    %1 = verif.has_been_reset %0, sync %reset

    //CHECK:  [[TMP1:%.+]] = comb.xor %reset, %true : i1
    //CHECK:  [[TMP2:%.+]] = comb.and %hbr, [[TMP1]] : i1
    //CHECK:  verif.clocked_assert %a if [[TMP2]], posedge [[CLK]] : i1
    verif.clocked_assert %a if %1, posedge %0 : i1

    //CHECK:  hw.output
    hw.output 
  }

  hw.module @AndLowering(in %a : i1, in %b : i1) {

    // Create two 1-bit inputs and combine them with an LTL 'and'.
    %and = ltl.and %a, %b : i1, i1

  }
  // CHECK-LABEL: hw.module @AndLowering(in %a : i1, in %b : i1)
  // CHECK: %[[AND:.*]] = comb.and %a, %b : i1
  
  hw.module @OrLowering(in %a : i1, in %b : i1){
    %or = ltl.or %a, %b : i1, i1
  }
  // CHECK-LABEL: hw.module @OrLowering(in %a : i1, in %b : i1)
  // CHECK: %[[OR:.*]] = comb.or %a, %b : i1

  hw.module @NotLowering(in %a : i1){
    %not = ltl.not %a :i1
    %notltl = ltl.not %not : !ltl.property
  }

  // CHECK-LABEL: hw.module @NotLowering(in %a : i1)
  // CHECK: %[[True:.*]] = hw.constant true
  // CHECK: %[[NOT:.*]] = comb.xor %a, %true : i1 
  // CHECK: %[[True_0:.*]] = hw.constant true
  // CHECK: %[[NOTLTL:.*]] = comb.xor %0, %true_0 : i1
  
  hw.module @Implication(in %a :i1, in %b : i1){
    %implication = ltl.implication %a, %b : i1, i1
  }
  // CHECK-LABEL: hw.module @Implication(in %a : i1, in %b : i1)
  // CHECK: %[[True:.*]] = hw.constant true
  // CHECK: %[[NOT:.*]] = comb.xor %a, %true : i1
  // CHECK: %[[Or:.*]] = comb.or %0, %b : i1 
  
  hw.module @Clock(in %a : i1, in %clock : i1){
    %newclock = ltl.clock %clock, posedge %a : i1
  } 
  // CHECK-LABEL: hw.module @Clock(in %a : i1, in %clock : i1)
  // CHECK: %[[Clock:.*]] = seq.to_clock %clock

  hw.module @Delay(in %a : i1, in %clock : i1){
    %newclock = ltl.clock %clock, posedge %a : i1
    %delay = ltl.delay %a, 2,0 : i1
  }
  // CHECK-LABEL: hw.module @Delay(in %a : i1, in %clock : i1)
  // CHECK: %[[Clock:.*]] = seq.to_clock %clock
  // CHECK: %[[False:.*]] = hw.constant false
  // CHECK: %[[True:.*]] = hw.constant true
  // CHECK: %[[DELAY_REG:.*]] = seq.shiftreg[2] %a, %0, %true powerOn %false : i1
  
  hw.module @Concat(in %a : i1, in %b : i1){
    %concat = ltl.concat %a, %b : i1, i1
    %concat2 = ltl.concat %a, %b, %a : i1 , i1 , i1
  }
  // CHECK-LABEL: hw.module @Concat(in %a : i1, in %b : i1)
  // CHECK: %[[CONCAT:.*]] = comb.concat %a, %b : i1
  // CHECK: %[[CONCAT2:.*]] = comb.concat %a, %b, %a : i1   

  hw.module @ImplicationDelay(in %a : i1, in %b : i1, in %clock : i1) {
    %newclock = ltl.clock %clock, posedge %a : i1
    %delay1 = ltl.delay %a, 2,0 : i1
    %1 = ltl.delay %delay1, 2, 0 : !ltl.sequence
    %2 = ltl.delay %1, 2, 0 : !ltl.sequence
    %res = ltl.implication %b, %1 : i1, !ltl.sequence

  }
}