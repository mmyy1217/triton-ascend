// RUN: triton-opt --ta-scope-profile="plan-id=-1" %s | FileCheck %s --check-prefix=PLAN
// RUN: triton-opt --ta-scope-profile="plan-id=1" %s | FileCheck %s --check-prefix=POINTER

// PLAN: ascend.scope_profile.manifest_json
// PLAN-NOT: scope.scope

// POINTER: %[[PTR:.*]] = scope.scope : () -> (!tt.ptr<i32>)
// POINTER: %[[INNER:.*]] = tt.addptr

// POINTER: scope.return %[[INNER]] : !tt.ptr<i32>
// POINTER: vector_mode = "simt"

module {
  tt.func public @scope_profile(%base: !tt.ptr<i32>, %offset: i32) attributes {
      ta.auto_blockify_v1,
      ta.auto_blockify_v1.superblock_factor = 1 : i32
  } {
    %c0 = arith.constant {ta.auto_blockify_v1.schedule} 0 : index
    %c1 = arith.constant {ta.auto_blockify_v1.schedule} 1 : index
    scf.for %iv = %c0 to %c1 step %c1 {
      %schedule = arith.index_cast %iv {ta.auto_blockify_v1.schedule} : index to i32
      %ptr = tt.addptr %base, %offset : !tt.ptr<i32>, i32
      %value = tt.load %ptr : !tt.ptr<i32>
      %one = arith.constant 1 : i32
      %sum = arith.addi %value, %one : i32
    } {ta.auto_blockify_v1.loop, ta.auto_blockify_v1.schedule}
    tt.return {ta.auto_blockify_v1.schedule}
  }
}
