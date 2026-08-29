// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// A canonical SIMT scope must select both the mixed BiShengIR pipeline and
// the SIMT-aware runtime launch path.
// CHECK-LABEL: func.func @simt_scope
// CHECK-SAME: parallel_mode = "mix_simd_simt"
// CHECK: scope.scope
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-SAME: hivm.vf_mode = #hivm.vf_mode<SIMT>
// CHECK-SAME: no_inline
// CHECK-SAME: outline
// CHECK-SAME: vector_mode = "simt"
tt.func public @simt_scope(%arg0: !tt.ptr<f32>) {
  scope.scope : () -> () {
    scope.return
  } {vector_mode = "simt"}
  tt.return
}
