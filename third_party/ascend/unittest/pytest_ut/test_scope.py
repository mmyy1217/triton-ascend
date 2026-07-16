#!/usr/bin/env python3
import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import pytest
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import (
    _analyze_auto_simt_scope_features,
    _estimate_auto_simt_scope_decision,
    _is_whole_body_void_simt_scope,
    _wrap_whole_body_void_simt_scope,
)


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel to MLIR."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, src, context, Options(), {}, {})
    return str(module)


# ============== Kernel definitions ==============


@triton.jit
def kernel_nested_scope(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test nested scopes."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        with al.scope(core_mode="vector"):
            with al.scope(core_mode="cube"):
                x = tl.load(x_ptr + i, mask=i < n)
                y = tl.load(y_ptr + i, mask=i < n)
                result = x + y
                tl.store(out_ptr + i, result, mask=i < n)


@triton.jit
def kernel_scope_escape(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test variable defined inside scope, used outside."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=i < n)
    # Use x outside of the scope
    a = x + 1.0
    tl.store(out_ptr + i, a, mask=i < n)


@triton.jit
def kernel_scope_cube(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test cube core mode."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="cube"):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)


@triton.jit
def kernel_scope_vector(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test vector core mode."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)


@triton.jit
def kernel_scope_disable_auto_sync(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test disable auto sync."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)


@triton.jit
def kernel_scope_vector_mode_simt(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test vector_mode SIMT annotation."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(vector_mode="simt"):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)


# ============== Pytest tests ==============


def test_nested_scope():
    """Test nested scopes compile successfully."""
    mlir = compile_kernel(
        kernel_nested_scope, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256}
    )
    assert "scope.scope" in mlir
    assert len(mlir) > 0


def test_scope_escape():
    """Test variable escaping from scope."""
    mlir = compile_kernel(kernel_scope_escape, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256})
    assert "scope.scope" in mlir
    assert len(mlir) > 0


def test_scope_cube_mode():
    """Test cube core mode generates correct attributes."""
    mlir = compile_kernel(
        kernel_scope_cube, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256}
    )
    assert "scope.scope" in mlir
    # Check for cube core type attribute
    assert "hivm.tcore_type" in mlir or "CUBE" in mlir.upper()


def test_scope_vector_mode():
    """Test vector core mode generates correct attributes."""
    mlir = compile_kernel(
        kernel_scope_vector, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256}
    )
    assert "scope.scope" in mlir
    # Check for vector core type attribute
    assert "hivm.tcore_type" in mlir or "VECTOR" in mlir.upper()


def test_scope_disable_auto_sync():
    """Test disable auto sync generates correct attributes."""
    mlir = compile_kernel(
        kernel_scope_disable_auto_sync,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert "scope.scope" in mlir
    # Check for disable auto sync attribute
    assert "hivm.disable_auto_sync" in mlir


def test_scope_vector_mode_simt():
    """Test vector_mode='simt' generates the canonical vec_mode attr."""
    mlir = compile_kernel(
        kernel_scope_vector_mode_simt,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert "scope.scope" in mlir
    assert "vec_mode" in mlir
    assert "simt" in mlir


def test_auto_simt_scope_cost_model_selects_rank2_gather_reduce():
    ttir = """
module {
  tt.func public @rank2_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %2 = tt.broadcast %1 : tensor<8x1xi32> -> tensor<8x8xi32>
    %3 = tt.addptr %arg0, %2 : !tt.ptr<f32>, tensor<8x8xi32>
    %4 = tt.load %3 : tensor<8x8x!tt.ptr<f32>>
    %5 = arith.cmpf olt, %4, %4 : tensor<8x8xf32>
    %6 = tt.broadcast %5 : tensor<8x8xi1> -> tensor<8x8xi1>
    %7 = "tt.reduce"(%4) ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<8x8xf32>) -> tensor<8xf32>
    %8 = tt.addptr %arg2, %0 : !tt.ptr<f32>, tensor<8xi32>
    tt.store %8, %7 : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
"""
    features = _analyze_auto_simt_scope_features(ttir)
    decision = _estimate_auto_simt_scope_decision(features)
    assert decision["eligible"]
    assert decision["choose_simt"]
    assert decision["decision_kind"] == "all_simt_only"
    assert set(decision["candidate_costs"]) == {"all_simd", "all_simt_only", "mixed_simd_simt"}
    assert decision["candidate_costs"]["all_simd"] == decision["simd_cost"]
    assert decision["candidate_costs"]["all_simt_only"] == decision["simt_cost"]
    assert decision["candidate_costs"]["mixed_simd_simt"] == decision["simt_cost"] + decision["boundary_cost"]
    assert decision["transition_costs"]["simd_to_simt"] > 0
    assert decision["transition_costs"]["simt_to_simd"] == 0
    assert decision["cost_breakdown"]["simd"]["aggregation"] == "max(vector_roofline, scalar_path)"
    assert decision["cost_breakdown"]["mixed_simd_simt"]["available_in_p1"] is False


def test_auto_simt_scope_cost_model_keeps_rank1_vector_kernel():
    ttir = """
module {
  tt.func public @rank1_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.addptr %arg0, %0 : !tt.ptr<f32>, tensor<128xi32>
    %2 = tt.load %1 : tensor<128x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<128xf32>
    %4 = tt.addptr %arg1, %0 : !tt.ptr<f32>, tensor<128xi32>
    tt.store %4, %3 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
"""
    features = _analyze_auto_simt_scope_features(ttir)
    decision = _estimate_auto_simt_scope_decision(features)
    assert not decision["choose_simt"]
    assert decision["decision_kind"] == "all_simd"
    assert decision["candidate_costs"]["all_simd"] == decision["simd_cost"]
    assert decision["reason"] == "rank1_or_scalar_kernel"


def test_auto_simt_scope_cost_model_selects_rank1_indirect_vector_reduce():
    ttir_template = """
module {
  tt.func public @fbgemm_like(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f16>, %arg3: !tt.ptr<i32>, %arg4: !tt.ptr<i32>, %arg5: !tt.ptr<f16>, %arg6: i32) {
    %c128_i64 = arith.constant 128 : i64
    %pid = tt.get_program_id x : i32
    %token_ptr = tt.addptr %arg3, %pid : !tt.ptr<i32>, i32
    %token = tt.load %token_ptr : !tt.ptr<i32>
    %expert_ptr = tt.addptr %arg4, %pid : !tt.ptr<i32>, i32
    %expert = tt.load %expert_ptr : !tt.ptr<i32>
    %scale_base = arith.muli %token, %arg6 : i32
    %scale_ptr0 = tt.addptr %arg5, %scale_base : !tt.ptr<f16>, i32
    %scale_ptr = tt.addptr %scale_ptr0, %expert : !tt.ptr<f16>, i32
    %scale = tt.load %scale_ptr : !tt.ptr<f16>
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %token_i64 = arith.extsi %token : i32 to i64
    %row_offset = arith.muli %token_i64, %c128_i64 : i64
    %in_base = tt.addptr %arg2, %row_offset : !tt.ptr<f16>, i64
    %in_base_vec = tt.splat %in_base : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>>
    %in_ptrs = tt.addptr %in_base_vec, %offs : tensor<128x!tt.ptr<f16>>, tensor<128xi32>
    %x = tt.load %in_ptrs : tensor<128x!tt.ptr<f16>>
    %xf = arith.extf %x : tensor<128xf16> to tensor<128xf32>
    %abs = math.absf %xf : tensor<128xf32>
    %mx = "tt.reduce"(%abs) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) : (tensor<128xf32>) -> f32
    tt.store %arg1, %mx : !tt.ptr<f32>
    %out_base_vec = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<128x!tt.ptr<f8E4M3FN>>
    %out_ptrs = tt.addptr %out_base_vec, %offs : tensor<128x!tt.ptr<f8E4M3FN>>, tensor<128xi32>
    %out = tt.fp_to_fp %xf, rounding = rtne : tensor<128xf32> -> tensor<128xf8E4M3FN>
    tt.store %out_ptrs, %out : tensor<128x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}
"""
    for block_d in (16, 128, 256, 1024, 2048):
        ttir = ttir_template.replace("128", str(block_d))
        features = _analyze_auto_simt_scope_features(ttir)
        decision = _estimate_auto_simt_scope_decision(features)
        assert features["max_tensor_rank"] == 1
        assert features["max_tensor_numel"] == block_d
        assert features["rank1_indirect_vector_reduce"]
        assert decision["rank1_indirect_whole_body_eligible"]
        assert decision["choose_simt"]
        assert decision["decision_kind"] == "all_simt_only"
        assert decision["reason"] == "simt_cost_lower_than_simd_with_margin"


def test_auto_simt_scope_cost_model_reports_dot_mixed_path():
    ttir = """
module {
  tt.func public @gather_dot_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<i32>, %arg3: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.load %arg2 : !tt.ptr<i32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %3 = tt.broadcast %2 : tensor<16x1xi32> -> tensor<16x16xi32>
    %4 = tt.addptr %arg0, %3 : !tt.ptr<f16>, tensor<16x16xi32>
    %5 = tt.load %4 : tensor<16x16x!tt.ptr<f16>>
    %6 = tt.addptr %arg1, %3 : !tt.ptr<f16>, tensor<16x16xi32>
    %7 = tt.load %6 : tensor<16x16x!tt.ptr<f16>>
    %8 = tt.dot %5, %7 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    %9 = tt.addptr %arg3, %3 : !tt.ptr<f32>, tensor<16x16xi32>
    tt.store %9, %8 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
"""
    features = _analyze_auto_simt_scope_features(ttir)
    decision = _estimate_auto_simt_scope_decision(features)
    assert features["has_dot"]
    assert features["dot_ops"] == 1
    assert decision["decision_kind"] == "mixed_simd_simt"
    assert decision["choose_simt"] is False
    assert decision["dot_mixed_eligible"] is True
    assert decision["whole_body_simt_eligible"] is False
    assert decision["candidate_costs"]["mixed_simd_simt"] < decision["candidate_costs"]["all_simd"]
    assert decision["reason"] == "dot_mixed_path_lower_than_simd_and_simt"


def test_auto_simt_scope_cost_model_selects_atomic_scatter_simt_route():
    ttir = """
module {
  tt.func public @atomic_scatter(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %idx_ptrs = tt.addptr %arg1, %offs : !tt.ptr<i32>, tensor<256xi32>
    %idx = tt.load %idx_ptrs : tensor<256x!tt.ptr<i32>>
    %val_ptrs = tt.addptr %arg2, %offs : !tt.ptr<f32>, tensor<256xi32>
    %val = tt.load %val_ptrs : tensor<256x!tt.ptr<f32>>
    %dst = tt.addptr %arg0, %idx : !tt.ptr<f32>, tensor<256xi32>
    tt.atomic_add %dst, %val sem = "relaxed" : tensor<256x!tt.ptr<f32>>, tensor<256xf32>
    tt.return
  }
}
"""
    features = _analyze_auto_simt_scope_features(ttir)
    decision = _estimate_auto_simt_scope_decision(features)
    assert features["has_atomic"]
    assert features["atomic_ops"] == 1
    assert decision["atomic_whole_body_eligible"]
    assert decision["choose_simt"]
    assert decision["decision_kind"] == "all_simt_only"
    assert decision["reason"] == "atomic_or_histogram_simt_route_lower_than_simd"


def test_auto_simt_scope_cost_model_selects_histogram_simt_route():
    ttir = """
module {
  tt.func public @histogram_kernel(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %idx_ptrs = tt.addptr %arg0, %offs : !tt.ptr<i32>, tensor<128xi32>
    %idx = tt.load %idx_ptrs : tensor<128x!tt.ptr<i32>>
    %hist = tt.histogram %idx : tensor<128xi32> -> tensor<128xi32>
    %out_ptrs = tt.addptr %arg1, %offs : !tt.ptr<i32>, tensor<128xi32>
    tt.store %out_ptrs, %hist : tensor<128x!tt.ptr<i32>>
    tt.return
  }
}
"""
    features = _analyze_auto_simt_scope_features(ttir)
    decision = _estimate_auto_simt_scope_decision(features)
    assert features["has_histogram"]
    assert features["histogram_ops"] == 1
    assert decision["atomic_whole_body_eligible"]
    assert decision["choose_simt"]
    assert decision["decision_kind"] == "all_simt_only"
    assert decision["reason"] == "atomic_or_histogram_simt_route_lower_than_simd"


def test_auto_simt_scope_cost_model_accounts_for_scan_without_forcing_simt():
    ttir = """
module {
  tt.func public @regular_scan(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %in_ptrs = tt.addptr %arg0, %offs : !tt.ptr<f32>, tensor<1024xi32>
    %x = tt.load %in_ptrs : tensor<1024x!tt.ptr<f32>>
    %scan = "tt.scan"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }) : (tensor<1024xf32>) -> tensor<1024xf32>
    %out_ptrs = tt.addptr %arg1, %offs : !tt.ptr<f32>, tensor<1024xi32>
    tt.store %out_ptrs, %scan : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
"""
    features = _analyze_auto_simt_scope_features(ttir)
    decision = _estimate_auto_simt_scope_decision(features)
    assert features["has_scan"]
    assert features["scan_ops"] == 1
    assert "scan" in decision["reason"]
    assert set(decision["candidate_costs"]) == {"all_simd", "all_simt_only", "mixed_simd_simt"}


def test_auto_simt_scope_wraps_whole_body_as_void_simt_scope():
    ttir = """
module {
  tt.func public @wrap_kernel(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    tt.return
  }
}
"""
    scoped = _wrap_whole_body_void_simt_scope(ttir)
    assert "scope.scope : () -> ()" in scoped
    assert 'vec_mode = "simt"' in scoped
    assert _is_whole_body_void_simt_scope(scoped)


# ============== Main for manual testing ==============

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Nested Scopes")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_nested_scope, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256}
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)

    print("\n" + "=" * 60)
    print("Test 2: Scope Escape")
    print("=" * 60)
    mlir = compile_kernel(kernel_scope_escape, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256})
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)
