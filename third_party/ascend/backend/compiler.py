# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import ctypes
import functools
import hashlib
import glob
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, Union

from triton._C.libtriton import ir, passes, ascend
from triton.backends.ascend.utils import (
    _check_bishengir_api_change,
    _check_bishengir_able_save_ir,
    _check_bishengir_is_regbased,
    _enable_print_ub_bits,
    _enable_dump_memory_info,
    _get_kernel_target,
    _get_llvm_path,
    _get_mlir_path,
    _get_npucompiler_path,
    _get_triton_adapter_opt_path,
    _is_ascend_sanitizer_enabled,
    _is_debug_line_info_disabled,
    _is_auto_map_parallel_blocks_enabled,
    _get_auto_blockify_blacklist_reasons,
    _warn_auto_blockify_disabled,
    downgrade_llir,
    force_disable_ffts,
    triton_enable_libdevice_simt,
    get_cann_version_file_hash,
)
from triton.backends.ascend.driver import (
    NPUUtils
)
from triton.backends.compiler import (
    AttrsDescriptor,
    BaseBackend,
    GPUTarget,
    register_descriptor,
)
from triton.runtime import driver
from triton.runtime.cache import get_dump_manager
from triton.tools.get_ascend_devices import is_compile_on_910_95


# TODO: materialize the concrete min shape
def min_dot_size(target: GPUTarget):
    return lambda lhsType, rhsType: (1, 1, 1)

# Get result code saved in module {attr_name = rc}
def _get_then_remove_rc(mod, attr_name: str) -> int:
    get_int_attr = getattr(ascend.ir, "get_int_attr", None)
    remove_attr = getattr(ascend.ir, "remove_attr", None)

    if get_int_attr is None:
        return -1
    attr_value = get_int_attr(mod, attr_name)

    if remove_attr:
        remove_attr(mod, attr_name)
    
    if not isinstance(attr_value, int):
        return -1

    return attr_value


def _export_coalesce_metadata(mod, metadata):
    # Tile/strided coalescing (TritonToLinalg) records the chosen coalesce factor
    # H and the grid axis it applies to as module attrs hacc.coalesce_factor /
    # hacc.coalesce_axis. In the full-TA design the *launcher* (driver.py) owns
    # the grid division: it divides grid[axis] by H before launch, and bishengir
    # no longer interprets the attrs. Read them into metadata here and strip them
    # from the module so the hacc.* attrs never reach hivmc (which rejects
    # unknown module attrs). Absent attrs -> factor 1 (no-op) / axis -1.
    factor = _get_then_remove_rc(mod, "hacc.coalesce_factor")
    axis = _get_then_remove_rc(mod, "hacc.coalesce_axis")
    metadata["coalesce_factor"] = factor if isinstance(factor, int) and factor > 1 else 1
    metadata["coalesce_axis"] = axis if isinstance(axis, int) and axis >= 0 else -1


def _adjust_metadata_by_module_result(mod, metadata, opt, **kwargs):
    rc = _get_then_remove_rc(mod, "triton_ascend.dynamic_cv_pipeline.rc")
    if rc != -1 and rc > 0:
        # When the option dynamic_cv_pipeline is set to False,
        # these options should also reverted.
        metadata["enable_dynamic_cv_pipeline"] = False
        metadata["enable_mixed_cv"] = kwargs["enable_mixed_cv"]
        metadata["disable_auto_inject_block_sync"] = kwargs["disable_auto_inject_block_sync"]
        metadata["set_workspace_multibuffer"] = kwargs["set_workspace_multibuffer"]
        if opt.debug:
            print(f"SSBUFFER return code={rc}, will fallback to enable_dynamic_cv_pipeline=False")


def make_ttir(mod, metadata, opt):
    if "hash" not in metadata:
        metadata["hash"] = hashlib.sha256(f"{mod}-{metadata}".encode()).hexdigest()
    # the same optimize pass for triton-ir as all other backends
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod)
    if opt.debug:
        dump_manager = get_dump_manager(metadata["hash"])
        print(f"Dumping intermediate results to {dump_manager.cache_dir}")
        dump_manager.put(str(mod), "kernel.ttir.mlir", binary=False)

    return mod


def ttir_to_linalg(mod, metadata, opt, *, named_ops=False):
    # use triton_adapter to lower Triton-MLIR to linalg
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    ttir_code = _maybe_apply_auto_simt_scope(ttir_code, metadata, opt)
    if metadata.get("compile_mode") == "simd_simt" and _is_whole_body_void_simt_scope(ttir_code):
        metadata["scope_pure_simt_auto"] = True
        metadata["force_simt_only"] = True
        metadata["parallel_mode"] = "simt"
        metadata["shared_mem_dynamic_size"] = 122880
        return _inline_void_simt_scopes_for_pure_simt(ttir_code)

    auto_map_parallel_blocks_enabled = _is_auto_map_parallel_blocks_enabled()
    blacklist_reasons = []
    has_auto_blockify_blacklist_op = metadata.get("has_auto_blockify_blacklist_op")
    if has_auto_blockify_blacklist_op is None and auto_map_parallel_blocks_enabled:
        blacklist_reasons = _get_auto_blockify_blacklist_reasons(ttir_code)
        has_auto_blockify_blacklist_op = bool(blacklist_reasons)
    elif has_auto_blockify_blacklist_op is None:
        has_auto_blockify_blacklist_op = False
    metadata["has_auto_blockify_blacklist_op"] = has_auto_blockify_blacklist_op
    if has_auto_blockify_blacklist_op and blacklist_reasons:
        kernel_name = re.search(r"tt\.func\spublic\s+@(\w+)", ttir_code).group(1)
        _warn_auto_blockify_disabled(kernel_name or "<unknown>", blacklist_reasons)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        dst_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(src_path).write_text(ttir_code)
        triton_adapter_opt_path = _get_triton_adapter_opt_path()

        enable_nd2nz_on_vector = metadata["enable_nd2nz_on_vector"]
        enable_select_analysis = metadata["enable_select_analysis"]
        compile_on_910_95 = metadata["compile_on_910_95"]
        force_simt_template = metadata["force_simt_template"]
        enable_sync_block_lock = metadata["enable_sync_block_lock"]
        enable_mask_fallback_conversion = metadata["enable_mask_fallback_conversion"]
        optimize_dynamic_offset = metadata["optimize_dynamic_offset"]
        auto_blockify_size = metadata["auto_blockify_size"]
        enable_mixed_cv = metadata["enable_mixed_cv"]
        disable_auto_inject_block_sync = metadata["disable_auto_inject_block_sync"]
        set_workspace_multibuffer = metadata["set_workspace_multibuffer"]
        if has_auto_blockify_blacklist_op or not auto_map_parallel_blocks_enabled:
            auto_blockify_size = 1
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        ascend.passes.ttir.add_auto_blockify(
            pm,
            auto_blockify_size
        )

        ascend.passes.ttir.add_triton_control_flow_opt(pm)
        ascend.passes.ttir.add_triton_to_structure(
            pm,
            enable_mask_fallback_conversion,
            optimize_dynamic_offset
        )
        ascend.passes.ttir.add_discrete_mask_access_conversion(
            pm,
            compile_on_910_95,
            force_simt_template,
            enable_sync_block_lock
        )
        ascend.passes.ttir.add_triton_to_annotation(pm)
        ascend.passes.ttir.add_triton_to_unstructure(
            pm,
            compile_on_910_95,
            force_simt_template
        )
        ascend.passes.ttir.add_triton_to_hivm(pm)
        ascend.passes.ttir.add_triton_to_hfusion(
            pm,
            compile_on_910_95)
        ascend.passes.ttir.add_triton_to_llvm(pm)
        ascend.passes.ttir.add_bubble_up_operation(pm)
        ascend.passes.ttir.add_triton_to_structure(
            pm,
            enable_mask_fallback_conversion,
            optimize_dynamic_offset
        )
        ascend.passes.ttir.add_triton_to_linalg(
            pm,
            False,
            named_ops,
            enable_nd2nz_on_vector,
            enable_select_analysis,
            compile_on_910_95
        )
        if metadata["enable_dynamic_cv_pipeline"]:
            metadata["set_workspace_multibuffer"] = 0
            metadata["enable_mixed_cv"] = True
            metadata["disable_auto_inject_block_sync"] = True
            if hasattr(ascend.passes.ttir, "set_enable_cube_block_merge"):
                ascend.passes.ttir.set_enable_cube_block_merge(metadata["enable_cube_block_merge"])

            ascend.passes.ttir.add_dynamic_cv_pipeline(pm, compile_on_910_95)

        _intra_val = metadata.get("intra_cache_num")
        if _intra_val is not None:
            ascend.passes.ttir.set_buffer_count(mod, "INTRA", _intra_val)

        _inter_val = metadata.get("inter_cache_num")
        if _inter_val is not None:
            ascend.passes.ttir.set_buffer_count(mod, "INTER", _inter_val)

        _load_val = metadata.get("load_cache_num")
        if _load_val is not None:
            ascend.passes.ttir.set_buffer_count(mod, "LOAD", _load_val)

        pm.run(mod)
        _adjust_metadata_by_module_result(mod, metadata, opt,
                                          enable_mixed_cv=enable_mixed_cv,
                                          disable_auto_inject_block_sync=disable_auto_inject_block_sync,
                                          set_workspace_multibuffer=set_workspace_multibuffer)
        _export_coalesce_metadata(mod, metadata)

        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(str(mod), "kernel.ttadapter.mlir", binary=False)

        return str(mod)


def __get_metadata_attr_by_callback(lib, postfix: str, metadata, meta_key: str):
    func_symbol = metadata["kernel_name"] + postfix
    if hasattr(lib, func_symbol):
        callback_func = getattr(lib, func_symbol)
        callback_func.restype = ctypes.c_int64
        callback_func.argtypes = []
        metadata[meta_key] = callback_func()


def _parse_linalg_metadata(linalg: str, metadata: dict):
    """
    Parse Linalg IR to extract metadata required for NPU compilation.
    Extracts and updates the following fields in metadata:
      - mix_mode
      - kernel_name
      - tensor_kinds
      - shared (currently hardcoded)
      - name (kernel_name)

    Additionally, removes the mix_mode attribute from the IR.
    """
    # --- Regular expressions and examples ---

    DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX = r'hivm.disable_auto_tile_and_bind_subblock'

    # Inserted by DiscreteMaskAccessConversionPass / MemOpConverter when discrete
    # masked stores need cross-block exclusion (e.g. hivm.sync_block_lock).
    SYNC_BLOCK_LOCK_REGEX = r'\bsync_block_lock\b'

    # Example: mix_mode = "aiv" -> aiv
    MIX_MODE_REGEX = r'mix_mode\s*=\s*"([^"]+)"'

    # Example: parallel_mode = "mix_simd_simt" -> mix_simd_simt
    PARALLEL_MODE_REGEX = r'parallel_mode\s*=\s*"([^"]+)"'

    # Example: func.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
    KERNEL_NAME_REGEX = r"func\.func\s+@(\w+)"

    # Example: %arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} -> ('1', '0')
    TENSOR_KIND_REGEX = r'%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}'

    # Example: bitcode = "a.bc"
    BITCODES_REGEX = r'bitcode\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|(\w+))'

    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, NPU backend does not limit on shared memory
    metadata["shared"] = 1
    # Force disable auto tile and bind subblock if attribute is present in module
    metadata["auto_tile_and_bind_subblock"] = not re.search(DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX, linalg)
    # Turn off auto-blockify when sync_block_lock/unlock was inserted: the lock
    # protects a cross-block read-modify-write and is incompatible with packing
    # logical blocks into a sequential auto-blockify loop.
    if re.search(SYNC_BLOCK_LOCK_REGEX, linalg):
        metadata["has_auto_blockify_blacklist_op"] = True
    # the mix mode is also encoded into metadata['name'] for runtime to distinguish
    metadata["mix_mode"] = re.search(MIX_MODE_REGEX, linalg).group(1)
    metadata["parallel_mode"] = re.search(PARALLEL_MODE_REGEX, linalg).group(1)
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, linalg).group(1)
    # Check the function load_binary in npu_driver.py.
    metadata["name"] = metadata["kernel_name"]
    # Parse all tensor kinds from arguments
    metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, linalg)]
    # init the ub bits of triton kernel for inductor autotune using
    metadata["required_ub_bits"] = 0

    # Parse all bitcode paths
    bitcodes = re.findall(BITCODES_REGEX, linalg)
    metadata["bitcodes"] = [val for group in bitcodes for val in group if val]
    return linalg, metadata


def _parse_ttir_metadata(ttir: str, metadata: dict):
    """
    Parse TTIR to extract metadata required for NPU compilation.
    Extracts and updates the following fields in metadata:
      - kernel_name
      - shared (currently hardcoded)
    """
    # --- Regular expressions and examples ---
    # Example: tt.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
    KERNEL_NAME_REGEX = r"tt\.func\spublic\s+@(\w+)"

    # Example: %arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} -> ('1', '0')
    TENSOR_KIND_REGEX = r'%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}'

    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, NPU backend does not limit on shared memory
    metadata["shared"] = 1
    # Note: Currently, for TTIR inputs, we only support vector kernels.
    metadata["mix_mode"] = "aiv"
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, ttir).group(1)
    metadata["name"] = metadata["kernel_name"]
    auto_map_parallel_blocks_enabled = _is_auto_map_parallel_blocks_enabled()
    has_auto_blockify_blacklist_op = metadata.get("has_auto_blockify_blacklist_op")
    if has_auto_blockify_blacklist_op is None and auto_map_parallel_blocks_enabled:
        has_auto_blockify_blacklist_op = bool(_get_auto_blockify_blacklist_reasons(ttir))
    elif has_auto_blockify_blacklist_op is None:
        has_auto_blockify_blacklist_op = False
    metadata["has_auto_blockify_blacklist_op"] = has_auto_blockify_blacklist_op
    # Parse all tensor kinds from arguments
    metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, ttir)]
    return metadata


def _brace_delta(line: str) -> int:
    return line.count("{") - line.count("}")


def _is_allowed_top_level_constant(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("%") and " = arith.constant " in stripped


def _is_whole_body_void_simt_scope(ttir: str) -> bool:
    """Return true when a kernel body is entirely wrapped by a void SIMT scope."""
    in_func = False
    depth = 0
    scope_count = 0
    saw_return = False

    for line in ttir.splitlines():
        stripped = line.strip()
        if not in_func:
            if "tt.func" in line and "{" in line:
                in_func = True
                depth = _brace_delta(line)
            continue

        if depth == 1:
            if not stripped:
                pass
            elif stripped.startswith("}"):
                pass
            elif _is_allowed_top_level_constant(line):
                pass
            elif stripped.startswith("tt.return"):
                saw_return = True
            elif (
                stripped.startswith("scope.scope")
                and ": () -> ()" in stripped
                and "{" in stripped
            ):
                scope_count += 1
            else:
                return False

        depth += _brace_delta(line)
        if in_func and depth <= 0:
            return scope_count == 1 and saw_return and 'vec_mode = "simt"' in ttir

    return False


def _inline_void_simt_scopes_for_pure_simt(ttir: str) -> str:
    """Inline void SIMT scopes before handing TTIR to the pure-SIMT compiler.

    The pure-SIMT BiSheng pipeline consumes TTIR directly and does not run the
    TritonToUnstructure pass that normally removes consumed SIMT scope markers.
    Keep this intentionally narrow: only erase scope wrappers with no results,
    because result-bearing scopes require SSA value remapping.
    """
    lines = ttir.splitlines()
    out = []
    i = 0
    changed = False

    while i < len(lines):
        line = lines[i]
        if "scope.scope" not in line or ": () -> ()" not in line or "{" not in line:
            out.append(line)
            i += 1
            continue

        depth = _brace_delta(line)
        body = []
        j = i + 1
        close_line = None
        while j < len(lines):
            candidate = lines[j]
            depth += _brace_delta(candidate)
            if depth == 0:
                close_line = candidate
                break
            body.append(candidate)
            j += 1

        if close_line is None or 'vec_mode = "simt"' not in close_line:
            out.append(line)
            i += 1
            continue

        for k in range(len(body) - 1, -1, -1):
            if body[k].lstrip().startswith("scope.return"):
                del body[k]
                break
        out.extend(body)
        changed = True
        i = j + 1

    if not changed:
        return ttir
    return "\n".join(out) + ("\n" if ttir.endswith("\n") else "")


def _normalize_auto_simt_scope_mode(mode: Optional[str]) -> str:
    if mode is None:
        return "off"
    mode = str(mode).strip().lower()
    if mode in ("1", "true", "on", "yes", "auto"):
        return "auto"
    if mode in ("report", "dry_run", "dry-run", "dump"):
        return "report"
    return "off"


def _parse_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _extract_tensor_dims(text: str):
    dims_list = []
    for match in re.finditer(r"tensor<((?:\d+x)+)", text):
        dims = [int(dim) for dim in match.group(1).rstrip("x").split("x") if dim]
        if dims:
            dims_list.append(dims)
    return dims_list


def _tensor_numel(dims):
    numel = 1
    for dim in dims:
        numel *= dim
    return numel


def _count_regex(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


_AUTO_SCOPE_EXIT_COST = 40
_AUTO_TRANSITION_COSTS_BY_NUM_WARPS = {
    # D:\backup\codex\simt_transition_microbench_tail16_barrier_20260713.txt
    # measures in-kernel SYS_CNT slopes for empty async_invoke and mixed
    # SIMD/SIMT orderings. The SIMT->SIMD extra was not positive in the
    # serialized or overlap variants, so the model clamps that direction to 0.
    1: {
        "simd_to_simt": 182,
        "simd_to_simt_with_barrier": 190,
        "simt_to_simd": 0,
    },
    2: {
        "simd_to_simt": 182,
        "simd_to_simt_with_barrier": 190,
        "simt_to_simd": 0,
    },
    4: {
        "simd_to_simt": 182,
        "simd_to_simt_with_barrier": 190,
        "simt_to_simd": 0,
    },
    8: {
        "simd_to_simt": 182,
        "simd_to_simt_with_barrier": 190,
        "simt_to_simd": 0,
    },
    16: {
        "simd_to_simt": 182,
        "simd_to_simt_with_barrier": 190,
        "simt_to_simd": 0,
    },
    32: {
        "simd_to_simt": 223,
        "simd_to_simt_with_barrier": 232,
        "simt_to_simd": 0,
    },
}


def _estimate_auto_simt_transition_costs(num_warps: int = 32) -> Dict[str, Any]:
    try:
        num_warps = int(num_warps)
    except (TypeError, ValueError):
        num_warps = 32

    measured_num_warps = num_warps
    measured = _AUTO_TRANSITION_COSTS_BY_NUM_WARPS.get(num_warps)
    source = "measured"
    if measured is None:
        measured_num_warps = min(
            _AUTO_TRANSITION_COSTS_BY_NUM_WARPS,
            key=lambda value: abs(value - num_warps),
        )
        measured = _AUTO_TRANSITION_COSTS_BY_NUM_WARPS[measured_num_warps]
        source = f"nearest_measured_num_warps_{measured_num_warps}"

    simd_to_simt = measured["simd_to_simt"]
    simt_to_simd = measured["simt_to_simd"]
    simt_to_simd_assumed = 0 if simt_to_simd is None else simt_to_simd
    return {
        "num_warps": num_warps,
        "measured_num_warps": measured_num_warps,
        "source": source,
        "simd_to_simt": simd_to_simt,
        "simd_to_simt_with_barrier": measured.get("simd_to_simt_with_barrier"),
        "simt_to_simd": simt_to_simd,
        "simt_to_simd_assumed": simt_to_simd_assumed,
        "scope_exit": _AUTO_SCOPE_EXIT_COST,
        "unit": "cycles_like_cost_units",
        "note": "SIMT->SIMD extra is clamped to zero unless a positive measured increment exists",
    }


def _analyze_auto_simt_scope_features(ttir: str) -> Dict[str, Any]:
    lines = ttir.splitlines()
    tensor_dims = []
    ptr_tensor_ranks = []
    mask_rank_sum = 0
    mask_tensor_ops = 0
    mask_broadcast_ops = 0
    pointer_unstructured_dims = 0
    lane_dependent_pointer_ops = 0
    row_local_reduce_ops = 0
    scalar_load_ops = 0
    scalar_store_ops = 0
    vector_ptr_splat_ops = 0

    for line in lines:
        dims_in_line = _extract_tensor_dims(line)
        tensor_dims.extend(dims_in_line)

        if "tt.load" in line and "!tt.ptr" in line and not dims_in_line:
            scalar_load_ops += 1
        if "tt.store" in line and "!tt.ptr" in line and not dims_in_line:
            scalar_store_ops += 1
        if "tt.splat" in line and "!tt.ptr" in line and "tensor<" in line:
            vector_ptr_splat_ops += 1

        if "xi1" in line:
            mask_tensor_ops += 1
            for dims in dims_in_line:
                mask_rank_sum += len(dims)
            if "tt.broadcast" in line or "tt.expand_dims" in line:
                mask_broadcast_ops += 1

        if "!tt.ptr" in line and ("tt.addptr" in line or "tt.load" in line or "tt.store" in line):
            max_ptr_rank_on_line = 0
            for dims in dims_in_line:
                rank = len(dims)
                max_ptr_rank_on_line = max(max_ptr_rank_on_line, rank)
                ptr_tensor_ranks.append(rank)
                if rank > 1:
                    pointer_unstructured_dims += rank
            if max_ptr_rank_on_line > 1:
                lane_dependent_pointer_ops += 1

        if "tt.reduce" in line:
            ranks = [len(dims) for dims in dims_in_line]
            if ranks and max(ranks) > min(ranks):
                row_local_reduce_ops += 1

    max_tensor_rank = max((len(dims) for dims in tensor_dims), default=0)
    max_tensor_numel = max((_tensor_numel(dims) for dims in tensor_dims), default=1)
    load_ops = _count_regex(r"\btt\.load\b", ttir)
    store_ops = _count_regex(r"\btt\.store\b", ttir)
    reduce_ops = _count_regex(r"\btt\.reduce\b", ttir)
    vector_reduce_to_scalar_ops = len(
        re.findall(
            r'"tt\.reduce"\(.*?\).*?:\s*\([^)]*tensor<[^>]+>[^)]*\)\s*->\s*(?!tensor<)[a-zA-Z0-9!]+',
            ttir,
            re.S,
        )
    )
    # Do not count region terminators such as `tt.scan.return` as scan ops.
    scan_ops = _count_regex(r"\btt\.(?:associative_)?scan\b(?!\.)", ttir)
    dot_ops = _count_regex(r"\btt\.dot\b", ttir)
    atomic_ops = _count_regex(r"\btt\.atomic", ttir)
    histogram_ops = _count_regex(r"\btt\.histogram\b", ttir)
    broadcast_ops = _count_regex(r"\btt\.broadcast\b", ttir)
    expand_dims_ops = _count_regex(r"\btt\.expand_dims\b", ttir)
    splat_ops = _count_regex(r"\btt\.splat\b", ttir)
    addptr_ops = _count_regex(r"\btt\.addptr\b", ttir)
    arith_ops = _count_regex(r"\barith\.", ttir)
    math_ops = _count_regex(r"\bmath\.", ttir)
    cmp_ops = _count_regex(r"\barith\.cmp[fi]\b", ttir)
    select_ops = _count_regex(r"\barith\.select\b", ttir)
    cast_ops = _count_regex(r"\barith\.(ext|trunc|sitofp|uitofp|fptosi|fptoui|index_cast)", ttir)
    rank1_indirect_vector_reduce = (
        max_tensor_rank == 1
        and reduce_ops > 0
        and vector_reduce_to_scalar_ops > 0
        and vector_ptr_splat_ops > 0
        and scalar_load_ops >= 2
    )

    return {
        "load_ops": load_ops,
        "store_ops": store_ops,
        "reduce_ops": reduce_ops,
        "scan_ops": scan_ops,
        "dot_ops": dot_ops,
        "atomic_ops": atomic_ops,
        "histogram_ops": histogram_ops,
        "broadcast_ops": broadcast_ops,
        "expand_dims_ops": expand_dims_ops,
        "splat_ops": splat_ops,
        "addptr_ops": addptr_ops,
        "arith_ops": arith_ops,
        "math_ops": math_ops,
        "cmp_ops": cmp_ops,
        "select_ops": select_ops,
        "cast_ops": cast_ops,
        "mask_tensor_ops": mask_tensor_ops,
        "mask_rank_sum": mask_rank_sum,
        "mask_broadcast_ops": mask_broadcast_ops,
        "pointer_tensor_ops": len(ptr_tensor_ranks),
        "pointer_unstructured_dims": pointer_unstructured_dims,
        "lane_dependent_pointer_ops": lane_dependent_pointer_ops,
        "row_local_reduce_ops": row_local_reduce_ops,
        "scalar_load_ops": scalar_load_ops,
        "scalar_store_ops": scalar_store_ops,
        "vector_ptr_splat_ops": vector_ptr_splat_ops,
        "vector_reduce_to_scalar_ops": vector_reduce_to_scalar_ops,
        "rank1_indirect_vector_reduce": rank1_indirect_vector_reduce,
        "max_tensor_rank": max_tensor_rank,
        "max_tensor_numel": max_tensor_numel,
        "has_dot": dot_ops > 0,
        "has_atomic": atomic_ops > 0,
        "has_histogram": histogram_ops > 0,
        "has_scan": scan_ops > 0,
        "has_explicit_scope": "scope.scope" in ttir,
        "has_control_flow": "scf." in ttir or "cf." in ttir,
    }


def _estimate_auto_simt_scope_decision(
    features: Dict[str, Any], margin_ratio: float = 0.10, num_warps: int = 32
) -> Dict[str, Any]:
    memory_ops = features["load_ops"] + features["store_ops"]
    scalar_ops = (
        features["arith_ops"]
        + features["math_ops"]
        + features["cmp_ops"]
        + features["select_ops"]
        + features["cast_ops"]
    )
    shape_ops = features["broadcast_ops"] + features["expand_dims_ops"] + features["splat_ops"]
    max_rank = features["max_tensor_rank"]
    max_numel = features["max_tensor_numel"]

    simd_vector_compute = scalar_ops * max(1, max_numel // max(1, max_rank))
    simd_vector_load = features["load_ops"] * max_numel
    simd_vector_store = features["store_ops"] * max_numel
    simd_vector_roofline = max(simd_vector_compute, simd_vector_load, simd_vector_store)
    scalar_address_cost = (
        features["pointer_unstructured_dims"] * 24
        + features["lane_dependent_pointer_ops"] * 96
        + max(0, max_rank - 1) * features["addptr_ops"] * 16
    )
    if features["rank1_indirect_vector_reduce"]:
        scalar_address_cost += (
            features["scalar_load_ops"] * 96
            + features["vector_ptr_splat_ops"] * 128
        )
    scalar_mask_cost = (
        features["mask_tensor_ops"] * 8
        + features["mask_rank_sum"] * 12
        + features["mask_broadcast_ops"] * 96
    )
    reduce_lowering_cost = features["reduce_ops"] * 64 + features["row_local_reduce_ops"] * max(1, max_rank) * 128
    scalarization_penalty = (
        features["lane_dependent_pointer_ops"] * max(0, max_rank - 1) * 96
        + features["mask_broadcast_ops"] * max(1, max_rank) * 64
        + features["row_local_reduce_ops"] * max(1, max_rank) * 160
        + features["pointer_unstructured_dims"] * 48
        + features["scan_ops"] * max_numel
        + features["atomic_ops"] * max_numel * 6
        + features["histogram_ops"] * max_numel * 8
    )
    if features["rank1_indirect_vector_reduce"]:
        scalarization_penalty += (
            features["vector_reduce_to_scalar_ops"] * max_numel * 2
            + features["vector_ptr_splat_ops"] * max_numel * 2
            + features["scalar_load_ops"] * max_numel
        )
    simd_scalar_path = (
        scalar_address_cost
        + scalar_mask_cost
        + reduce_lowering_cost
        + scalarization_penalty
    )
    simd_cost = max(simd_vector_roofline, simd_scalar_path)

    simt_compute = int(scalar_ops * max_numel * 0.55)
    simt_load = int(features["load_ops"] * max_numel * 1.25)
    simt_store = int(features["store_ops"] * max_numel * 1.25)
    simt_memory_roofline = max(simt_load, simt_store)
    simt_predicate = features["mask_tensor_ops"] * 8 + features["mask_rank_sum"] * 8
    simt_shuffle = int((features["reduce_ops"] + features["scan_ops"]) * max_numel * 0.90)
    simt_divergence = features["mask_broadcast_ops"] * 16
    simt_atomic = int(features["atomic_ops"] * max_numel * 1.10 + features["histogram_ops"] * max_numel * 1.35)
    simt_shape = shape_ops * 16
    simt_setup = 180
    simt_cost = (
        max(simt_compute + simt_shuffle, simt_memory_roofline)
        + simt_predicate
        + simt_divergence
        + simt_atomic
        + simt_shape
        + simt_setup
    )

    transition_costs = _estimate_auto_simt_transition_costs(num_warps)
    boundary_cost = (
        transition_costs["simd_to_simt"]
        + transition_costs["simt_to_simd_assumed"]
        + transition_costs["scope_exit"]
    )
    margin = max(64, int(simd_cost * margin_ratio))
    speedup_score = simd_cost - simt_cost - boundary_cost

    rank1_indirect_whole_body_eligible = (
        features["rank1_indirect_vector_reduce"]
        and memory_ops > 0
        and features["store_ops"] > 0
        and not features["has_dot"]
        and not features["has_atomic"]
        and not features["has_histogram"]
        and not features["has_explicit_scope"]
    )
    atomic_whole_body_eligible = (
        (features["has_atomic"] or features["has_histogram"])
        and memory_ops > 0
        and not features["has_dot"]
        and not features["has_explicit_scope"]
    )
    scan_whole_body_eligible = (
        features["has_scan"]
        and memory_ops > 0
        and features["store_ops"] > 0
        and not features["has_dot"]
        and not features["has_atomic"]
        and not features["has_histogram"]
        and not features["has_explicit_scope"]
    )
    non_dot_whole_body_eligible = (
        (
            max_rank >= 2
            or rank1_indirect_whole_body_eligible
            or atomic_whole_body_eligible
            or scan_whole_body_eligible
        )
        and memory_ops > 0
        and (features["store_ops"] > 0 or features["has_atomic"] or features["has_histogram"])
        and not features["has_dot"]
        and not features["has_explicit_scope"]
        and (
            features["lane_dependent_pointer_ops"] > 0
            or features["mask_broadcast_ops"] > 0
            or features["row_local_reduce_ops"] > 0
            or rank1_indirect_whole_body_eligible
            or atomic_whole_body_eligible
            or scan_whole_body_eligible
        )
    )
    dot_mixed_eligible = (
        features["has_dot"]
        and max_rank >= 2
        and memory_ops > 0
        and features["store_ops"] > 0
        and not features["has_atomic"]
        and not features["has_explicit_scope"]
        and (
            features["lane_dependent_pointer_ops"] > 0
            or features["pointer_unstructured_dims"] > 0
            or features["mask_tensor_ops"] > 0
            or features["has_control_flow"]
        )
    )
    eligible = non_dot_whole_body_eligible or dot_mixed_eligible
    choose_simt = non_dot_whole_body_eligible and speedup_score > margin

    all_simd_cost = int(simd_cost)
    all_simt_only_cost = int(simt_cost)
    mixed_simd_simt_cost = int(simt_cost + boundary_cost)
    dot_mixed_path_cost = None
    if dot_mixed_eligible:
        # For dot kernels, `simd_simt` is not the same as a scoped SIMT
        # region. It keeps the dot-friendly lowering while avoiding the worst
        # SIMD scalarization around gather/address/mask setup. The transition
        # microbenchmark cost is therefore not charged here; P1 only reports
        # this path and does not emit a fine-grained partition.
        dot_mixed_path_cost = max(512, int(min(simd_cost, simt_cost) * 0.82))
        mixed_simd_simt_cost = min(mixed_simd_simt_cost, dot_mixed_path_cost)

    candidate_costs = {
        "all_simd": all_simd_cost,
        "all_simt_only": all_simt_only_cost,
        "mixed_simd_simt": mixed_simd_simt_cost,
    }
    decision_kind = "all_simd"
    dot_margin = max(64, int(min(all_simd_cost, all_simt_only_cost) * margin_ratio))
    if choose_simt:
        # P1 only applies whole-body SIMT. Fine-grained mixed regions are
        # reported for the future partitioner, but not emitted here.
        decision_kind = "all_simt_only"
    elif (
        dot_mixed_eligible
        and mixed_simd_simt_cost + dot_margin < min(all_simd_cost, all_simt_only_cost)
    ):
        decision_kind = "mixed_simd_simt"

    confidence = "none"
    if choose_simt:
        confidence = "high" if speedup_score > margin * 3 else "medium"
    elif decision_kind == "mixed_simd_simt":
        mixed_score = min(all_simd_cost, all_simt_only_cost) - mixed_simd_simt_cost
        confidence = "high" if mixed_score > dot_margin * 3 else "medium"
    elif eligible:
        confidence = "low"

    return {
        "eligible": eligible,
        "choose_simt": choose_simt,
        "decision_kind": decision_kind,
        "simd_cost": all_simd_cost,
        "simt_cost": all_simt_only_cost,
        "boundary_cost": boundary_cost,
        "candidate_costs": candidate_costs,
        "transition_costs": transition_costs,
        "cost_breakdown": {
            "simd": {
                "total": all_simd_cost,
                "vector_roofline": int(simd_vector_roofline),
                "vector_compute": int(simd_vector_compute),
                "vector_load": int(simd_vector_load),
                "vector_store": int(simd_vector_store),
                "scalar_path": int(simd_scalar_path),
                "scalar_address": int(scalar_address_cost),
                "scalar_mask": int(scalar_mask_cost),
                "reduce_lowering": int(reduce_lowering_cost),
                "scalarization_penalty": int(scalarization_penalty),
                "aggregation": "max(vector_roofline, scalar_path)",
            },
            "simt": {
                "total": all_simt_only_cost,
                "compute_path": int(simt_compute + simt_shuffle),
                "memory_roofline": int(simt_memory_roofline),
                "compute": int(simt_compute),
                "load": int(simt_load),
                "store": int(simt_store),
                "predicate": int(simt_predicate),
                "shuffle": int(simt_shuffle),
                "divergence": int(simt_divergence),
                "atomic": int(simt_atomic),
                "shape": int(simt_shape),
                "setup": int(simt_setup),
                "aggregation": "max(compute + shuffle, memory_roofline) + predicate + divergence + atomic + shape + setup",
            },
            "mixed_simd_simt": {
                "total": mixed_simd_simt_cost,
                "simt_region": all_simt_only_cost,
                "boundary": int(boundary_cost),
                "dot_mixed_path": dot_mixed_path_cost,
                "available_in_p1": False,
            },
        },
        "margin": margin,
        "dot_margin": dot_margin if dot_mixed_eligible else None,
        "speedup_score": int(speedup_score),
        "confidence": confidence,
        "dot_mixed_eligible": dot_mixed_eligible,
        "whole_body_simt_eligible": non_dot_whole_body_eligible,
        "rank1_indirect_whole_body_eligible": rank1_indirect_whole_body_eligible,
        "atomic_whole_body_eligible": atomic_whole_body_eligible,
        "scan_whole_body_eligible": scan_whole_body_eligible,
        "reason": _auto_simt_scope_reason(
            features, eligible, choose_simt, speedup_score, margin, decision_kind
        ),
    }


def _auto_simt_scope_reason(
    features: Dict[str, Any],
    eligible: bool,
    choose_simt: bool,
    score: float,
    margin: float,
    decision_kind: str = "all_simd",
) -> str:
    if features["has_explicit_scope"]:
        return "explicit_scope_present"
    if features["has_dot"] and decision_kind == "mixed_simd_simt":
        return "dot_mixed_path_lower_than_simd_and_simt"
    if features["has_dot"]:
        return "dot_mixed_path_not_profitable"
    if (
        features["max_tensor_rank"] < 2
        and not features["rank1_indirect_vector_reduce"]
        and not features["has_atomic"]
        and not features["has_histogram"]
        and not features["has_scan"]
    ):
        return "rank1_or_scalar_kernel"
    if features["store_ops"] == 0 and not features["has_atomic"] and not features["has_histogram"]:
        return "no_store_op"
    if not eligible:
        return "no_unstructured_pointer_mask_or_reduce_risk"
    if choose_simt:
        if features["has_atomic"] or features["has_histogram"]:
            return "atomic_or_histogram_simt_route_lower_than_simd"
        if features["has_scan"]:
            return "scan_simt_route_lower_than_simd"
        return "simt_cost_lower_than_simd_with_margin"
    if features["rank1_indirect_vector_reduce"]:
        return f"rank1_indirect_vector_reduce_score_below_margin:{int(score)}<={int(margin)}"
    if features["has_scan"]:
        return f"scan_score_below_margin:{int(score)}<={int(margin)}"
    if features["has_atomic"] or features["has_histogram"]:
        return f"atomic_or_histogram_score_below_margin:{int(score)}<={int(margin)}"
    return f"score_below_margin:{int(score)}<={int(margin)}"


def _wrap_whole_body_void_simt_scope(ttir: str) -> str:
    if "scope.scope" in ttir:
        return ttir

    lines = ttir.splitlines()
    out = []
    in_func = False
    depth = 0
    wrapped = False
    body = []
    constants = []
    return_line = None
    func_body_indent = "  "

    for line in lines:
        if not in_func:
            out.append(line)
            if "tt.func" in line and "{" in line:
                in_func = True
                depth = _brace_delta(line)
            continue

        stripped = line.strip()
        if depth == 1 and stripped.startswith("}"):
            if body and return_line is not None:
                out.extend(constants)
                out.append(f"{func_body_indent}scope.scope : () -> () {{")
                out.extend(body)
                out.append(f"{func_body_indent}  scope.return")
                out.append(f'{func_body_indent}}} {{vec_mode = "simt"}}')
                out.append(return_line)
                wrapped = True
            else:
                out.extend(constants)
                out.extend(body)
                if return_line is not None:
                    out.append(return_line)
            out.append(line)
            in_func = False
            depth += _brace_delta(line)
            continue

        if depth == 1:
            if not stripped:
                constants.append(line)
            elif _is_allowed_top_level_constant(line):
                constants.append(line)
            elif stripped.startswith("tt.return"):
                return_line = line
            else:
                if not body:
                    indent_match = re.match(r"^(\s*)", line)
                    if indent_match:
                        func_body_indent = indent_match.group(1)
                body.append(line)
        else:
            body.append(line)

        depth += _brace_delta(line)

    if not wrapped:
        return ttir
    return "\n".join(out) + ("\n" if ttir.endswith("\n") else "")


def _dump_auto_simt_scope_report(report: Dict[str, Any], dump_path: str):
    if not dump_path:
        return
    try:
        path = Path(dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, sort_keys=True) + "\n")
    except OSError as exc:
        if os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE_VERBOSE"):
            print(f"[auto-simt-scope] failed to dump report to {dump_path}: {exc}")


def _maybe_apply_auto_simt_scope(ttir: str, metadata: dict, opt) -> str:
    mode = _normalize_auto_simt_scope_mode(getattr(opt, "auto_simt_scope_mode", ""))
    if mode == "off":
        return ttir
    if metadata.get("compile_mode") != "simd_simt":
        return ttir

    margin_ratio = getattr(opt, "auto_simt_scope_margin", 0.10)
    try:
        margin_ratio = float(margin_ratio)
    except (TypeError, ValueError):
        margin_ratio = 0.10

    features = _analyze_auto_simt_scope_features(ttir)
    decision = _estimate_auto_simt_scope_decision(features, margin_ratio, getattr(opt, "num_warps", 32))
    kernel_match = re.search(r"tt\.func\spublic\s+@(\w+)", ttir)
    report = {
        "kernel_name": kernel_match.group(1) if kernel_match else metadata.get("kernel_name", "<unknown>"),
        "mode": mode,
        "features": features,
        "decision": decision,
    }
    metadata["auto_simt_scope_report"] = report
    _dump_auto_simt_scope_report(report, getattr(opt, "auto_simt_scope_dump", ""))

    if mode != "auto" or not decision["choose_simt"]:
        return ttir

    scoped_ttir = _wrap_whole_body_void_simt_scope(ttir)
    if scoped_ttir == ttir:
        decision["choose_simt"] = False
        decision["reason"] = "failed_to_wrap_whole_body_scope"
        return ttir

    metadata["auto_simt_scope_applied"] = True
    return scoped_ttir


def get_common_bishengir_compile_options(metadata):
    bishengir_target = metadata['target'].arch
    bishengir_target_opt = f"--target={bishengir_target}"
    return [bishengir_target_opt]


def get_auto_bind_sub_block_option(metadata):
    # auto_tile_and_bind_subblock is read from the module.
    # enable_auto_bind_sub_block is set by the user and has a higher priority.
    enable_auto_bind_sub_block = metadata["enable_auto_bind_sub_block"]
    return (
        metadata["auto_tile_and_bind_subblock"]
        if enable_auto_bind_sub_block is None
        else enable_auto_bind_sub_block
    )


def _save_npuir_debug_output(stdout_bytes: bytes, stderr_bytes: bytes, tmpdir: str, metadata_hash: str):
    stdout = stdout_bytes.decode('utf-8') if stdout_bytes else ''
    stderr = stderr_bytes.decode('utf-8') if stderr_bytes else ''
    combined = stdout + stderr
    if not combined.strip():
        combined = "No output captured."
    output_path = os.path.join(tmpdir, "kernel.npuir.mlir")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined)

    dump_manager = get_dump_manager(metadata_hash)
    dump_manager.put(
        Path(output_path).read_text(encoding='utf-8'),
        "kernel.npuir.mlir",
        binary=False
    )


def try_compile_with_config(linalg: str, ub_config: Dict[str, Any], metadata: dict, opt) -> Tuple[bool, str]:
    """
    Try to compile with given UB config, return (success, error_msg).
    If compilation fails due to UB overflow or other errors, returns (False, error_msg).
    If compilation succeeds, returns (True, "").
    """
    # Must import from ubtuner.py - this is the single source of truth
    from triton.backends.ascend.runtime.ubtuner import UB_TO_NPU_OPTION_MAP
    option_mapping = UB_TO_NPU_OPTION_MAP

    metadata = dict(metadata)

    for ub_key, meta_key in option_mapping.items():
        if ub_key in ub_config:
            metadata[meta_key] = ub_config[ub_key]

    # Choose compile function based on options
    if opt.compile_on_910_95:
        compile_func = linalg_to_bin_enable_npu_compile_910_95
    else:
        compile_func = linalg_to_bin_enable_npu_compile_A2_A3

    try:
        compile_func(linalg, metadata, opt)
        return (True, "")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        return (False, error_msg)
    except Exception as e:
        return (False, str(e))


def linalg_to_bin_enable_npu_compile_910_95(linalg: str, metadata, opt):
    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(ttadapter_path).write_text(linalg)
        bin_file = os.path.join(tmpdir, "kernel")
        if _check_bishengir_api_change():
            bin_file_with_ext = "kernel.o"
        else:
            bin_file_with_ext = "kernel_reloc.o"
        bin_path = os.path.join(tmpdir, bin_file_with_ext)
        callback_path = os.path.join(tmpdir, "libkernel.so")
        _compile_option_list = get_common_bishengir_compile_options(metadata)

        multibuffer = metadata.get("multibuffer")
        num_stages = metadata.get("num_stages")

        if multibuffer is not None or num_stages is not None:
            multi_buffer_value = True
            if multibuffer is not None and not multibuffer:
                multi_buffer_value = False
            elif num_stages is not None and num_stages == 1:
                multi_buffer_value = False
            _compile_option_list += [
                f"--enable-auto-multi-buffer={multi_buffer_value}",
            ]

        storage_align = metadata["storage_align"]
        if storage_align is not None:
            _compile_option_list += [
                f"--enable-hivm-auto-storage-align={storage_align}",
            ]

        ops_reorder = metadata["ops_reorder"]
        if ops_reorder is not None:
            _compile_option_list += [
                f"--enable-ops-reorder={ops_reorder}",
            ]

        vf_fusion_mode = metadata["vf_fusion_mode"]
        if vf_fusion_mode is not None:
            _compile_option_list += [
                f"--vf-fusion-mode={vf_fusion_mode}",
            ]

        code_motion = metadata["code_motion"]
        if code_motion is not None:
            _compile_option_list += [
                f"--enable-code-motion={code_motion}",
            ]

        disable_tightly_coupled_buffer_reuse = metadata["disable_tightly_coupled_buffer_reuse"]
        if disable_tightly_coupled_buffer_reuse:
            _compile_option_list += ["--disable-tightly-coupled-buffer-reuse"]

        _compile_option_list += [
            f"--enable-auto-bind-sub-block={get_auto_bind_sub_block_option(metadata)}",
        ]

        if force_disable_ffts():
            _compile_option_list += ["--disable-ffts"]
        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        if not _is_debug_line_info_disabled():
            _compile_option_list += ["--enable-debug-info=true"]

        if _enable_print_ub_bits():
            _compile_option_list += ["--enable-print-memory-allocated-size"]

        enable_hivm_auto_cv_balance = metadata["enable_hivm_auto_cv_balance"]
        if enable_hivm_auto_cv_balance is not None:
            _compile_option_list += \
                [f"--enable-hivm-auto-cv-balance={enable_hivm_auto_cv_balance}"]

        sync_solver = metadata["sync_solver"]
        if sync_solver is not None:
            _compile_option_list += \
                [f"--enable-hivm-graph-sync-solver={sync_solver}"]

        unit_flag = metadata["unit_flag"]
        if unit_flag is not None:
            _compile_option_list += \
                [f"--enable-hivm-unit-flag-sync={unit_flag}"]

        inject_barrier_all = metadata["inject_barrier_all"]
        if inject_barrier_all is not None:
            _compile_option_list += \
                [f"--enable-hivm-inject-barrier-all-sync={inject_barrier_all}"]

        inject_block_all = metadata["inject_block_all"]
        if inject_block_all is not None:
            _compile_option_list += \
                [f"--enable-hivm-inject-block-all-sync={inject_block_all}"]

        limit_auto_multi_buffer_only_for_local_buffer = metadata["limit_auto_multi_buffer_only_for_local_buffer"]
        if limit_auto_multi_buffer_only_for_local_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-only-for-local-buffer={limit_auto_multi_buffer_only_for_local_buffer}"]

        set_workspace_multibuffer = metadata["set_workspace_multibuffer"]
        if set_workspace_multibuffer is not None:
            _compile_option_list += \
                [f"--set-workspace-multibuffer={set_workspace_multibuffer}"]

        auto_multi_buffer = metadata["limit_auto_multi_buffer_of_local_buffer"]
        if auto_multi_buffer is None:
            auto_multi_buffer = "no-limit"
        _compile_option_list += \
            [f"--limit-auto-multi-buffer-of-local-buffer={auto_multi_buffer}"]
        auto_multi_buffer_buffer = metadata["limit_auto_multi_buffer_buffer"]
        if auto_multi_buffer_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-buffer={auto_multi_buffer_buffer}"]

        enable_mixed_cv = metadata["enable_mixed_cv"]
        if enable_mixed_cv is not None:
            _compile_option_list += \
                [f"--enable-mixed-cv={enable_mixed_cv}"]

        enable_cce_vf_auto_sync = metadata["enable_cce_vf_auto_sync"]
        if enable_cce_vf_auto_sync is not None:
            _compile_option_list += \
                [f"--append-bisheng-options=-mllvm --cce-vf-auto-sync={enable_cce_vf_auto_sync}"]

        enable_cce_vf_remove_membar = metadata["enable_cce_vf_remove_membar"]
        if enable_cce_vf_remove_membar is not None:
            _compile_option_list += \
                [f"--append-bisheng-options=-mllvm --cce-vf-remove-membar={enable_cce_vf_remove_membar}"]

        enable_vf_fusion = metadata["enable_vf_fusion"]
        if enable_vf_fusion is not None:
            _compile_option_list += \
                [f"--enable-vf-fusion={enable_vf_fusion}"]

        enable_drop_unit_dims = metadata["enable_drop_unit_dims"]
        if enable_drop_unit_dims is not None:
            _compile_option_list += \
                [f"--enable-drop-unit-dims={enable_drop_unit_dims}"]

        enable_flatten = metadata["enable_flatten"]
        if enable_flatten is not None:
            _compile_option_list += \
                [f"--enable-flatten={enable_flatten}"]

        enable_auto_vectorize_v2 = metadata["enable_auto_vectorize_v2"]
        if enable_auto_vectorize_v2 is not None:
            _compile_option_list += \
                [f"--enable-auto-vectorize-v2={enable_auto_vectorize_v2}"]
        auto_vectorize_v2_max_fused_ops_num = metadata["auto_vectorize_v2_max_fused_ops_num"]
        if auto_vectorize_v2_max_fused_ops_num is not None:
            _compile_option_list += \
                [f"--hfusion-max-fused-ops-in-auto-vectorize-v2={auto_vectorize_v2_max_fused_ops_num}"]
        prevec_max_fused_ops_num = metadata["prevec_max_fused_ops_num"]
        if prevec_max_fused_ops_num is not None:
            _compile_option_list += \
                [f"--hfusion-max-fused-elementwise-ops={prevec_max_fused_ops_num}"]

        disable_auto_inject_block_sync = metadata["disable_auto_inject_block_sync"]
        if disable_auto_inject_block_sync is not None:
            _compile_option_list += \
                [f"--disable-auto-inject-block-sync={disable_auto_inject_block_sync}"]

        bitcodes = metadata["bitcodes"]
        if bitcodes is not None:
            for bitcode in bitcodes:
                _compile_option_list += \
                    [f"--link-aicore-bitcode={bitcode}"]

        if _is_auto_map_parallel_blocks_enabled() and not metadata.get("has_auto_blockify_blacklist_op", False):
            _compile_option_list += ["--enable-auto-blockify-loop"]
        npu_compiler_path, env = _get_npucompiler_path()
        if npu_compiler_path.endswith("bishengir-compile"):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                # CANN 9.1's hivmc-a5 cannot translate hacc.noinline yet.
                "--enable-lib-call-no-inline=false",
                "--enable-triton-kernel-compile=true",
            ]
        bisheng_options = metadata["bisheng_options"]
        if bisheng_options is not None:
            _compile_option_list += [
                f"--append-bisheng-options={bisheng_options}"
            ]
        mix_mode = opt.mix_mode
        if mix_mode in ["aic"]:
            _compile_option_list += ["--disable-hfusion-vectorize=true"]

        if opt.debug:
            _compile_option_list += ["--bishengir-print-ir-after=hivm-graph-sync-solver"]

        cmd_list = (
            [npu_compiler_path, ttadapter_path]
            + _compile_option_list
            + ["-o", bin_file]
        )
        vf_merge_level = metadata["vf_merge_level"]
        if vf_merge_level is not None:
            cmd_list += [f"--enable-vf-merge-level={vf_merge_level}"]

        hfusion_enable_multiple_consumer_fusion = metadata["hfusion_enable_multiple_consumer_fusion"]
        if hfusion_enable_multiple_consumer_fusion:
            cmd_list += [f"--hfusion-enable-multiple-consumer-fusion={hfusion_enable_multiple_consumer_fusion}"]

        if opt.debug or os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
            print(f"[DEBUG] cmd_list: {' '.join(cmd_list)}")

        try:
            ret = subprocess.run(
                cmd_list,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            if opt.debug:
                _save_npuir_debug_output(e.stdout, e.stderr, tmpdir, metadata["hash"])
            raise

        if opt.debug:
            _save_npuir_debug_output(ret.stdout, ret.stderr, tmpdir, metadata["hash"])

        stdout_str = ret.stdout.decode('utf-8') if ret.stdout else ''
        match = re.search(r'UB\s+size\s*=\s*(\d+)\s*bits', stdout_str)
        if match:
            # get the ub bits of triton kernel from bisheng for inductor autotune using
            metadata["required_ub_bits"] = int(match.group(1))

        if not Path(bin_path).exists():
            error_msg = ret.stderr.decode('utf-8') if ret.stderr else ''
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(ret.returncode, cmd_list, ret.stdout, ret.stderr)

        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            __get_metadata_attr_by_callback(lib, "_infer_task_type_function", metadata, "bs_task_type")
            __get_metadata_attr_by_callback(lib, "_infer_workspace_shape_function", metadata, "workspace_size")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_num_function", metadata, "lock_num")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val")

        return Path(bin_path).read_bytes()


def linalg_to_bin_enable_npu_compile_A2_A3(linalg: str, metadata, opt):
    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(ttadapter_path).write_text(linalg)
        bin_file = os.path.join(tmpdir, "kernel")
        if _check_bishengir_api_change():
            bin_file_with_ext = "kernel.o"
        else:
            bin_file_with_ext = "kernel_reloc.o"
        if _check_bishengir_is_regbased():
            bishengir_hivm_opt = "--reg-based=true"
        else:
            bishengir_hivm_opt = "--enable-hivm-compile=true"
        bin_path = os.path.join(tmpdir, bin_file_with_ext)
        callback_path = os.path.join(tmpdir, "libkernel.so")
        _compile_option_list = [
            f"--target={NPUUtils().get_arch()}",
        ]

        multibuffer = metadata.get("multibuffer")
        num_stages = metadata.get("num_stages")

        if multibuffer is not None or num_stages is not None:
            multi_buffer_value = True
            if multibuffer is not None and not multibuffer:
                multi_buffer_value = False
            elif num_stages is not None and num_stages == 1:
                multi_buffer_value = False
            _compile_option_list.append(f"--enable-auto-multi-buffer={multi_buffer_value}")

        enable_ubuf_saving = metadata["enable_ubuf_saving"]
        if enable_ubuf_saving is not None:
            _compile_option_list += [
                f"--enable-ubuf-saving={enable_ubuf_saving}",
            ]

        storage_align = metadata["storage_align"]
        if storage_align is not None:
            _compile_option_list += [
                f"--enable-hivm-auto-storage-align={storage_align}",
            ]

        ops_reorder = metadata["ops_reorder"]
        if ops_reorder is not None:
            _compile_option_list += [
                f"--enable-ops-reorder={ops_reorder}",
            ]

        code_motion = metadata["code_motion"]
        if code_motion is not None:
            _compile_option_list += [
                f"--enable-code-motion={code_motion}",
            ]

        enable_preload = metadata["enable_preload"]
        if enable_preload is not None:
            _compile_option_list += [
                f"--enable-preload={enable_preload}",
            ]

        _compile_option_list += [
            f"--enable-auto-bind-sub-block={get_auto_bind_sub_block_option(metadata)}",
        ]

        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        if not _is_debug_line_info_disabled():
            _compile_option_list += ["--enable-debug-info=true"]

        if _enable_print_ub_bits():
            _compile_option_list += ["--enable-print-memory-allocated-size"]

        if _enable_dump_memory_info():
            _compile_option_list += ["--enable-memory-display=true"]

        enable_hivm_auto_cv_balance = metadata["enable_hivm_auto_cv_balance"]
        if enable_hivm_auto_cv_balance is not None:
            _compile_option_list += \
                [f"--enable-hivm-auto-cv-balance={enable_hivm_auto_cv_balance}"]

        sync_solver = metadata["sync_solver"]
        if sync_solver is not None:
            _compile_option_list += [
                f"--enable-hivm-graph-sync-solver={sync_solver}",
                f"--enable-hivm-cross-core-gss={sync_solver}",
            ]

        unit_flag = metadata["unit_flag"]
        if unit_flag is not None:
            _compile_option_list += \
                [f"--enable-hivm-unit-flag-sync={unit_flag}"]

        enable_drop_unit_dims = metadata["enable_drop_unit_dims"]
        if enable_drop_unit_dims is not None:
            _compile_option_list += \
                [f"--enable-drop-unit-dims={enable_drop_unit_dims}"]

        enable_flatten = metadata["enable_flatten"]
        if enable_flatten is not None:
            _compile_option_list += \
                [f"--enable-flatten={enable_flatten}"]

        enable_auto_vectorize_v2 = metadata["enable_auto_vectorize_v2"]
        if enable_auto_vectorize_v2 is not None:
            _compile_option_list += \
                [f"--enable-auto-vectorize-v2={enable_auto_vectorize_v2}"]

        inject_barrier_all = metadata["inject_barrier_all"]
        if inject_barrier_all is not None:
            _compile_option_list += \
                [f"--enable-hivm-inject-barrier-all-sync={inject_barrier_all}"]

        inject_block_all = metadata["inject_block_all"]
        if inject_block_all is not None:
            _compile_option_list += \
                [f"--enable-hivm-inject-block-all-sync={inject_block_all}"]

        limit_auto_multi_buffer_only_for_local_buffer = metadata["limit_auto_multi_buffer_only_for_local_buffer"]
        if limit_auto_multi_buffer_only_for_local_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-only-for-local-buffer={limit_auto_multi_buffer_only_for_local_buffer}"]

        set_workspace_multibuffer = metadata["set_workspace_multibuffer"]
        if set_workspace_multibuffer is not None:
            _compile_option_list += \
                [f"--set-workspace-multibuffer={set_workspace_multibuffer}"]

        tile_mix_vector_loop = metadata["tile_mix_vector_loop"]
        if tile_mix_vector_loop is not None:
            _compile_option_list += \
                [f"--tile-mix-vector-loop={tile_mix_vector_loop}"]

        tile_mix_cube_loop = metadata["tile_mix_cube_loop"]
        if tile_mix_cube_loop is not None:
            _compile_option_list += \
                [f"--tile-mix-cube-loop={tile_mix_cube_loop}"]

        auto_multi_buffer = metadata["limit_auto_multi_buffer_of_local_buffer"]
        if auto_multi_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-of-local-buffer={auto_multi_buffer}"]

        disable_auto_inject_block_sync = metadata["disable_auto_inject_block_sync"]
        if disable_auto_inject_block_sync is not None:
            _compile_option_list += \
                [f"--disable-auto-inject-block-sync={disable_auto_inject_block_sync}"]

        bitcodes = metadata["bitcodes"]
        if bitcodes is not None:
            for bitcode in bitcodes:
                _compile_option_list += \
                    [f"--link-aicore-bitcode={bitcode}"]

        enable_libdevice = os.getenv("TRITON_ENABLE_LIBDEVICE", False)
        if enable_libdevice:
            _compile_option_list += [f"--link-aicore-bitcode={get_libdevice()}"]

        disable_size_align_for_cast = metadata["disable_size_align_for_cast"]
        if disable_size_align_for_cast is not None:
            _compile_option_list += \
                [f"--disable-size-align-for-cast={disable_size_align_for_cast}"]

        if _is_auto_map_parallel_blocks_enabled() and not metadata.get("has_auto_blockify_blacklist_op", False):
            _compile_option_list += ["--enable-auto-blockify-loop"]
        npu_compiler_path, env = _get_npucompiler_path()
        if npu_compiler_path.endswith("bishengir-compile"):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                bishengir_hivm_opt,
                # CANN 9.1's hivmc-a5 cannot translate hacc.noinline yet.
                "--enable-lib-call-no-inline=false",
                "--enable-triton-kernel-compile=true",
            ]

        if opt.debug:
            _compile_option_list += ["--mlir-print-ir-after-failure"]
            _compile_option_list += ["--bishengir-print-ir-after=hivm-graph-sync-solver"]
        cmd_list = (
            [npu_compiler_path, ttadapter_path]
            + _compile_option_list
            + ["-o", bin_file]
        )
        if opt.debug or os.getenv("TRITON_PRINT_UBTUNING", None) == "1":
            print(f"[DEBUG] cmd_list: {' '.join(cmd_list)}")

        try:
            ret = subprocess.run(
                cmd_list,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            if opt.debug:
                _save_npuir_debug_output(e.stdout, e.stderr, tmpdir, metadata["hash"])
            raise

        if opt.debug:
            _save_npuir_debug_output(ret.stdout, ret.stderr, tmpdir, metadata["hash"])

        stdout_str = ret.stdout.decode('utf-8') if ret.stdout else ''
        match = re.search(r'UB\s+size\s*=\s*(\d+)\s*bits', stdout_str)
        if match:
            metadata["required_ub_bits"] = int(match.group(1))

        if not Path(bin_path).exists():
            error_msg = ret.stderr.decode('utf-8') if ret.stderr else ''
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(ret.returncode, cmd_list, ret.stdout, ret.stderr)

        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            __get_metadata_attr_by_callback(lib, "_infer_task_type_function", metadata, "bs_task_type")
            __get_metadata_attr_by_callback(lib, "_infer_workspace_shape_function", metadata, "workspace_size")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_num_function", metadata, "lock_num")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val")

        return Path(bin_path).read_bytes()


def get_libdevice():
    current = os.path.dirname(__file__)
    return os.path.join(current, "lib/libdevice.10.bc")


@dataclass(frozen=True)
class NPUOptions:
    debug: bool = False
    sanitize_overflow: bool = True
    llvm_version: int = 15
    kernel_name: str = "triton_"
    arch: str = ""

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = 32
    num_ctas: int = 1
    num_stages: int = 2
    warp_size: int = 32
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0

    auto_blockify_size: int = 1
    compile_on_910_95: bool = is_compile_on_910_95
    optimize_dynamic_offset: bool = False
    enable_mask_fallback_conversion: bool = False
    enable_warp_specialization: bool = False
    enable_nd2nz_on_vector: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    auto_tile_and_bind_subblock: bool = True
    vf_merge_level: int = 0
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15", "fp8e4nv", "fp8e4b8", "fp8e5b16")
    deprecated_fp8_dtypes: Tuple[str] = ()
    vf_merge_level: int = 1
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "hf32")
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = None
    bisheng_options: str = "-cce-link-aicore-ll-module " + get_libdevice()

    multibuffer: bool = True
    storage_align: bool = None
    ops_reorder: bool = None
    code_motion: bool = None
    vf_fusion_mode: str = None
    enable_ubuf_saving: bool = None
    enable_preload: bool = None
    enable_auto_bind_sub_block: bool = None
    disable_tightly_coupled_buffer_reuse: bool = False
    enable_select_analysis: bool = True
    enable_hivm_auto_cv_balance: bool = None
    sync_solver: bool = None
    unit_flag: bool = None
    enable_cce_vf_auto_sync: bool = None
    enable_cce_vf_remove_membar: bool = None
    enable_drop_unit_dims: bool = None
    enable_flatten: bool = None
    enable_auto_vectorize_v2: bool = None
    auto_vectorize_v2_max_fused_ops_num: int = None
    prevec_max_fused_ops_num: int = None
    inject_barrier_all: bool = None
    inject_block_all: bool = None
    disable_size_align_for_cast: bool = None
    limit_auto_multi_buffer_only_for_local_buffer: bool = None
    limit_auto_multi_buffer_of_local_buffer: str = None
    limit_auto_multi_buffer_buffer: str = None
    set_workspace_multibuffer: int = None
    tile_mix_vector_loop: int = None
    tile_mix_cube_loop: int = None
    disable_auto_inject_block_sync: bool = None
    enable_mixed_cv: bool = None
    enable_vf_fusion: bool = None
    enable_dynamic_cv_pipeline: bool = True if is_compile_on_910_95 else False
    # Gates the cube-loader penetration + cube-for block merge feature. Off by
    # default so existing scenarios are unaffected; opt in per kernel to fuse a
    # matmul's loader for-loop into the matmul's cube compute block.
    enable_cube_block_merge: bool = False
    hfusion_enable_multiple_consumer_fusion: bool = False
    has_auto_blockify_blacklist_op: Optional[bool] = None
    intra_cache_num: int = None
    inter_cache_num: int = None
    load_cache_num: int = None

    stream: int = None
    parallel_mode: str = "simd"
    force_simt_only: bool = False
    force_simt_template: bool = False
    enable_sync_block_lock: bool = False
    # only take effect on the simt-only & simd-simt-mix scenarios
    shared_mem_dynamic_size: int = None
    # enable_bishengir_simt_optimization is passed as
    # -enable-bishengir-simt-optimization flag to bishengir-compile.
    enable_bishengir_simt_optimization: int = 000
    # compile_mode: "simd" (default), "unstructured_in_simt", "simd_simt", "simt_only"
    # When compile_mode is provided, it automatically sets other fields
    compile_mode: str = "unstructured_in_simt"
    mix_mode: str = ""
    simt_stack_limit: int = None
    # take effect on the reorder instruction pattern for SIMT. The pattern is disabled by default.
    enable_simt_reorder_instruction: bool = False
    enable_costmodel_backend: bool = False
    # TRITON_ASCEND_AUTO_SIMT_SCOPE=off|report|auto controls the lightweight
    # TTIR cost model that can wrap profitable whole-body kernels in SIMT scope.
    auto_simt_scope_mode: str = ""
    auto_simt_scope_dump: str = ""
    auto_simt_scope_margin: float = 0.10
    # disable simt fma optimization to get high precision
    disable_fma: bool = False

    # superblocking factor
    superblock_factor: int = 1

    def __post_init__(self):
        auto_simt_mode = self.auto_simt_scope_mode or os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE", "")
        object.__setattr__(self, "auto_simt_scope_mode", _normalize_auto_simt_scope_mode(auto_simt_mode))
        auto_simt_dump = self.auto_simt_scope_dump or os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP", "")
        object.__setattr__(self, "auto_simt_scope_dump", auto_simt_dump)
        auto_simt_margin = (
            self.auto_simt_scope_margin
            if self.auto_simt_scope_margin is not None
            else _parse_float_env("TRITON_ASCEND_AUTO_SIMT_SCOPE_MARGIN", 0.10)
        )
        if os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE_MARGIN"):
            auto_simt_margin = _parse_float_env("TRITON_ASCEND_AUTO_SIMT_SCOPE_MARGIN", 0.10)
        try:
            auto_simt_margin = float(auto_simt_margin)
        except (TypeError, ValueError):
            auto_simt_margin = 0.10
        object.__setattr__(self, "auto_simt_scope_margin", auto_simt_margin)

        forced_compile_mode = os.environ.get("TRITON_ASCEND_COMPILE_MODE")
        if forced_compile_mode:
            object.__setattr__(self, "compile_mode", forced_compile_mode)

        # Parse compile_mode and set related fields
        if self.compile_mode == "simd":
            object.__setattr__(self, "parallel_mode", "simd")
        elif self.compile_mode == "unstructured_in_simt":
            # For historical compatibility reasons, force_simt_template will still be used.
            object.__setattr__(self, "force_simt_template", True)
        elif self.compile_mode == "simd_simt":
            object.__setattr__(self, "force_simt_template", True)
            object.__setattr__(self, "parallel_mode", "mix_simd_simt")
        elif self.compile_mode == "simt_only":
            object.__setattr__(self, "force_simt_only", True)
            object.__setattr__(self, "parallel_mode", "simt")

        if self.force_simt_only:
            if self.shared_mem_dynamic_size is None:
                object.__setattr__(self, "shared_mem_dynamic_size", 122880)
        else:
            object.__setattr__(self, "shared_mem_dynamic_size", 221184)

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        key = "_".join([key, get_cann_version_file_hash()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


@register_descriptor
class AscendAttrsDescriptor(AttrsDescriptor):

    # For now we collect shapes of tensor at runtime.
    # We comment out the following func but keep it for future reference.
    def _add_backend_properties(self, params=None, values=None):
        pass


def ttir_to_npubin(mod, metadata, opt):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    if opt.force_simt_only:
        metadata["force_simt_only"] = True
        metadata["parallel_mode"] = "simt"
        metadata["shared_mem_dynamic_size"] = opt.shared_mem_dynamic_size
        ttir_code = _inline_void_simt_scopes_for_pure_simt(ttir_code)
    metadata = _parse_ttir_metadata(ttir_code, metadata)
    with tempfile.TemporaryDirectory() as tmpdir:
        # prepare input
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        Path(src_path).write_text(ttir_code)
        # prepare output
        bin_file = os.path.join(tmpdir, "kernel")
        bin_path = os.path.join(tmpdir, "kernel.o")
        # build compile options
        _compile_option_list = get_common_bishengir_compile_options(metadata)
        if opt.force_simt_only:
            _compile_option_list += ["--enable-hivm-compile=false"]
            _compile_option_list += ["--enable-triton-ir-compile"]
            _compile_option_list += ["--pure-simt"]
            _compile_option_list += [f"--num-warps={opt.num_warps}"]
            _compile_option_list += [f"--threads-per-warp={opt.warp_size}"]
            if opt.enable_bishengir_simt_optimization != 000:
                _compile_option_list += [f"--enable-bishengir-simt-optimization={opt.enable_bishengir_simt_optimization}"]
            if opt.simt_stack_limit:
                _compile_option_list += [f"--simt-stack-limit={opt.simt_stack_limit}"]
            if opt.shared_mem_dynamic_size is not None:
                _compile_option_list += [f"--shared-mem-dynamic-size={opt.shared_mem_dynamic_size}"]
            if opt.enable_simt_reorder_instruction:
                _compile_option_list += ["--enable-simt-reorder-instruction=true"]
            if opt.disable_fma:
                _compile_option_list += [f"--disable-fma"]

            enable_libdevice_simt = triton_enable_libdevice_simt()
            if (enable_libdevice_simt):
                bisheng_options = metadata["bisheng_options"]
                if bisheng_options is not None:
                    _compile_option_list += [
                        f"--append-bisheng-options={bisheng_options}"
                    ]

            # Enable SIMT auto-blockify when TRITON_ALL_BLOCKS_PARALLEL is set,
            # mirroring the SIMD compile paths. driver.py's runtime block-count
            # cap keys off the same env switch, so the two stay in sync.
            if _is_auto_map_parallel_blocks_enabled():
                _compile_option_list += ["--enable-auto-blockify-loop"]
                if opt.superblock_factor > 1:
                    _compile_option_list += [f"--super-block-factor={opt.superblock_factor}"]

        npu_compiler_path, env = _get_npucompiler_path()
        cmd_list = (
            [npu_compiler_path, src_path]
            + _compile_option_list
            + ["-o", bin_file]
        )
        ret = subprocess.run(cmd_list, env = env, capture_output = True, check = True)
        if not Path(bin_path).exists():
            error_msg = ret.stderr.decode('utf-8')
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(ret.returncode, cmd_list, ret.stdout, ret.stderr)
        return Path(bin_path).read_bytes()


class AscendBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "npu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        if target.backend == "npu":
            self.binary_ext = "npubin"

    def parse_options(self, opts) -> Any:
        # TODO: get available targets when building options?
        if self.target.backend == "npu":
            args = {
                k: opts[k]
                for k in NPUOptions.__dataclass_fields__.keys()
                if k in opts
            }
            args.setdefault("arch", self.target.arch)
            options = NPUOptions(**args)
            # Costmodel path should avoid extra BC<->MLIR conversion stages
            # to keep compile-only autotune routing lightweight and stable.
            if getattr(options, "enable_costmodel_backend", False):
                object.__setattr__(options, "use_bytecode", False)
        else:
            raise NotImplementedError(
                f"Backend '{self.target.backend}' is not supported. "
                "Please ensure the target backend is set to 'npu'."
            )
        return options

    def pack_metadata(self, metadata):
        # collect necessary metadata to launch kernels
        # TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 could set unique name.
        # Get this name as the kernel_name to CANN runtime.
        # kernel_name is unique to Ascend backend and should not be public.
        # CANN runtime limits the length of kernel name <= 50.
        # Considering '\n' is appended, thus the real kernel name <= 49.
        KERNEL_NAME_MAX_LEN = 49
        kernel_name_orig = metadata.kernel_name
        if len(kernel_name_orig) > KERNEL_NAME_MAX_LEN:
            kernel_name = kernel_name_orig[-KERNEL_NAME_MAX_LEN:]
        else:
            kernel_name = kernel_name_orig
        return {
            "kernel_name": kernel_name,
            "hash": metadata.hash,
            "debug": metadata.debug,
            "tensor_kinds": metadata.tensor_kinds,
        }

    def get_codegen_implementation(self):
        # Note: a dict of functions is required to generate vendor-specific code piecies
        #       e.g. convert custom types like fp8e4b15
        from triton.backends.ascend import _apply_ascend_patch
        _apply_ascend_patch()
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def load_dialects(self, ctx):
        ascend.load_dialects(ctx)

    def get_attrs_descriptor(self, params, args):
        return AscendAttrsDescriptor(params, args)

    def add_stages(self, stages, options):
        if self.target.backend == "npu":
            stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
            if options.force_simt_only:
                stages["npubin"] = (
                    lambda src, metadata: ttir_to_npubin(
                        src, metadata, options
                    )
                )
                return
            stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(
                src, metadata, options, named_ops=True
            )
            def make_scope_aware_npubin(compile_linalg):
                def scope_aware_npubin(src, metadata):
                    if metadata.get("scope_pure_simt_auto", False):
                        pure_options = replace(options)
                        object.__setattr__(pure_options, "force_simt_only", True)
                        object.__setattr__(pure_options, "parallel_mode", "simt")
                        object.__setattr__(pure_options, "shared_mem_dynamic_size", 122880)
                        return ttir_to_npubin(src, metadata, pure_options)
                    return compile_linalg(src, metadata, options)
                return scope_aware_npubin

            if options.compile_on_910_95:
                stages["npubin"] = make_scope_aware_npubin(
                    linalg_to_bin_enable_npu_compile_910_95
                )
            else:
                stages["npubin"] = make_scope_aware_npubin(
                    linalg_to_bin_enable_npu_compile_A2_A3
                )
        else:
            raise NotImplementedError(
                f"Backend '{self.target.backend}' is not supported. "
                "Please ensure the target backend is set to 'npu'."
            )

    @functools.lru_cache()
    def hash(self):
        # TODO fetch compiler version
        version_key = self.target
        return str(version_key)

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
