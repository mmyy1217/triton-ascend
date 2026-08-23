---
name: costmodel-pipeline-debug
description: Use when debugging triton-ascend StageCostModel IR dumps, pass failures, unexpected route decisions, AutoBlockify V1 behavior, or SIMT scope materialization.
---

# Costmodel Pipeline Debug

## Pipeline stages and IR dumps

Set `TRITON_ASCEND_STAGECOST_IR_DUMP=<dir>` to dump intermediate TTIR after each stage. Files are named `<kernel_name>_<stage>.ttir`:

| File | Stage | When |
|------|-------|------|
| `<kernel>_after_layout_merge.ttir` | After Layout Merge (coalescing) | `compile_mode=simd_simt` + `auto_simt_scope != off` |
| `<kernel>_after_costmodel_scope.ttir` | After costmodel + scope materialization | Always when costmodel runs |
| `<kernel>_after_auto_blockify_v1.ttir` | After AutoBlockify V1 scheduling loop | When V1 materializes |
| `<kernel>_after_refine_superblock.ttir` | After SuperBlock factor refinement | When factor > 1 |

Additionally, the JSON report is available via `TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP=<file>`.

The JSON dump appends JSONL; StageCost IR files with the same kernel/stage name are overwritten. A compile-cache hit may skip the pass and produce no new dump.

The existing `opt.debug=True` dumps (coarser-grained):
- `kernel.ttir.mlir` — before any costmodel passes
- `kernel.ttadapter.mlir` — after all costmodel + triton_adapter lowering
- `kernel.mlir` — linalg level
- `kernel.npuir.mlir` — NPUIR level

## Debugging workflow

### 1. "Costmodel chose the wrong route"

1. Get the JSON report (`TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP=<file>`)
2. Check `candidate_costs` — are the scores plausible?
3. Dump IR before costmodel (`after_layout_merge`) and inspect the kernel structure
4. Check `features.post_transform` — Layout Merge should be visible; normal online scoring sees V1 as not yet materialized
5. If `stage_model.applied=true`, inspect `stage_model.logical_stages`; otherwise diagnose the aggregate analytical fallback

### 2. "Scope materialization failed or scopes disappeared"

1. Check `materialized_simt_anchor_count` in the report — if 0, no anchors were found
2. Dump `after_costmodel_scope.ttir` and search for `scope.scope` ops
3. Confirm each scope has `vector_mode = "simt"`; `MaterializeSimtScopes.cpp` writes this attribute only
4. If the scope disappears later, trace `TritonToLinalg` and downstream scope passes; the TTIR converter consumes `vector_mode`

### 3. "Layout Merge didn't fire"

1. Dump `after_layout_merge.ttir` and compare with `kernel.ttir.mlir`
2. If identical, Layout Merge didn't match any pattern
3. Check `features.post_transform.ttir_layout_merge_applied` and `features.post_transform.coalesce_factor` in the report
4. Layout Merge only runs when `compile_mode == "simd_simt" and auto_simt_scope_mode != "off"`
5. The four sub-passes (ImplicitPermute → StridedAxis → TileChunk → RowCoalescing) each match specific patterns — if none match, the kernel's access pattern isn't recognized

### 4. "AutoBlockify V1 didn't fire"

1. Check `metadata["ta_auto_blockify_v1_materialized"]` — if False, V1 skipped; this metadata is not a JSON report field
2. Common skip reasons: `has_auto_blockify_blacklist_op=True`, `enable_ta_auto_blockify_v1=False`
3. V1 only transforms entry kernels (`tt.func` with public visibility and no return values)
4. V1 requires `tt.get_program_id` ops in the function body

### 5. "Pass failure with error message"

1. If bishengir-compile fails, use `--mlir-print-ir-after-all --print-pass-id 2> pass.log`
2. Find the failing pass ID in `pass.log`, then read the IR right before it
3. For `.ttadapter` paste artifacts, prefer bytecode form (`.bcmlir`) over text form
4. Common paste artifacts: trailing `"` in loc, `to_tensor ... to tensor<T>` suffix, `arith.cmpi ... : <result-type>`

## Environment variables quick reference

| Variable | Effect |
|----------|--------|
| `TRITON_ASCEND_COMPILE_MODE=simd_simt` | Enable mix compilation mode |
| `TRITON_ASCEND_AUTO_SIMT_SCOPE=auto` | Enable costmodel auto selection |
| `TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP=<file>` | Dump JSON report to file (JSONL) |
| `TRITON_ASCEND_STAGECOST_IR_DUMP=<dir>` | Dump intermediate TTIR after each stage |
| `TRITON_ASCEND_AUTO_SIMT_PROFILE=<file>` | Override the SIMD/SIMT profile |

## Key source files

- `third_party/ascend/backend/compiler.py` — Python pass scheduling
  - `_dump_stagecost_ir()` — IR dump helper
  - `_run_ttir_layout_merge()` — Layout Merge entry
  - `_run_cpp_simd_simt_costmodel()` — Costmodel + scope materialization
  - `_run_ta_simt_auto_blockify_v1()` — AutoBlockify V1
  - `_refine_ta_simt_auto_blockify_v1_superblock()` — SuperBlock refinement
  - `ttir_to_linalg()` — Main pipeline orchestrator
- `third_party/ascend/costmodel/lib/AscendModel/RouteModel/` — C++ costmodel
  - `SelectSimdSimtCostModel.cpp` — Pass that runs costmodel + materializes scopes
  - `StagePartitioner.cpp` — Phase/Stage划分
  - `StageCostModels.cpp` — 20 kinds mapped to 9 model families per mode
  - `StageRouteCostModel.cpp` — Dynamic programming route solver
  - `SimdSimtCostModel.cpp` — Top-level analysis + scoring
- `third_party/ascend/lib/AutoBlockifyV1/SIMTAutoBlockifyV1.cpp` — AutoBlockify V1 pass
- `third_party/ascend/lib/TritonToLinalg/TTIRLayoutMergePass.cpp` — Layout Merge pass
- `third_party/ascend/costmodel/lib/AscendModel/RouteModel/Transforms/MaterializeSimtScopes.cpp` — Scope materialization
