# Costmodel Pipeline Debug

Use when debugging the StageCostModel compilation pipeline — IR dumps, pass failures, scope materialization issues, or unexpected routing decisions. Trigger when the user mentions IR dump, pass failure, scope materialization, or wants to trace the compilation flow step by step.

## Pipeline stages and IR dumps

Set `TRITON_ASCEND_STAGECOST_IR_DUMP=<dir>` to dump intermediate TTIR after each stage. Files are named `<kernel_name>_<stage>.ttir`:

| File | Stage | When |
|------|-------|------|
| `<kernel>_after_layout_merge.ttir` | After Layout Merge (coalescing) | `compile_mode=simd_simt` + `auto_simt_scope != off` |
| `<kernel>_after_costmodel_scope.ttir` | After costmodel + scope materialization | Always when costmodel runs |
| `<kernel>_after_auto_blockify_v1.ttir` | After AutoBlockify V1 scheduling loop | When V1 materializes |
| `<kernel>_after_refine_superblock.ttir` | After SuperBlock factor refinement | When factor > 1 |

Additionally, the JSON report is available via `TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP=<file>`.

The existing `opt.debug=True` dumps (coarser-grained):
- `kernel.ttir.mlir` — before any costmodel passes
- `kernel.ttadapter.mlir` — after all costmodel + triton_adapter lowering
- `kernel.mlir` — linalg level
- `kernel.npuir.mlir` — NPUIR level

## Debugging workflow

### 1. "Costmodel chose the wrong route"

1. Get the JSON report (`TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP=<file>`)
2. Check `candidateCosts` — are the scores plausible?
3. Dump IR before costmodel (`after_layout_merge`) and inspect the kernel structure
4. Check `features` — did Layout Merge / AutoBlockify V1 run as expected?
5. If scores look wrong, the issue is in the costmodel's StageCostModel evaluation — check `stageModel.stages` for per-stage costs

### 2. "Scope materialization failed or scopes disappeared"

1. Check `materialized_simt_anchor_count` in the report — if 0, no anchors were found
2. Dump `after_costmodel_scope.ttir` and search for `scope.scope` ops
3. If scopes exist in TTIR but disappear in NPUIR output, check bishengir's `InlineScopePass` and `AutoScopePass` — they check `vector_type` attribute (not `vector_mode`)
4. The fix in `MaterializeSimtScopes.cpp` writes both `vector_mode` and `vector_type` attributes to ensure compatibility

### 3. "Layout Merge didn't fire"

1. Dump `after_layout_merge.ttir` and compare with `kernel.ttir.mlir`
2. If identical, Layout Merge didn't match any pattern
3. Check `features.ttirLayoutMergeApplied` and `features.coalesceFactor` in the report
4. Layout Merge only runs when `compile_mode == "simd_simt" and auto_simt_scope_mode != "off"`
5. The four sub-passes (ImplicitPermute → StridedAxis → TileChunk → RowCoalescing) each match specific patterns — if none match, the kernel's access pattern isn't recognized

### 4. "AutoBlockify V1 didn't fire"

1. Check `metadata["ta_auto_blockify_v1_materialized"]` — if False, V1 skipped
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
| `TRITON_ASCEND_AUTO_SIMT_SCOPE_MARGIN=0.05` | Adjust gain margin threshold (default 0.10) |
| `FORCE_COSTMODEL_MIXROUTE=1` | Force costmodel to select mixed (bypass gate) |

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
  - `StageCostModels.cpp` — 19 StageCostModel types
  - `StageRouteCostModel.cpp` — Dynamic programming route solver
  - `SimdSimtCostModel.cpp` — Top-level analysis + scoring
- `third_party/ascend/lib/AutoBlockifyV1/SIMTAutoBlockifyV1.cpp` — AutoBlockify V1 pass
- `third_party/ascend/lib/TritonToLinalg/TTIRLayoutMergePass.cpp` — Layout Merge pass
- `third_party/ascend/costmodel/lib/AscendModel/RouteModel/Transforms/MaterializeSimtScopes.cpp` — Scope materialization
