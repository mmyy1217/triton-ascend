---
name: costmodel-dev-guide
description: Use when modifying the triton-ascend StageCostModel, including StageCostModel kinds, feature extraction, route solving, scope materialization, or compiler pipeline integration.
---

# Costmodel Dev Guide

## Architecture overview

The StageCostModel V2 pipeline runs entirely at TTIR level, before triton_adapter lowering:

```
TTIR (make_ttir output)
  │
  ├─ 1. Layout Merge          (TTIRLayoutMergePass)
  │     ImplicitPermute → StridedAxis → TileChunk → RowCoalescing → CSE
  │
  ├─ 2. V1 policy resolution  (_resolve_auto_blockify_v1_policy)
  │     Resolves availability only; does not rewrite TTIR
  │
  ├─ 3. Costmodel Pass        (SelectSimdSimtCostModel)
  │     3.1 Feature extraction  (analyzeSimdSimtFeatures)
  │     3.2 Stage partition    (StagePartitioner::partition)
  │     3.3 Cost evaluation    (20 kinds; 9 model families per mode)
  │     3.4 Route solving       (solveStageRoutes — dynamic programming)
  │     3.5 Decision + scope materialization
  │
  └─ 4. Post-processing
        all-SIMT only: materialize AutoBlockify V1, then refine SuperBlock
        mixed: keep selected local anchor scopes for downstream lowering
```

## Key data structures

### SimdSimtCostReport (SimdSimtCostModel.h:326)
Top-level report struct. Contains candidate costs, decision, features, breakdown, and StageCostModelSummary. Serialized to JSON via `toJSON()` / `printJSON()`.

### SimdSimtFeatureSummary (SimdSimtCostModel.h:188)
Kernel characteristics extracted before scoring. Key fields:
- `ttirLayoutMergeApplied` / `coalesceFactor` / `coalesceAxis`
- `autoBlockifyV1Applied` / `autoBlockifyV1LoopCount` / `autoBlockifyV1ScheduleOpCount`
- `autoBlockifyV1HasDynamicTripCount`

### StageCostModelSummary (StageRouteCostModel.h:214)
Stage-level analysis result. Contains:
- `domain` — control-flow domain classification
- `phases` (LogicalPhaseCost[]) — phase-level costs
- `stages` (LogicalStageCost[]) — stage-level costs within phases
- `transition` (StageTransitionCost) — cost of crossing stage boundaries
- `allSimd` / `allSimt` / `mixed` (StageRoutePlan) — three route plans with implementations, cycles, and superblock factors

### StageRoutePlan (StageRouteCostModel.h:200)
Per-route plan with:
- `candidate` — route kind (AllSIMD / AllSIMTOnly / MixedSimdSimt)
- `legal` — structurally lowerable
- `implementations` (StageImplementation[]) — per-stage implementation choices
- `totalCycles` — estimated total cycles
- `routeSuperblockFactor` — selected superblock factor

## How to add a new StageCostModel type

1. Add the enum value to `StageCostModelKind` in `StageCostModels.h`
2. Update `stringifyStageCostModel` / `parseStageCostModel`
3. Extend an existing SIMD/SIMT model family or register a paired family in `StageCostModels.cpp`
4. Update Stage partition/classification only when the new kind needs new discovery evidence
5. Add unit/lit coverage; JSON serialization follows the generic Stage structures

## How to modify route solving

The route solver (`solveStageRoutes`) uses dynamic programming over `(exit mode, route class, route superblock factor)`:
1. Enumerate legal SIMD/SIMT implementations and F1/F2/F4 factors per Stage
2. Keep the cheapest prefix for each state
3. For mixed routes, charge each physical local scope's two directional transitions and exact UB tensor handoff
4. Produce AllSIMD / AllSIMTOnly / Mixed plans with per-Stage implementations

Adjacent Stage labels do not add hardware transition cost (`entryTransitionCycles` is currently zero). `StageCostEvaluator`, not `StagePartitioner`, builds the cost table.

## How to add a new feature

1. Add the field to `SimdSimtFeatureSummary` in `SimdSimtCostModel.h`
2. Populate it in `analyzeSimdSimtFeatures` in `SimdSimtCostModel.cpp`
3. Add JSON serialization in `SimdSimtFeatureSummary::toJSON()`
4. If the feature affects scoring, consume it in the relevant `StageCostModel`

## How to add a new pipeline stage

1. Implement the pass as a TableGen-defined Pass in `costmodel/include/AscendModel/Transforms/Passes.td`
2. Implement the pass in `costmodel/lib/AscendModel/Transforms/`
3. Register in `PassRegistration.cpp`
4. If it produces IR changes that should be dumpable, add a `_dump_stagecost_ir()` call in `compiler.py` after the pass runs
5. If it has report-worthy information, add fields to `SimdSimtCostReport` and serialize in `toJSON()`

## Debugging during development

- Use `TRITON_ASCEND_STAGECOST_IR_DUMP=<dir>` to dump TTIR after each stage
- Use `TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP=<file>` to get the JSON report
- For pass-level IR inspection, use `--mlir-print-ir-after-all` on bishengir-compile
- The costmodel C++ code can be built locally (pass-level verification only, no NPU runtime needed)

## Build

The costmodel is part of the triton-ascend build. Key CMake targets:
- `triton` — the main triton library (includes costmodel)
- `triton-mlir-opt` — standalone MLIR tool for pass testing

For wheel builds:
```bash
TRITON_BUILD_TUTORIALS=0 TRITON_INTERPRET=0 \
  TRITON_BUILD_WITH_CCACHE=1 \
  python setup.py bdist_wheel -- \
  --target triton --target triton-mlir-opt
```

For NPU testing, the wheel must be installed on `arm_npu` (real NPU + CANN environment).
