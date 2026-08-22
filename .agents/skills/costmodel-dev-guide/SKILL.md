# Costmodel Dev Guide

Use when developing or modifying the StageCostModel — adding new StageCostModel types, modifying route solving, adding new features, or extending the pipeline. Trigger when the user wants to modify the costmodel, add a new cost model, change routing logic, or add new analysis passes.

## Architecture overview

The StageCostModel V2 pipeline runs entirely at TTIR level, before triton_adapter lowering:

```
TTIR (make_ttir output)
  │
  ├─ 1. Layout Merge          (TTIRLayoutMergePass)
  │     ImplicitPermute → StridedAxis → TileChunk → RowCoalescing → CSE
  │
  ├─ 2. AutoBlockify V1       (SIMTAutoBlockifyV1.cpp)
  │     Wraps function body in scf.for, replaces tt.get_program_id
  │
  ├─ 3. Costmodel Pass        (SelectSimdSimtCostModel)
  │     3.1 Feature extraction  (analyzeSimdSimtFeatures)
  │     3.2 Stage partition    (StagePartitioner::partition)
  │     3.3 Cost evaluation    (StageCostModels — 19 types)
  │     3.4 Route solving       (solveStageRoutes — dynamic programming)
  │     3.5 Decision + scope materialization
  │
  └─ 4. Post-processing       (metadata export, attribute cleanup)
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
2. Implement the scoring logic in `StageCostModels.cpp` — follows the pattern of existing models
3. Register the new type in `StagePartitioner.cpp` if it affects stage partitioning
4. Add JSON serialization in the struct's `toJSON()` method
5. The route solver (`solveStageRoutes` in `StageRouteCostModel.cpp`) automatically picks up new types via the cost table

## How to modify route solving

The route solver (`solveStageRoutes`) uses dynamic programming:
1. For each stage, evaluate all three route kinds (AllSIMD / AllSIMTOnly / Mixed)
2. Compute transition costs between adjacent stages
3. Find the minimum-cost path through the stage graph
4. The result is a `StageRoutePlan` with per-stage `StageImplementation` entries

To modify: edit `StageRouteCostModel.cpp`. The cost table is built by `StagePartitioner` from `StageCostModels`.

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
- Use `FORCE_COSTMODEL_MIXROUTE=1` to force mixed route for testing scope materialization
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
