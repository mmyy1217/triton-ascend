# Costmodel Report Analysis

Use when analyzing the JSON report produced by the StageCostModel SIMD/SIMT route selection pass. Trigger when the user mentions costmodel report, JSON report, route selection, candidate scores, stage cost, or wants to understand why the costmodel chose all_simd/all_simt_only/mixed.

## How to get the report

Set env var `TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP=<file_path>` before running the kernel. The pass appends one JSON line per kernel to this file (JSONL format).

Alternatively, the report is stored as a module attribute `ascend.simt_costmodel.report_json` and accessible via `metadata["auto_simt_scope_report"]` in Python.

## Report structure

The JSON report (`SimdSimtCostReport`) contains these top-level fields:

```
schemaVersion          # report schema version (currently 14)
model                  # model name ("ascend_stage_route_cost_v3_cpp")
candidateCosts         # {allSimd, allSimtOnly, mixedSimdSimt} — raw cycle scores
candidateRatiosToBest  # ratios of each candidate to the best score
decision               # all_simd | all_simt_only | mixed_simd_simt
bestScore              # the winning score
features               # SimdSimtFeatureSummary (kernel characteristics)
breakdown              # SimdSimtCostBreakdown (detailed cost terms)
stageModel             # StageCostModelSummary (phases, stages, routes)
mode                   # "report" or "auto"
recommended_decision_kind  # what the model recommends
effective_decision_kind    # what actually got applied (may differ from recommended)
selection_source           # "cpp_cost_model" | "backend_default" | "report_mode"
application_reason          # why this decision was applied (or not)
action_supported           # whether the recommended decision was materializable
materialized_simt_anchor_count  # number of scope.scope regions materialized
selected_superblock_factor     # superblock factor for the chosen route
```

## Key analysis paths

### 1. Decision mismatch: recommended != effective

When `recommended_decision_kind != effective_decision_kind`, the costmodel recommended one route but the backend applied another. Check `application_reason`:

- `explicit_scope_present` — kernel already has explicit scope, model defers
- `no_materializable_mixed_anchor` — mixed selected but no anchors found
- `scope_superblock_not_materializable` — mixed F2/F4 not yet supported
- `superblock_warp_limit_exceeded` — factor × num_warps > 64
- `superblock_requires_auto_blockify_v1` — all_simt_only F2/F4 needs V1
- `candidate_not_materializable` — generic fallback

### 2. Score comparison

`candidateCosts` has three scores. Lower = better. `candidateRatiosToBest` shows how far each is from the winner. If all three are within ~5%, the kernel is borderline and the decision may be sensitive to profile calibration.

### 3. Stage model details

`stageModel` contains:
- `domain` — control-flow domain classification (e.g., `scalar_control_flow`, `loop_nest`, `reduction_tree`)
- `phases` — LogicalPhaseCost array (phase-level costs)
- `stages` — LogicalStageCost array (stage-level costs within phases)
- `transition` — StageTransitionCost (cost of crossing stage boundaries)
- `allSimd` / `allSimt` / `mixed` — three StageRoutePlan objects, each containing:
  - `candidate` — route kind
  - `legal` — whether this route is structurally lowerable
  - `implementations` — per-stage implementation choices
  - `entryTransitionCycles` / `logicalStageCycles` / `logicalPhaseCycles`
  - `routeSuperblockFactor` — superblock factor selected for this route
  - `totalCycles` — total estimated cycles
  - `source` — how the route was computed

### 4. Feature summary

`features` (SimdSimtFeatureSummary) tells you what the model saw:
- `ttirLayoutMergeApplied` / `coalesceFactor` / `coalesceAxis` — Layout Merge state
- `autoBlockifyV1Applied` / `autoBlockifyV1LoopCount` / `autoBlockifyV1ScheduleOpCount` — AutoBlockify V1 state
- `numWarps` / `threadsPerWarp` — execution config
- Operation counts by category (load/store/compute/control)

## Quick diagnosis checklist

1. Is `decision` what you expected? If not, check `candidateCosts` to see if the scores make sense.
2. Is `effective_decision_kind` different from `recommended_decision_kind`? Check `application_reason`.
3. For mixed decisions, check `materialized_simt_anchor_count` — if 0, scope materialization failed.
4. For performance issues, compare `candidateRatiosToBest` — if the margin is thin, the profile may need recalibration.
5. Check `features.ttirLayoutMergeApplied` and `features.autoBlockifyV1Applied` to confirm preprocessing ran.
