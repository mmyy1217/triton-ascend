---
name: costmodel-report-analysis
description: Use when inspecting triton-ascend StageCostModel JSON or JSONL reports, candidate scores, stage routes, recommended/effective decisions, or scope materialization outcomes.
---

# Costmodel Report Analysis

## How to get the report

Set env var `TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP=<file_path>` before running the kernel. The pass appends one JSON line per kernel to this file (JSONL format).

Alternatively, the report is stored as a module attribute `ascend.simt_costmodel.report_json` and accessible via `metadata["auto_simt_scope_report"]` in Python.

## Report structure

The JSON report (`SimdSimtCostReport`) contains these top-level fields:

```
schema_version         # report schema version (currently 14)
model                  # model name ("ascend_stage_route_cost_v3_cpp")
candidate_costs        # {all_simd, all_simt_only, mixed_simd_simt}
candidate_ratios_to_best
decision_kind          # all_simd | all_simt_only | mixed_simd_simt
best_score             # the winning score
features               # SimdSimtFeatureSummary (kernel characteristics)
compute_only / memory / structure / mixed / simt_execution / op_breakdown
stage_model             # StageCostModelSummary (logical phases/stages/routes)
mode                   # "report" or "auto"
recommended_decision_kind  # what the model recommends
effective_decision_kind    # what actually got applied (may differ from recommended)
selection_source           # "cpp_cost_model" | "backend_default" | "report_mode"
application_reason          # why this decision was applied (or not)
action_supported           # whether the recommended decision was materializable
materialized_simt_anchor_count  # selected materializable anchor roots
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

`candidate_costs` has three scores. Lower = better. `candidate_ratios_to_best` shows how far each is from the winner. If all three are within ~5%, the kernel is borderline and the decision may be sensitive to profile calibration.

### 3. Stage model details

`stage_model` contains:
- `applied` — false means Stage discovery missed and aggregate analytical fallback was used
- `domain` — `triangular_recurrence` / `loaded_index_rowwise_reduction` / `indirect_underfilled_dot`
- `logical_phases` / `logical_stages` — phase and flattened Stage arrays
- `transition_cost` — physical local-scope transition and UB handoff parameters
- `routes.all_simd` / `routes.all_simt_only` / `routes.mixed_simd_simt` — three StageRoutePlan objects, each containing:
  - `candidate` — route kind
  - `legal` — whether this route is structurally lowerable
  - `stages[].implementation` — per-Stage mode and factor
  - `stages[].entry_transition_system_cycles` / `logical_stage_system_cycles`
  - `logical_phase_system_cycles`
  - `route_superblock_factor` — selected superblock factor
  - `total_system_cycles` — total estimated cycles
  - `source` — how the route was computed

### 4. Feature summary

`features` (SimdSimtFeatureSummary) tells you what the model saw:
- `post_transform.ttir_layout_merge_applied` / `coalesce_factor` / `coalesce_axis`
- `post_transform.auto_blockify_v1_applied` / `auto_blockify_v1_loop_count` / `auto_blockify_v1_schedule_op_count`
- Normal online scoring runs before V1 materialization, so `auto_blockify_v1_applied` is normally false
- Operation counts by category (load/store/compute/control)

## Quick diagnosis checklist

1. Is `decision_kind` what you expected? If not, check `candidate_costs` to see if the scores make sense.
2. Is `effective_decision_kind` different from `recommended_decision_kind`? Check `application_reason`.
3. For mixed decisions, check `materialized_simt_anchor_count` — if 0, scope materialization failed.
4. For performance issues, compare `candidate_ratios_to_best` — if the margin is thin, the profile may need recalibration.
5. Check `features.post_transform`; do not treat normal pre-V1 scoring as a preprocessing failure.
