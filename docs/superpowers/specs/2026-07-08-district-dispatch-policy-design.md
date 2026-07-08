# District Dispatch Policy Design

**Date:** 2026-07-08  
**Status:** Drafted for review  
**Scope:** Next bounded spatial-ranking experiment after the rejected near-miss ranker and rejected cold-zone district backfill

## Goal

Test whether a **district-aware dispatch ranking policy** can improve top-k operational outcomes without changing the current winning model family (`lightgbm_conservative`) or score semantics.

## Context

Recent model-side experiments have already been tried and rejected:

- plain ranker
- near-miss ranker
- hard-negative over-sampling
- score-semantic changes
- cold-zone district backfill

The current accepted baseline still comes from the same scoring pipeline:

- train the two-stage pipeline as-is
- keep `spatial_probability`, `probability`, and `score` unchanged
- generate `dispatch_rank` as a separate operational ordering

The remaining gap is operational concentration: the current dispatch policy limits **target-time concentration**, but it does not limit **district concentration** inside a target time.

## Recommendation

Run the next experiment as a **dispatch-only reranker policy**, not a model experiment.

Specifically:

1. Keep the spatial model selection path unchanged.
2. Keep probability and score generation unchanged.
3. Extend dispatch ranking so the early dispatch set cannot be monopolized by multiple zones from the same district within the same target time.

This is the smallest meaningful district-aware candidate-set policy because it reuses the accepted dispatch-ranking seam instead of introducing another feature pack or retraining rule.

## Non-goals

This experiment must **not**:

- change `lightgbm_conservative`
- add or remove model candidates
- change training labels
- change feature engineering
- change `spatial_probability`
- change `probability`
- change `score`
- add another district-prior feature experiment in the same pass

## Proposed behavior

### Current behavior

`assign_dispatch_rank()` sorts rows by `probability`, then selects the first pass using only a per-target-time quota. If the top rows for a target time come from the same district, they can monopolize the early dispatch ranks.

### Proposed behavior

Extend `assign_dispatch_rank()` to enforce two caps in the first-pass selection:

1. **Per-target-time quota** — existing behavior, unchanged
2. **Per-(target_time, district) quota** — new rule for district coverage

The proposed district quota for this experiment is:

- `1` row per `(target_time, district)` in the first pass

After the first pass:

- append the remaining rows in original score order
- compute `dispatch_rank` exactly as today
- compute `dispatch_score` exactly as today

This preserves the current score contract while introducing district diversity only in the operational dispatch ordering.

## Components and file boundaries

### `analysis/run_two_stage_experiment.py`

This is the only production-code file expected to change.

Planned changes:

- add a small helper like `_dispatch_district_quota_for_target(target_col: str) -> int`
- extend `assign_dispatch_rank()` to accept a district quota
- count district usage with `district.fillna("Unknown")`
- keep the existing target-time quota behavior intact
- keep output column names unchanged

### `tests/test_two_stage_experiment.py`

Add focused tests for:

- existing target-time concentration limiting still working
- district coverage inside one target time
- no behavioral drift when district quota is irrelevant

## Data flow

1. The holdout prediction frame is created exactly as today with:
   - `target_time`
   - `zone_id`
   - `district`
   - `region`
   - `spatial_probability`
   - `activity_probability`
   - `probability`

2. `assign_dispatch_rank()` sorts by `probability`.

3. During the first-pass dispatch selection:
   - if a target time is already full, skip
   - if a district inside that target time is already represented at quota, skip
   - otherwise, select the row

4. After first-pass selection, append all skipped rows in score order.

5. Emit:
   - `dispatch_rank`
   - `dispatch_score`

6. Reuse the existing summary and error-analysis scripts.

## Error handling

- Missing district values are treated as `"Unknown"` for quota counting.
- Empty prediction frames keep the current no-op behavior.
- Horizons where the district quota adds no effect (for example, `2h` when target-time quota is already `1`) should still run through the same code path without special branching beyond the quota values.

## Testing strategy

### Focused tests

Run the existing focused suite covering:

- `tests/test_two_stage_experiment.py`
- `tests/test_ghost_ranking_features.py`
- `tests/test_ranking_metrics.py`
- `tests/test_spatial_sampling.py`
- `tests/test_spatial_model_error_analysis.py`
- `tests/test_spatial_ranking_diagnostics.py`

### New unit tests

Add tests that prove:

1. a same-target-time, same-district cluster does not monopolize early dispatch ranks when another district has the next-best score
2. target-time quota still dominates if the target-time quota is already exhausted
3. district quota does not change ordering when all top rows are already from distinct districts

### Full experiment checks

After unit tests pass:

1. rerun `analysis/run_two_stage_experiment.py`
2. rerun `analysis/analyze_spatial_model_errors.py --k 50`
3. inspect:
   - `analysis/two_stage_summary_latest.csv`
   - `analysis/spatial_model_error_summary_latest.csv`
   - `analysis/spatial_model_error_by_district_latest.csv`

## Acceptance gate

Keep the experiment only if **both** are true:

1. **Dispatch precision@50 or artifact top50 precision improves in at least 2 horizons**
2. **30m artifact top50 precision stays at or above 0.10**

Secondary diagnostic only, not a hard gate:

- district-level miss and false-positive concentration in `analysis/spatial_model_error_by_district_latest.csv` should look less concentrated if the policy is helping

## Rollback rule

If the gate fails:

1. revert the policy change
2. rerun the focused validation suite
3. log the rejection in `WORKLOG.md`
4. move next to **district-hour priors**, not another district coverage variant in the same session

## Why this is the next smallest step

- It is smaller than another model experiment.
- It reuses the already-accepted dispatch seam.
- It directly targets district concentration, which the current policy does not address.
- It does not repeat the failed cold-zone backfill or near-miss-ranker lines of work.
