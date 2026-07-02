# Spatial Model Error Analysis

Date: 2026-07-02

## Objective

Analyze why the Stage 2 spatial model still has near-zero exact top-k performance after adding near-miss diagnostics. The task is a highly imbalanced spatial ranking problem. Primary diagnostic metric for this pass is neighbor hit@50; exact precision@50 remains the strict operational metric.

## Artifacts Created

- `analysis/analyze_spatial_model_errors.py`
- `analysis/spatial_model_error_summary_latest.csv`
- `analysis/spatial_model_error_by_district_latest.csv`
- `analysis/spatial_model_error_by_region_latest.csv`
- `analysis/spatial_model_error_by_target_time_latest.csv`

## Key Findings

The prediction artifact rank is global across the full holdout, not per forecast time. This matters because the global top-50 and a per-target-time operational top-50 answer different questions. The analyzer now reports both `artifact` and `per_target_time` rank scopes.

Per-target-time ranking still performs poorly, so the failure is not only a rank-definition artifact. Across all target times, per-target-time top-50 captures only 2 positives for 30m, 2 positives for 1h, and 3 positives for 2h.

| Horizon | Rank scope | Positives | Positive rank p50 | Positive rank p90 | Top50 TP | Top50 FP | Top50 missed positives | Precision@50 | Recall@50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30m | artifact | 50 | 3547.0 | 63080.4 | 0 | 50 | 50 | 0.000000 | 0.000000 |
| 30m | per_target_time | 50 | 482.5 | 733.6 | 2 | 5698 | 48 | 0.000351 | 0.040000 |
| 1h | artifact | 57 | 11086.0 | 39814.0 | 1 | 49 | 56 | 0.020000 | 0.017544 |
| 1h | per_target_time | 57 | 603.0 | 755.4 | 2 | 5648 | 55 | 0.000354 | 0.035088 |
| 2h | artifact | 55 | 15941.0 | 56966.8 | 0 | 50 | 55 | 0.000000 | 0.000000 |
| 2h | per_target_time | 55 | 601.0 | 757.6 | 3 | 5597 | 52 | 0.000536 | 0.054545 |

Score deciles show broad ranking signal but weak within-decile ordering. For 30m, the highest score decile has a base rate of 0.003471 versus the overall base rate of 0.000579, but exact top-50 still has zero true positives. This means the model can roughly separate broad risk bands but cannot order the top few zones precisely enough.

Scores are not calibrated as probabilities. Examples: 30m median score is 0.953603 while the observed base rate is 0.000579; 2h p95 score is 0.989835 while the observed base rate is 0.000648. Treat current scores as ranking scores only, not probabilities.

Geographic error concentration is material. Yau Tsim Mong is repeatedly missed despite high positive counts: 11 missed positives at 30m, 15 at 1h, and 22 at 2h in the per-target-time top-50 view. Kowloon West is the main regional miss cluster: 19 missed positives at 30m, 23 at 1h, and 25 at 2h.

Top-k false positives are concentrated in low-yield or ambiguous areas. For example, per-target-time top-50 false positives are heavily allocated to `Unknown` and Tuen Mun for 1h, and to `Unknown`, Tuen Mun, Kwun Tong, Sha Tin, and Yuen Long across horizons. This suggests the model is over-ranking historical or broad activity proxies without enough local discrimination.

## Interpretation

The current model has useful coarse spatial signal but insufficient exact-zone discrimination. The main problem is not that all positives are impossible to find; many positives appear within the same district or neighboring rings. The problem is that top-ranked zones within each target time are dominated by false positives from broad geographic priors.

The current target is extremely sparse at H3 resolution 9. Exact-zone precision@50 is a harsh metric under this label density, so model iteration should optimize a primary near-miss metric while retaining exact precision as a guardrail.

## Recommended Next Experiments

1. Train or select using a near-miss-aware target: exact positive plus ring-1 or same-district positives as a graded label. This directly aligns with the observed neighbor/district signal and should be evaluated against neighbor hit@50 and exact precision@50.

2. Add a district-conditioned ranking feature set. The missed-positive concentration in Yau Tsim Mong and Kowloon West suggests the model needs local competition features, such as district-relative hotspot distance, district-relative recent activity rank, and district x hour historical rates.

3. Penalize over-ranked false-positive districts. Add fold-safe features for recent false-positive pressure or down-rank zones/districts that consistently consume top-k slots without positives. Validate carefully to avoid leakage.

4. Calibrate output scores separately from ranking. Use isotonic or Platt calibration only after ranking improves; current scores are saturated and should not drive probability-facing UX.

5. Consider evaluating H3 resolution 8 as an auxiliary model objective. If exact operational action can tolerate larger cells, resolution 8 may reduce label sparsity enough to learn stable top-k ordering, while resolution 9 can remain the final display granularity.

## Next Primary Metric

Use neighbor hit@50 as the primary model-iteration metric, with exact precision@50 and district hit@50 as guardrails. Neighbor hit@50 is the best compromise between the current sparse exact labels and operationally useful spatial proximity.
