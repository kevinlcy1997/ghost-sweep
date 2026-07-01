# Spatial Context Feature Pack Design

## Goal

Improve Stage 2 zone ranking for the Ghost Sweep two-stage model by adding spatial context features suggested by the feature correlation analysis.

The immediate goal is better street/block-level ranking for 30m, 1h, and 2h horizons. Stage 1 activity timing stays unchanged in this iteration.

## Motivation

The feature correlation analysis showed strong Stage 1 recency signal but weak individual numeric signal for Stage 2. The strongest spatial signals are broad geography and historical activity. That means the next model improvement should add better spatial structure before changing model architecture.

## Feature Groups

### Ring-2 H3 Context

The current model has ring-1 neighbor activity. Add ring-2 features so the model can learn enforcement movement across nearby blocks:

- `ring2_event_count_24h`
- `ring2_event_count_7d`
- `ring2_active_zones_24h`
- `ring2_to_ring1_24h_ratio`

Ring-2 means cells at H3 grid distance exactly 2 from the current zone.

### Hotspot Distance

Add distance-style features from each candidate zone to recent historical event locations available before `target_time`:

- `distance_to_nearest_event_3h_m`
- `distance_to_nearest_event_24h_m`
- `distance_to_district_recent_centroid_24h_m`

If no matching event exists in the lookback window, use a large bounded sentinel distance so tree and linear models can handle the feature consistently.

### District-Relative Features

The correlation report showed strong district and region effects. Add district-relative features so the model learns whether a zone is unusually active within its district:

- `zone_24h_share_of_district`
- `zone_7d_rank_in_district`
- `zone_same_hour_percentile_in_district`

Ranking is computed within each `(target_time, district)` group.

### Road Context

Use existing road-coverage artifacts when available:

- `nearest_road_m`
- `road_segment_count`
- `road_source_mismatch`
- `has_drivable_road`

If the road coverage CSV is missing, features default safely and training still works. This keeps local experimentation robust while allowing richer road features when the artifact exists.

## Integration

Add a single feature-enrichment function in `ghost_ranking_features.py` and call it from `add_engineered_ranking_features`.

Update `analysis/run_zone_ranking_experiment.py` `NUMERIC_FEATURES` so both single-stage and two-stage spatial training consume the new columns, because `analysis/run_two_stage_experiment.py` imports those feature lists.

## Evaluation

Run focused tests first. Then run the two-stage experiment for 30m, 1h, and 2h.

Primary metrics:

- spatial precision@20
- spatial precision@50
- spatial average precision
- spatial top-decile lift
- district hit-rate@50
- region hit-rate@50

The results should be captured in Notion after training.

## Acceptance Criteria

- Tests cover ring-2, hotspot distance, district-relative, and road context behavior.
- The training feature table includes the new columns without leakage.
- Two-stage training runs for all current horizons.
- Spatial metadata includes the new feature names.
- A Notion checkpoint records what changed and the resulting metrics.
