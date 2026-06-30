# Two-Stage Zone Risk Model Design

## Goal

Improve the Hong Kong road-zone risk model by separating two questions that are currently mixed into one sparse classification problem:

1. Is enforcement activity likely anywhere in Hong Kong soon?
2. If activity is likely, which road-access zones should be watched?

The model should support the existing multi-horizon outputs for the next 30 minutes, 1 hour, and 2 hours.

## Current Problem

The current single-stage `(target_time, zone)` classifier treats every road-access zone at every hourly snapshot as a candidate. This creates a very sparse target because most zone-hour rows have no event. The final chronological holdout is technically time-safe, but the latest holdout can contain too few positives to judge model quality. Recent runs had only 5 positives for 30 minutes, 10 positives for 1 hour, and 19 positives for 2 hours across 11,398 holdout rows.

That makes metrics such as `precision@20` brittle. A useful model may look bad if the latest period is quiet, and a lucky model may look good if a few positives happen to land near the top.

## Proposed Model Shape

Use a two-stage design:

```text
P(zone z has event soon)
=
P(any event activity in HK soon)
*
P(zone z has event soon | activity exists)
```

Stage 1 is an activity timing model. It predicts whether Hong Kong has any event activity in the next horizon.

Stage 2 is a spatial ranking model. It scores every road-access zone for risk conditional on activity existing. It is not a single-label multi-class classifier because multiple zones can have events at the same time. It is a multi-label ranking problem where every zone can receive its own marginal risk score.

## Stage 1: Activity Timing

Stage 1 uses all valid hourly snapshots, including no-event hours. No-event hours are meaningful here because the model must learn when activity is unlikely.

The target is horizon-specific:

```text
activity_next_30m = any event in HK in [target_time, target_time + 30m)
activity_next_1h  = any event in HK in [target_time, target_time + 1h)
activity_next_2h  = any event in HK in [target_time, target_time + 2h)
```

Candidate features include hour, day of week, weekend flag, recent city-wide event counts, recent district activity counts, number of active districts, rolling same-hour activity rate, and recent trend ratios.

The output is a city-level activity probability per horizon.

## Stage 2: Spatial Ranking

Stage 2 scores road-access H3 zones. It should focus on active windows and hard negatives instead of letting no-event hours dominate.

Training rows remain `(target_time, zone_id)`, but sampling changes:

- Include positive zone rows from windows where events occurred.
- Include hard negative zones from the same active windows.
- Prefer negatives from nearby zones, same district, high-history zones, and recent hotspot zones that did not fire.
- Optionally include a small sample of inactive-window negatives so the ranking model still sees low-activity context, but do not let these dominate.

The target remains multi-label:

```text
zone_has_event_next_horizon = 1 if zone has one or more events in the horizon
```

The output is a per-zone risk score or probability-like value. These zone probabilities do not need to sum to 1 because multiple zones can be positive at the same time.

## Split Strategy

Use chronological splits only. Do not use random splits.

For model selection, use purged rolling-origin validation:

```text
Fold 1: [ train ] [ purge gap ] [ validation ]
Fold 2: [       train       ] [ purge gap ] [ validation ]
Fold 3: [             train             ] [ purge gap ] [ validation ]
```

The purge gap should be at least the horizon length. For example, the 2-hour model should have at least a 2-hour gap between training rows and validation rows. This avoids overlapping label windows.

For the final benchmark, use a positive-count holdout:

```text
[ earlier train data ][ latest contiguous holdout with enough positives ]
```

The final holdout should be the latest contiguous time period that satisfies a minimum positive-count requirement per horizon. If the latest operating period is too sparse, the window expands backward until the minimum is reached. The report must show exact holdout start/end timestamps, row count, positive count, and base rate for each horizon.

## Metrics

Stage 1 should report binary timing metrics:

- ROC AUC
- Average precision
- Brier score
- Expected calibration error
- Precision/recall at chosen city-activity thresholds
- Activity base rate

Stage 2 should report ranking and spatial metrics:

- Precision@20, @50, @100
- Recall@20, @50, @100
- Average precision
- Top-decile lift
- District hit-rate@50
- Region hit-rate@50
- Brier score and expected calibration error if probabilities are used

Final product-facing reports should show both city activity probability and top-zone risk. The dashboard should label zone outputs as marginal risk or risk score, not as exclusive class probability.

## Dashboard Implications

The dashboard should show:

- City activity probability by horizon.
- Top ranked zones by horizon.
- Zone marginal risk or score.
- Risk band.
- Exact split windows and positive counts used for the current model run.
- Stage 1 and Stage 2 metrics separately.

The map should continue to show multiple high-risk zones at the same time.

## Data Contract

New artifacts should be explicit:

- `activity_model_metadata_<horizon>.json`
- `activity_predictions_<horizon>_latest.csv`
- `spatial_model_metadata_<horizon>.json`
- `spatial_zone_predictions_<horizon>_latest.csv`
- `two_stage_summary_latest.csv`

The current single-stage artifacts can remain during migration, but the dashboard should prefer two-stage artifacts when they exist.

## Acceptance Criteria

The implementation is successful when:

- Stage 1 and Stage 2 can be trained for 30m, 1h, and 2h horizons.
- Split metadata records train, validation, purge, and final holdout windows.
- Final holdout has enough positives or explicitly reports that the minimum could not be met.
- Dashboard exposes both city activity and zone ranking outputs.
- Tests cover purged split behavior, positive-count holdout construction, active-window sampling, and multi-label zone outputs.

## Open Design Decision

The default minimum positive count should be chosen during implementation planning. A reasonable starting point is 50 positives per horizon for the final holdout, with a fallback that uses all available recent history if the dataset cannot satisfy that threshold.
