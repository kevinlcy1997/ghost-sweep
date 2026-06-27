# Hong Kong Road-Zone Risk Model Execution Plan

## Overview

Reshape Ghost Sweep from sparse custom grid-cell binary classification into a Hong Kong-specific road-zone risk ranking system. The practical output is a ranked map/list of road-adjacent zones most likely to see at least one alert in the next 1-2 hours. The implementation should preserve the current JSON-to-DB sync, feature engineering, HTML report, and MLflow tracking, while adding better spatial units, ranking targets, geography features, and ranking metrics.

## Current Code Anchors

- `ghost_utils.py:5` defines the current 0.005-degree snapped grid cell.
- `ghost_districts.py:42` assigns nearest Hong Kong police district/region.
- `ghost_features.py:8` builds one prediction row from historical events.
- `ghost_features.py:129` builds the current training table.
- `analysis/sync_json_to_db.py:49` salvages complete alert records from malformed JSON.
- `analysis/run_ml_experiment.py:48` configures MLflow local tracking.
- `analysis/run_ml_experiment.py:218` logs the current recent-event modeling window.

## Stack

- Python, SQLite, pandas, scikit-learn, LightGBM, MLflow.
- Add geospatial packages: `h3`, `geopandas`, `pyogrio`, `rtree` or `pygeos` if needed, and `osmnx` only if government road data is insufficient.
- Store derived geographic artifacts under `analysis/geo/` or `data/geo/` depending on whether they are generated artifacts or durable source files.

## Task 1: Freeze the Existing Baseline

Create a reproducible baseline before changing the target.

1. Run `analysis/sync_json_to_db.py` against `ghost_alerts.json`.
2. Run `ghost_predict.py clean`.
3. Run `analysis/run_ml_experiment.py`.
4. Confirm MLflow has one parent run and five child model runs with `FINISHED` status.
5. Save the current report and metrics as the baseline run name `baseline-grid-cell-next-window`.

Verification:

```powershell
py analysis/sync_json_to_db.py
py ghost_predict.py clean
py analysis/run_ml_experiment.py
py -m pytest -q
```

Expected evidence: `analysis/mlflow_tracking.db`, `analysis/mlruns/`, `analysis/best_experiment_model.joblib`, and an HTML report exist. Tests pass.

## Task 2: Add Road-Zone Spatial Units

Add a new module `ghost_zones.py` that supports H3 or road-linked zone IDs without removing the existing grid cell.

Implementation shape:

```python
def compute_h3_zone(lat: float, lng: float, resolution: int = 8) -> str:
    ...

def assign_zone(lat: float, lng: float) -> dict:
    return {
        "grid_cell": compute_grid_cell(lat, lng),
        "h3_zone": compute_h3_zone(lat, lng),
        "district": district,
        "region": region,
    }
```

Start with H3 resolution 8 as the default. Resolution 8 is coarse enough to reduce sparsity while still showing neighborhood-scale hotspots. Keep resolution configurable so later experiments can compare H3 8 vs 9.

Tests:

- `tests/test_ghost_zones.py`
- Validate stable H3 IDs for known Hong Kong coordinates.
- Validate zone assignment includes `grid_cell`, `h3_zone`, `district`, and `region`.

## Task 3: Add Hong Kong Geography Survey Data

Create a lightweight geography survey step to characterize where alerts occur.

Add script:

```text
analysis/survey_hk_geography.py
```

Outputs:

```text
analysis/geo/hk_zone_summary.csv
analysis/geo/hk_zone_summary.html
```

Survey should include:

- events per district and region
- events per H3 zone
- active zones per district
- top recurring zones
- zones with one-off events only
- zone sparsity distribution
- approximate urban-core concentration

If road network data is available, add road-adjacent filtering. If not, mark road-linking as deferred and use H3+distrct as the first practical spatial layer.

Verification:

```powershell
py analysis/survey_hk_geography.py
```

Expected evidence: CSV and HTML survey files with nonzero zones and all 21 districts represented where data exists.

## Task 4: Redefine the Target

Add a new training-table builder that predicts future alert presence by zone.

New target:

```text
alert_next_2h = at least one event in this zone within the next 2 hours
```

Optional secondary targets:

```text
event_count_next_2h
alert_next_1h
repeat_hotspot_next_24h
minutes_until_next_alert
```

Add module or functions:

```text
ghost_ranking_features.py
```

Core function:

```python
def build_zone_ranking_training_data(
    events: list[dict],
    zone_col: str = "h3_zone",
    forecast_hours: int = 2,
    lookback_days: int = 14,
) -> pd.DataFrame:
    ...
```

Rows should represent `zone_id + target_time`. The label must be built from future data only. Features must be built from past data only.

Tests:

- Synthetic events with known future events.
- Assert no leakage: a future event cannot affect features for an earlier row.
- Assert target is positive only when an event occurs inside the forecast horizon.

## Task 5: Engineer Practical Predictive Features

Add features in three groups.

Spatial history:

```text
zone_event_count_1h
zone_event_count_3h
zone_event_count_24h
zone_event_count_7d
zone_hours_since_last_event
neighbor_event_count_3h
neighbor_event_count_24h
district_event_count_3h
district_event_count_24h
district_active_zones_24h
```

Temporal pattern:

```text
hour
day_of_week
is_weekend
hour_bucket
zone_same_hour_rate
district_same_hour_rate
dow_hour_rate
```

Geography:

```text
district
region
h3_zone
zone_lat
zone_lng
is_urban_core
```

If road data is integrated:

```text
road_density
distance_to_major_road
road_class_mix
near_tunnel_or_bridge
```

Verification: feature table should include no future-derived columns except labels and should have lower sparsity than the 0.005-degree grid baseline.

## Task 6: Change Model Selection Around Ranking

Update `analysis/run_ml_experiment.py` or create `analysis/run_zone_ranking_experiment.py`.

Model candidates:

- historical district-hour baseline
- logistic regression balanced
- random forest balanced
- LightGBM balanced
- optional LightGBM ranker after the classifier baseline works

Primary metrics:

```text
precision_at_10
precision_at_20
recall_at_20
top_decile_lift
average_precision
roc_auc
```

Keep ROC-AUC, but do not optimize only for it. The product is a top-risk-zone list, so ranking quality matters more.

Tests:

- Unit test `precision_at_k`.
- Unit test `top_decile_lift`.
- Smoke test that experiment logging writes MLflow metrics with `precision_at_20`.

## Task 7: Update HTML + MLflow Reporting

The report should answer product questions:

- What are the top risky zones for the next 2 hours?
- Which districts dominate the risk?
- How much better is the model than a historical baseline?
- Which features drive ranking quality?
- How sparse is the target after moving from grid cells to zones?

Artifacts to log in MLflow:

```text
zone_feature_ranking_latest.csv
zone_predictions_latest.csv
zone_ranking_report_<timestamp>.html
best_zone_model.joblib
hk_zone_summary.csv
```

MLflow metrics:

```text
precision_at_10
precision_at_20
recall_at_20
top_decile_lift
average_precision
roc_auc
```

## Task 8: Add Map-Friendly Outputs

Create prediction output:

```text
ghost_zone_forecast.geojson
```

Each feature should include:

```text
zone_id
geometry
score
risk_rank
district
region
top_feature_reason
recent_events_24h
```

This lets the UI or dashboard render a risk heatmap without needing to understand model internals.

Verification:

- GeoJSON validates as a FeatureCollection.
- Feature count equals the number of scored zones.
- Top 20 zones have non-null score, district, and geometry.

## Task 9: Acceptance Criteria

The plan is complete when:

1. The current baseline remains reproducible and logged in MLflow.
2. A new H3/zone-based dataset is generated from the synced DB.
3. The target is `alert_next_2h`, built without future leakage.
4. At least four models including a historical baseline are evaluated.
5. The main ranking metric is `precision_at_20`.
6. MLflow displays model runs, metrics, params, and artifacts.
7. HTML report includes top zones, feature ranking, ranking metrics, and sparsity comparison.
8. A GeoJSON forecast artifact exists for map display.
9. `py -m pytest -q` passes.

## Execution Order

1. Freeze current MLflow baseline.
2. Add H3 zone assignment and tests.
3. Build geography survey report.
4. Build leakage-safe zone ranking target.
5. Add ranking metrics and tests.
6. Run baseline vs model comparison.
7. Generate HTML, CSV, model, and GeoJSON artifacts.
8. Verify MLflow UI and full test suite.

