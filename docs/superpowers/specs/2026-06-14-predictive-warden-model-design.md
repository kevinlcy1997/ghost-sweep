# Ghost Sweep — Predictive Warden Activity Model

**Date:** 2026-06-14
**Status:** Draft
**Scope:** Phase 1 — data pipeline, feature engineering, baseline model

---

## 1. Problem Statement

The 走鬼 Ghost Alert system collects real-time traffic warden sighting reports from the Hong Kong community. The goal is to predict **where and when** wardens will appear in the next 1–4 hours, based on historical sighting patterns.

This is a spatiotemporal binary classification problem: for each grid cell × time window, predict the probability of warden activity.

## 2. Approach

**Algorithm:** LightGBM (gradient boosted trees) — fast training, handles categorical features natively, small memory footprint, interpretable via feature importance.

**Spatial unit:** 0.005° grid cells (~500 m, roughly 2–3 street blocks) for prediction. The existing poll sweep in `ghost_listener.py` uses a coarser 0.05° step for API queries — the finer prediction grid subdivides each API cell into 100 prediction cells. Approximately 11,600 cells covering the Hong Kong bounding box (lat 22.15–22.56, lng 113.83–114.41). Note: most cells will be empty (water, mountains); only cells with historical activity are scored during prediction.

**Zoning:** Each grid cell is mapped to one of Hong Kong's **21 police districts** across **5 police regions**. District assignment is based on official HK Police Force boundaries. Districts are used as categorical features so the model learns zone-specific patrol patterns (e.g., "Mong Kok is most active weekday mornings" vs "Wan Chai peaks at evening").

### 2.1 Police District Structure

Source: [HK Police Force — Police Districts](https://www.access.gov.hk/mobile/en/howtomakeinfo/hkpf_district.html)

| Region | Districts |
|--------|-----------|
| Hong Kong Island | Eastern, Wan Chai, Central, Western |
| Kowloon East | Wong Tai Sin, Kwun Tong, Tseung Kwan O, Sau Mau Ping |
| Kowloon West | Yau Tsim, Mong Kok, Sham Shui Po, Kowloon City |
| New Territories North | Tai Po, Tuen Mun, Yuen Long, Border |
| New Territories South | Tsuen Wan, Kwai Tsing, Sha Tin, Airport, Lantau |

Railway District is excluded — it covers the MTR network territory-wide (not a geographic patrol zone). Traffic and Crime units are also specialized and excluded. This leaves **21 geographic districts** used for spatial classification.

**Boundary data:** Source district boundary polygons from the HK government open data portal (data.gov.hk) as GeoJSON. Each grid cell's centroid is point-in-polygon tested to assign its district. Cells that fall outside all district polygons (e.g., open water) are assigned to the nearest district.

## 3. Data Pipeline

### 3.1 Current State

`ghost_listener.py` polls 5 API endpoints every N seconds and writes de-duplicated alerts to `ghost_alerts.json`. Each alert has: `alert_record_id`, `lat`, `lng`, `address`, `alert_type`, `create_dt`, `upvote`, `downvote`, `_source`, `_first_seen`, `_last_seen`.

### 3.2 New: SQLite Database (`ghost_db.py`)

Flat JSON is not suited for time-series ML queries. A SQLite database provides efficient slicing by time, location, and district.

**Table: `sightings`**

| Column | Type | Description |
|--------|------|-------------|
| `alert_record_id` | TEXT PK | Unique alert ID |
| `lat` | REAL | Latitude |
| `lng` | REAL | Longitude |
| `address` | TEXT | Street address |
| `alert_type` | TEXT | Alert classification |
| `create_dt` | TEXT | Alert creation timestamp |
| `upvote` | INTEGER | Community upvotes |
| `downvote` | INTEGER | Community downvotes |
| `grid_cell` | TEXT | Computed cell ID (e.g., `22.30_114.15`) |
| `district` | TEXT | Police district name |
| `region` | TEXT | Police region name |
| `first_seen` | TEXT | UTC ISO timestamp of first observation |
| `last_seen` | TEXT | UTC ISO timestamp of last observation |

**Table: `poll_cycles`**

| Column | Type | Description |
|--------|------|-------------|
| `cycle_id` | INTEGER PK | Auto-increment |
| `timestamp` | TEXT | UTC ISO timestamp |
| `total_alerts` | INTEGER | Alerts found this cycle |
| `new_alerts` | INTEGER | New alerts this cycle |
| `duration_sec` | REAL | Cycle duration |

**Indexes:** `(grid_cell, create_dt)`, `(district, create_dt)`, `(create_dt)`

**Migration:** One-time import of existing `ghost_alerts.json` into SQLite.

**Live ingestion:** After each poll cycle, `ghost_listener.py` writes to both JSON (backward compat) and SQLite via `ghost_db.insert_cycle()`.

### 3.3 Data Cleaning (`ghost_clean.py`)

Raw alerts are community reports — multiple people often report the same warden within minutes from slightly different positions. Without cleaning, the model overcounts activity in popular areas.

**Event consolidation:** Collapse raw alerts into unique warden events via spatiotemporal clustering.

- **Spatial threshold:** 20 meters (haversine distance) — same street corner
- **Temporal threshold:** 15 minutes
- **Algorithm:** Greedy clustering. Sort alerts by `create_dt`, iterate. For each alert, check if it falls within 20m AND 15min of any existing event cluster's latest report. If yes, merge. If no, start a new cluster.

Each cluster produces one **event record** in the `events` table:

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | INTEGER PK | Auto-increment |
| `lat` | REAL | Centroid latitude of all alerts in cluster |
| `lng` | REAL | Centroid longitude of all alerts in cluster |
| `address` | TEXT | Address from alert with highest upvotes |
| `create_dt` | TEXT | Earliest `create_dt` in cluster |
| `end_dt` | TEXT | Latest `create_dt` in cluster |
| `duration_min` | REAL | Time span of cluster |
| `report_count` | INTEGER | Number of raw alerts in cluster |
| `total_upvotes` | INTEGER | Sum of upvotes across cluster |
| `total_downvotes` | INTEGER | Sum of downvotes across cluster |
| `grid_cell` | TEXT | Computed from centroid |
| `district` | TEXT | Police district from centroid |
| `region` | TEXT | Police region from centroid |

**Additional cleaning rules:**

| Issue | Rule |
|-------|------|
| Spam/trolling | Exclude alerts with 0 upvotes AND ≥3 downvotes |
| Missing coordinates | Drop alerts where lat=0 or lng=0 |
| Off-territory | Drop alerts outside the HK bounding box |

**Pipeline order:**

```
ghost_listener.py → sightings table (raw) → ghost_clean.py → events table → ghost_features.py
```

The model trains on **events**, not raw alerts. Raw alerts are preserved in `sightings` for auditing.

### 3.4 Grid Cell Assignment

Each event's centroid `(lat, lng)` is snapped to a grid cell ID by rounding to the nearest 0.005° step:

```
cell_lat = round(lat / 0.005) * 0.005
cell_lng = round(lng / 0.005) * 0.005
cell_id = f"{cell_lat:.3f}_{cell_lng:.3f}"
```

This produces ~500m cells — roughly 2–3 street blocks. Fine enough to be actionable for users.

### 3.5 District Assignment

Each grid cell's centroid is mapped to a police district via point-in-polygon lookup against the district boundary GeoJSON. Since only cells with historical activity matter, the mapping is computed lazily and cached — new cells are looked up on first encounter.

Fallback: If official GeoJSON boundaries are unavailable, use nearest-known-district assignment based on proximity to district station coordinates.

## 4. Feature Engineering (`ghost_features.py`)

For each `(grid_cell, time_window)` pair, compute the following features:

### 4.1 Temporal Features

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0–23) |
| `day_of_week` | Monday=0 to Sunday=6 |
| `is_weekend` | Boolean |
| `month` | 1–12 |

### 4.2 Cell-Level Features

| Feature | Description |
|---------|-------------|
| `cell_historical_freq` | Total events ever in this cell |
| `cell_7d_count` | Events in this cell in last 7 days |
| `cell_24h_count` | Events in this cell in last 24 hours |
| `cell_last_seen_hours_ago` | Hours since last sighting in this cell |
| `neighbor_24h_count` | Sum of events in adjacent 8 cells in last 24h |
| `streak_active` | 1 if cell had events on consecutive days |
| `upvote_ratio` | Avg upvote/(upvote+downvote) in this cell |
| `avg_report_count` | Avg reports per event in this cell (community density signal) |

### 4.3 District-Level Features

| Feature | Description |
|---------|-------------|
| `district` | Police district name (categorical, LightGBM native) |
| `region` | Police region name (categorical) |
| `district_24h_count` | Total events in this district in last 24h |
| `district_historical_rate` | Average daily events for this district |
| `district_active_cells` | Number of cells in this district with events in last 24h |
| `district_hour_rate` | Average events at this hour for this district |

### 4.4 Cross Features

| Feature | Description |
|---------|-------------|
| `hour_historical_rate` | Average events at this hour across all days |
| `dow_hour_rate` | Average events for this day-of-week + hour combo |

### 4.5 Label (Target Variable)

`has_warden` — binary: was there at least one event in this cell in the next 1-hour window?

Separate label columns for 2h and 4h windows can be generated for multi-horizon models.

## 5. Model Training (`ghost_model.py`)

### 5.1 Training Split

Time-based split to prevent data leakage:
- **Train:** First 80% of chronological data
- **Validation:** Next 10%
- **Test:** Final 10%

No random shuffling — predictions must only use past data.

### 5.2 Minimum Data Gate

Refuse to train until at least **14 continuous days** of data have been collected. Display a message with the current accumulation progress and estimated date when training can begin.

### 5.3 Class Imbalance

Most cell × time-window pairs will have no warden activity (heavy class imbalance). Handle with:
- LightGBM's `is_unbalance=True` parameter
- Optionally: undersampling negative examples at 5:1 ratio

### 5.4 Hyperparameters

Start with LightGBM defaults, then tune via validation set:
- `num_leaves`: 31
- `learning_rate`: 0.05
- `n_estimators`: 500 (with early stopping on validation loss)
- `categorical_feature`: `[district, region]`

### 5.5 Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| AUC-ROC | Overall discrimination ability |
| Precision@k | Of top-k predicted cells, how many are actually active? |
| Recall | Of actual active cells, how many did we catch? |
| F1 | Balance of precision and recall |

### 5.6 Retraining

Weekly automatic retrain as new data accumulates. Model serialized as `.joblib`.

### 5.7 Feature Importance

After training, log and save the top-10 feature importances. This reveals whether the model relies more on temporal patterns, spatial history, or district-level signals — useful for understanding warden behavior.

## 6. Prediction Output (`ghost_predict.py`)

### 6.1 Forecast Generation

Given the current time and a trained model:
1. Build feature vectors for all active grid cells (cells with any historical events) for the next time window
2. Run inference to get probability scores
3. Rank cells by probability
4. For high-risk cells, include recent event coordinates for street-level pinpointing
5. Output forecast JSON

### 6.2 Output Format (`ghost_forecast.json`)

```json
{
  "generated_at": "2026-06-14T15:00:00Z",
  "forecast_window": "1h",
  "model_version": "2026-06-14",
  "training_samples": 12450,
  "cells": [
    {
      "cell": "22.315_114.170",
      "lat": 22.315,
      "lng": 114.170,
      "district": "Mong Kok",
      "region": "Kowloon West",
      "probability": 0.82,
      "risk": "high",
      "top_features": ["dow_hour_rate", "district_24h_count"],
      "recent_events": [
        {"lat": 22.3154, "lng": 114.1698, "address": "37 Dundas St", "create_dt": "2026-06-14T11:37:50Z", "report_count": 5},
        {"lat": 22.3148, "lng": 114.1702, "address": "42 Sai Yeung Choi St", "create_dt": "2026-06-14T10:15:22Z", "report_count": 3}
      ]
    }
  ]
}
```

The `recent_events` array gives users street-block-level detail within each predicted cell — showing the exact spots where wardens have been sighted recently.

### 6.3 Risk Tiers

| Tier | Probability | Meaning |
|------|-------------|---------|
| High | ≥ 0.7 | Strong likelihood of warden activity |
| Medium | 0.4–0.7 | Moderate likelihood |
| Low | < 0.4 | Low likelihood |

## 7. Integration with Existing System

### 7.1 Changes to `ghost_listener.py`

Minimal — after each poll cycle's `save_store()`, also call `ghost_db.insert_cycle()` to write to SQLite. No other changes.

### 7.2 CLI Interface (`ghost_predict.py`)

```
python ghost_predict.py train              # Train/retrain model from SQLite
python ghost_predict.py forecast           # Generate next-hour forecast
python ghost_predict.py forecast --hours 4 # Generate 4-hour forecast
python ghost_predict.py stats              # Show model performance metrics
python ghost_predict.py districts          # Show per-district activity summary
```

## 8. Project Structure

```
Ghost_Sweep/
├── ghost_listener.py          ← existing (minor addition: SQLite write)
├── ghost_db.py                ← SQLite schema, ingestion, migration
├── ghost_clean.py             ← data cleaning + event consolidation
├── ghost_features.py          ← feature engineering pipeline (reads events)
├── ghost_model.py             ← LightGBM training + evaluation
├── ghost_predict.py           ← CLI for forecasts + stats
├── ghost_districts.py         ← district boundary mapping + lookup
├── ghost_alerts.json          ← existing flat archive (kept)
├── ghost_alerts.db            ← new SQLite database
├── ghost_forecast.json        ← prediction output
├── data/
│   └── hk_police_districts.geojson  ← district boundary polygons
├── models/
│   └── model_latest.joblib    ← serialized trained model
└── requirements.txt           ← add: lightgbm, scikit-learn, pandas, shapely
```

## 9. Dependencies

| Package | Purpose |
|---------|---------|
| `lightgbm` | Gradient boosted tree model |
| `scikit-learn` | Metrics, train/test split, preprocessing |
| `pandas` | DataFrame operations for feature engineering |
| `shapely` | Point-in-polygon for district assignment |
| `joblib` | Model serialization (bundled with sklearn) |

Existing: `cryptography`, `requests`

## 10. Future Phases (Out of Scope)

- **Phase 2:** Real-time prediction integrated into the poll loop
- **Phase 3:** Weather data integration (rain reduces outdoor enforcement)
- **Phase 4:** Deep learning upgrade (LSTM) when sufficient data accumulated
- **Phase 5:** Web dashboard with heatmap visualization
- **Phase 6:** Per-district models for high-data zones, global model as fallback
