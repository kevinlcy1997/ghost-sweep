# Spatial Context Feature Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add spatial context features to Stage 2 zone ranking and evaluate their impact on the two-stage multi-horizon model.

**Architecture:** Extend the existing `ghost_ranking_features.add_engineered_ranking_features` path so both the legacy zone-ranking runner and the two-stage runner consume the same new features. Road context is joined from the existing road coverage CSV when present; other features are computed from the zone-time feature frame.

**Tech Stack:** Python, pandas, h3, pytest, scikit-learn/LightGBM pipelines, existing Ghost Sweep analysis scripts.

---

### Task 1: Add Failing Tests For Spatial Feature Pack

**Files:**
- Modify: `tests/test_engineered_ranking_features.py`
- Test: `tests/test_engineered_ranking_features.py`

- [ ] **Step 1: Add tests for ring-2, district-relative, hotspot, and road-context features**

Append tests that build small H3 frames and assert:

```python
def test_add_engineered_ranking_features_adds_ring2_context():
    zone = compute_h3_zone(22.3154, 114.1698, resolution=8)
    ring1 = next(cell for cell in h3.grid_ring(zone, 1))
    ring2 = next(cell for cell in h3.grid_ring(zone, 2))
    rows = pd.DataFrame([
        {"target_time": pd.Timestamp("2026-06-20 06:00:00"), "zone_id": zone, "hour": 6, "day_of_week": 5, "zone_event_count_3h": 0, "zone_event_count_24h": 2, "zone_event_count_7d": 4, "district_event_count_3h": 0, "district_event_count_24h": 10},
        {"target_time": pd.Timestamp("2026-06-20 06:00:00"), "zone_id": ring1, "hour": 6, "day_of_week": 5, "zone_event_count_3h": 0, "zone_event_count_24h": 5, "zone_event_count_7d": 8, "district_event_count_3h": 0, "district_event_count_24h": 10},
        {"target_time": pd.Timestamp("2026-06-20 06:00:00"), "zone_id": ring2, "hour": 6, "day_of_week": 5, "zone_event_count_3h": 0, "zone_event_count_24h": 7, "zone_event_count_7d": 11, "district_event_count_3h": 0, "district_event_count_24h": 10},
    ])
    enhanced = add_engineered_ranking_features(rows)
    base = enhanced.loc[enhanced["zone_id"] == zone].iloc[0]
    assert base["neighbor_event_count_24h"] == 5
    assert base["ring2_event_count_24h"] == 7
    assert base["ring2_event_count_7d"] == 11
    assert base["ring2_active_zones_24h"] == 1
    assert base["ring2_to_ring1_24h_ratio"] == 7 / 5
```

Add equivalent tests for:

```python
def test_add_engineered_ranking_features_adds_district_relative_features()
def test_add_engineered_ranking_features_adds_hotspot_distance_features()
def test_add_engineered_ranking_features_can_join_road_context()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_engineered_ranking_features.py -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_features
```

Expected: FAIL because new columns do not exist.

### Task 2: Implement Feature Computation

**Files:**
- Modify: `ghost_ranking_features.py`
- Test: `tests/test_engineered_ranking_features.py`

- [ ] **Step 1: Add helper functions**

Add focused helpers for safe distance, ring-2 lookup, district-relative ranking, hotspot distance, and optional road context join.

- [ ] **Step 2: Wire helpers into `add_engineered_ranking_features`**

Ensure empty frames and frames missing optional columns still return all new numeric columns with default values.

- [ ] **Step 3: Run tests and verify GREEN**

Run:

```powershell
C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_engineered_ranking_features.py -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_features
```

Expected: all tests pass.

### Task 3: Include New Columns In Model Feature List

**Files:**
- Modify: `analysis/run_zone_ranking_experiment.py`
- Test: `tests/test_two_stage_experiment.py`

- [ ] **Step 1: Add new columns to `NUMERIC_FEATURES`**

Add:

```python
"ring2_event_count_24h",
"ring2_event_count_7d",
"ring2_active_zones_24h",
"ring2_to_ring1_24h_ratio",
"distance_to_nearest_event_3h_m",
"distance_to_nearest_event_24h_m",
"distance_to_district_recent_centroid_24h_m",
"zone_24h_share_of_district",
"zone_7d_rank_in_district",
"zone_same_hour_percentile_in_district",
"nearest_road_m",
"road_segment_count",
"road_source_mismatch",
"has_drivable_road",
```

- [ ] **Step 2: Add a metadata test**

Extend two-stage tests to assert that the spatial metadata feature list contains at least one new spatial context feature.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_engineered_ranking_features.py tests/test_two_stage_experiment.py -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_features
```

Expected: all tests pass.

### Task 4: Verify Broader Modeling Surface

**Files:**
- No code changes expected unless tests expose integration issues.

- [ ] **Step 1: Run focused model tests**

Run:

```powershell
C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_spatial_sampling.py tests/test_model_iteration.py tests/test_multi_horizon_iteration.py tests/test_two_stage_experiment.py tests/test_engineered_ranking_features.py -q -p no:cacheprovider --basetemp .pytest_tmp_spatial_features
```

Expected: all tests pass.

- [ ] **Step 2: Run two-stage training**

Run:

```powershell
C:\Users\Kevin\.local\bin\rtk.exe C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe analysis/run_two_stage_experiment.py
```

Expected: generates refreshed `analysis/two_stage_summary_latest.csv` and metadata JSON files.

### Task 5: Capture Results

**Files:**
- Notion: `Ghost Sweep Project Memory`

- [ ] **Step 1: Read latest summary**

Run:

```powershell
C:\Users\Kevin\.local\bin\rtk.exe read analysis/two_stage_summary_latest.csv
```

- [ ] **Step 2: Write a Notion checkpoint**

Capture changed feature groups, verification commands, and metric deltas or current metric snapshot.

- [ ] **Step 3: Final report**

Report touched files, verification evidence, training result paths, and next recommendation.
