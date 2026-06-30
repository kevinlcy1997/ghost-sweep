# Two-Stage Zone Risk Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan includes cost-aware model routing; use the cheapest suggested tier that can satisfy each task's review gate, and escalate only when the listed escalation trigger occurs.

**Goal:** Build a two-stage, multi-horizon Hong Kong zone-risk pipeline that separately predicts city-wide activity timing and road-zone spatial risk, then exposes both in the service dashboard.

**Architecture:** Keep the existing single-stage model artifacts as fallback. Add a split utility, a city activity feature builder, spatial active-window sampling, a new two-stage experiment runner, and dashboard/manifest preference for two-stage artifacts. Use chronological purged validation and latest contiguous positive-count holdouts for each horizon.

**Tech Stack:** Python 3.12, pandas, scikit-learn, LightGBM, joblib, MLflow-compatible artifacts, H3 road-access grid, local `http.server` dashboard service, pytest.

---

## File Structure

Create `analysis/two_stage_splits.py` for reusable chronological split utilities. This module owns purged rolling-origin folds and positive-count holdout masks so both model stages use the same leakage rules.

Create `ghost_activity_features.py` for city-level hourly activity rows. This module owns Stage 1 features and labels and should not depend on model training code.

Modify `ghost_ranking_features.py` to add active-window spatial sampling while preserving `build_zone_ranking_training_data` behavior for legacy scripts.

Create `analysis/run_two_stage_experiment.py` for model training, evaluation, artifact writing, and multi-horizon orchestration. It should import the existing preprocessing constants and scoring helpers where practical, but keep Stage 1 and Stage 2 metrics explicit.

Modify `analysis/dashboard_service.py` so `/api/model-metrics`, `/api/predictions`, and `/api/grid.geojson` prefer two-stage artifacts when present and fall back to existing single-stage files otherwise.

Modify `analysis/build_dashboard_manifest.py` and `start-dev.ps1` so traceability and dev startup include the new artifacts.

Add tests in `tests/test_two_stage_splits.py`, `tests/test_activity_features.py`, `tests/test_spatial_sampling.py`, `tests/test_two_stage_experiment.py`, and update existing dashboard/manifest tests.

## Agent and Model Routing

| Workstream | Agent Role | Suggested Model Tier | Cost Control | Review Gate |
| --- | --- | --- | --- | --- |
| Split utilities | Builder | mid-tier | Edits one new module and one focused test file | Purge gap and positive-count holdout tests pass |
| Activity feature builder | Builder | mid-tier | New feature module with synthetic data tests only | No-event hours included and future labels are correct |
| Spatial sampling | Builder | mid-tier | Small extension to existing feature module | Positives retained, active-window hard negatives sampled, inactive sample bounded |
| Two-stage experiment runner | Architect/Builder | high-capability | Cross-module ML contract and artifact design | Synthetic runner test passes and artifact schema matches spec |
| Dashboard and manifest integration | Builder | mid-tier | Bounded service/manifest edits with fallback tests | API tests prove two-stage preference and legacy fallback |
| Startup and verification | Reviewer | low-cost | Run listed commands and inspect outputs only | Commands pass or failures are summarized exactly |

### Task 1: Chronological Split Utilities

**Files:**
- Create: `analysis/two_stage_splits.py`
- Test: `tests/test_two_stage_splits.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** The task is a focused utility module with deterministic pandas behavior and clear tests.
**Inputs Needed:** This task section and the approved design spec at `docs/superpowers/specs/2026-06-30-two-stage-zone-risk-model-design.md`.
**Expected Output:** Passing tests for purged rolling splits and positive-count holdouts.
**Review Gate:** Main agent confirms masks are chronological, purge gap is at least horizon length, and metadata reports fallback clearly.

- [ ] **Step 1: Write the failing split tests**

Create `tests/test_two_stage_splits.py`:

```python
import pandas as pd

from analysis.two_stage_splits import make_positive_count_holdout, make_purged_rolling_splits


def test_make_purged_rolling_splits_keeps_horizon_gap():
    df = pd.DataFrame({"target_time": pd.date_range("2026-06-01", periods=16, freq="h")})

    splits = make_purged_rolling_splits(df, horizon_minutes=120, n_splits=3)

    assert len(splits) == 3
    for split in splits:
        train_end = df.loc[split.train_mask, "target_time"].max()
        validation_start = df.loc[split.validation_mask, "target_time"].min()
        assert validation_start - train_end >= pd.Timedelta(minutes=120)
        assert split.metadata["purge_minutes"] == 120
        assert split.metadata["train_rows"] > 0
        assert split.metadata["validation_rows"] > 0


def test_make_positive_count_holdout_expands_back_to_minimum_positives():
    df = pd.DataFrame(
        {
            "target_time": pd.date_range("2026-06-01", periods=8, freq="h"),
            "target": [0, 1, 0, 1, 0, 0, 0, 1],
        }
    )

    split = make_positive_count_holdout(df, "target", min_positives=2)

    holdout = df.loc[split.holdout_mask]
    assert holdout["target"].sum() == 2
    assert holdout["target_time"].min() == pd.Timestamp("2026-06-01 03:00:00")
    assert split.metadata["met_min_positives"] is True
    assert split.metadata["holdout_positives"] == 2
    assert split.metadata["train_rows"] == 3


def test_make_positive_count_holdout_reports_fallback_when_threshold_not_met():
    df = pd.DataFrame(
        {
            "target_time": pd.date_range("2026-06-01", periods=4, freq="h"),
            "target": [0, 0, 1, 0],
        }
    )

    split = make_positive_count_holdout(df, "target", min_positives=5)

    assert split.metadata["met_min_positives"] is False
    assert split.metadata["holdout_positives"] == 1
    assert split.holdout_mask.all()
    assert split.train_mask.sum() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_two_stage_splits.py -v`

Expected: FAIL with `ModuleNotFoundError` for `analysis.two_stage_splits`.

- [ ] **Step 3: Implement split utilities**

Create `analysis/two_stage_splits.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PurgedSplit:
    train_mask: pd.Series
    validation_mask: pd.Series
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HoldoutSplit:
    train_mask: pd.Series
    holdout_mask: pd.Series
    metadata: dict[str, Any]


def _time_series(df: pd.DataFrame, time_col: str) -> pd.Series:
    if time_col not in df:
        raise ValueError(f"Missing required time column: {time_col}")
    return pd.to_datetime(df[time_col])


def make_purged_rolling_splits(
    df: pd.DataFrame,
    horizon_minutes: int,
    n_splits: int = 4,
    time_col: str = "target_time",
) -> list[PurgedSplit]:
    frame = df.copy()
    frame[time_col] = _time_series(frame, time_col)
    times = pd.Series(sorted(frame[time_col].dropna().unique()))
    if len(times) < n_splits + 3:
        raise ValueError("Need more target_time values for purged rolling splits.")

    validation_size = max(1, len(times) // (n_splits + 2))
    purge = pd.Timedelta(minutes=horizon_minutes)
    splits: list[PurgedSplit] = []
    for split_index in range(n_splits):
        validation_start_index = len(times) - validation_size * (n_splits - split_index)
        validation_end_index = min(len(times), validation_start_index + validation_size)
        if validation_start_index <= 0 or validation_end_index <= validation_start_index:
            continue
        validation_times = times.iloc[validation_start_index:validation_end_index]
        validation_start = pd.Timestamp(validation_times.min())
        train_cutoff = validation_start - purge
        train_mask = frame[time_col] <= train_cutoff
        validation_mask = frame[time_col].isin(set(validation_times))
        if not train_mask.any() or not validation_mask.any():
            continue
        metadata = {
            "fold": len(splits) + 1,
            "purge_minutes": int(horizon_minutes),
            "train_start": frame.loc[train_mask, time_col].min().isoformat(),
            "train_end": frame.loc[train_mask, time_col].max().isoformat(),
            "validation_start": frame.loc[validation_mask, time_col].min().isoformat(),
            "validation_end": frame.loc[validation_mask, time_col].max().isoformat(),
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
        }
        splits.append(PurgedSplit(train_mask=train_mask, validation_mask=validation_mask, metadata=metadata))
    if not splits:
        raise ValueError("Could not construct non-empty purged rolling splits.")
    return splits


def make_positive_count_holdout(
    df: pd.DataFrame,
    target_col: str,
    min_positives: int = 50,
    time_col: str = "target_time",
) -> HoldoutSplit:
    if target_col not in df:
        raise ValueError(f"Missing required target column: {target_col}")
    frame = df.copy()
    frame[time_col] = _time_series(frame, time_col)
    by_time = (
        frame.groupby(time_col, sort=True)[target_col]
        .sum()
        .reset_index()
        .sort_values(time_col, ascending=False)
    )
    cumulative = 0
    selected_times: list[pd.Timestamp] = []
    for row in by_time.itertuples(index=False):
        selected_times.append(pd.Timestamp(getattr(row, time_col)))
        cumulative += int(getattr(row, target_col))
        if cumulative >= min_positives:
            break

    selected_set = set(selected_times)
    holdout_mask = frame[time_col].isin(selected_set)
    train_mask = ~holdout_mask
    positives = int(frame.loc[holdout_mask, target_col].sum())
    rows = int(holdout_mask.sum())
    metadata = {
        "min_positives": int(min_positives),
        "met_min_positives": bool(positives >= min_positives),
        "holdout_start": frame.loc[holdout_mask, time_col].min().isoformat() if rows else None,
        "holdout_end": frame.loc[holdout_mask, time_col].max().isoformat() if rows else None,
        "holdout_rows": rows,
        "holdout_positives": positives,
        "holdout_base_rate": float(positives / rows) if rows else 0.0,
        "train_rows": int(train_mask.sum()),
    }
    return HoldoutSplit(train_mask=train_mask, holdout_mask=holdout_mask, metadata=metadata)
```

- [ ] **Step 4: Run split tests**

Run: `C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_two_stage_splits.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
C:\Users\Kevin\.local\bin\rtk.exe git add analysis/two_stage_splits.py tests/test_two_stage_splits.py
C:\Users\Kevin\.local\bin\rtk.exe git commit -m "feat: add leakage-safe two-stage splits"
```

### Task 2: City Activity Feature Builder

**Files:**
- Create: `ghost_activity_features.py`
- Test: `tests/test_activity_features.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** The module is new and deterministic but requires careful future-window labeling.
**Inputs Needed:** `ghost_ranking_features.py` for timestamp parsing and event enrichment behavior.
**Expected Output:** Stage 1 hourly rows with no-event hours retained and horizon-specific labels.
**Review Gate:** Main agent verifies all features are past-only and no-event rows remain in the data.

- [ ] **Step 1: Write failing activity feature tests**

Create `tests/test_activity_features.py`:

```python
import pandas as pd

from ghost_activity_features import build_activity_training_data


def _event(create_dt: str, lat: float = 22.302, lng: float = 114.172) -> dict:
    return {"create_dt": create_dt, "lat": lat, "lng": lng, "name": "ticket"}


def test_build_activity_training_data_keeps_no_event_hours_and_future_labels():
    rows = build_activity_training_data(
        [
            _event("2026-06-01 00:10:00"),
            _event("2026-06-01 03:30:00"),
        ],
        horizon_minutes=60,
        lookback_hours=1,
    )

    assert list(rows["target_time"]) == [
        pd.Timestamp("2026-06-01 01:00:00"),
        pd.Timestamp("2026-06-01 02:00:00"),
        pd.Timestamp("2026-06-01 03:00:00"),
    ]
    assert rows.loc[rows["target_time"] == pd.Timestamp("2026-06-01 02:00:00"), "activity_next_1h"].item() == 0
    assert rows.loc[rows["target_time"] == pd.Timestamp("2026-06-01 03:00:00"), "activity_next_1h"].item() == 1
    assert rows.loc[rows["target_time"] == pd.Timestamp("2026-06-01 01:00:00"), "city_event_count_1h"].item() == 1


def test_build_activity_training_data_uses_requested_horizon_target_name():
    rows = build_activity_training_data(
        [_event("2026-06-01 00:10:00"), _event("2026-06-01 01:20:00")],
        horizon_minutes=30,
        lookback_hours=1,
    )

    assert "activity_next_30m" in rows.columns
    assert rows["activity_next_30m"].isin([0, 1]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_activity_features.py -v`

Expected: FAIL with `ModuleNotFoundError` for `ghost_activity_features`.

- [ ] **Step 3: Implement activity features**

Create `ghost_activity_features.py` with a `build_activity_training_data(events, horizon_minutes, lookback_hours=168, resolution=DEFAULT_H3_RESOLUTION)` function that enriches events via `enrich_events_with_zones`, creates hourly target times from first event plus `lookback_hours` to last event floored to hour, includes all hours, and emits these columns: `target_time`, `hour`, `day_of_week`, `is_weekend`, `hour_bucket`, `city_event_count_1h`, `city_event_count_3h`, `city_event_count_24h`, `city_event_count_7d`, `active_districts_24h`, `active_regions_24h`, `city_3h_to_24h_ratio`, `city_24h_to_7d_ratio`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, and `activity_next_<slug>`.

- [ ] **Step 4: Run activity feature tests**

Run: `C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_activity_features.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
C:\Users\Kevin\.local\bin\rtk.exe git add ghost_activity_features.py tests/test_activity_features.py
C:\Users\Kevin\.local\bin\rtk.exe git commit -m "feat: add city activity features"
```

### Task 3: Active-Window Spatial Sampling

**Files:**
- Modify: `ghost_ranking_features.py`
- Test: `tests/test_spatial_sampling.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** This is a focused extension of existing feature data without changing legacy builder output.
**Inputs Needed:** Existing `ghost_ranking_features.py` and Task 1 split contracts.
**Expected Output:** Sampling function that keeps all positives, adds hard negatives from active windows, and bounds inactive negatives.
**Review Gate:** Tests prove multi-label positives can coexist at one timestamp and inactive-window negatives cannot dominate.

- [ ] **Step 1: Write failing sampling tests**

Create `tests/test_spatial_sampling.py`:

```python
import pandas as pd

from ghost_ranking_features import sample_spatial_training_rows


def test_sample_spatial_training_rows_keeps_all_positive_zones():
    df = pd.DataFrame(
        {
            "target_time": ["2026-06-01 10:00"] * 5,
            "zone_id": ["a", "b", "c", "d", "e"],
            "zone_event_count_24h": [2, 1, 9, 0, 5],
            "district_event_count_24h": [3, 3, 7, 0, 7],
            "alert_next_1h": [1, 1, 0, 0, 0],
        }
    )

    sampled = sample_spatial_training_rows(df, "alert_next_1h", negative_ratio=1, random_state=7)

    assert set(sampled.loc[sampled["alert_next_1h"] == 1, "zone_id"]) == {"a", "b"}
    assert len(sampled) == 4
    assert sampled["target_time"].nunique() == 1


def test_sample_spatial_training_rows_limits_inactive_negatives():
    df = pd.DataFrame(
        {
            "target_time": ["2026-06-01 10:00"] * 3 + ["2026-06-01 11:00"] * 4,
            "zone_id": list("abcdefg"),
            "zone_event_count_24h": [0, 0, 0, 3, 2, 1, 0],
            "district_event_count_24h": [0, 0, 0, 4, 4, 1, 0],
            "alert_next_1h": [0, 0, 0, 1, 0, 0, 0],
        }
    )

    sampled = sample_spatial_training_rows(
        df,
        "alert_next_1h",
        negative_ratio=2,
        inactive_negative_fraction=0.5,
        random_state=11,
    )

    inactive = sampled[sampled["target_time"] == "2026-06-01 10:00"]
    assert len(inactive) <= 1
    assert sampled["alert_next_1h"].sum() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_spatial_sampling.py -v`

Expected: FAIL because `sample_spatial_training_rows` is missing.

- [ ] **Step 3: Implement sampling**

Add `sample_spatial_training_rows(df, target_col, negative_ratio=5, inactive_negative_fraction=0.02, random_state=42)` to `ghost_ranking_features.py`. Implementation should copy the frame, mark active timestamps where grouped target sum is positive, keep all positives, sort active negatives by `zone_event_count_24h`, `district_event_count_24h`, and `zone_event_count_7d` if present, sample up to `positive_count * negative_ratio` active negatives, sample at most `ceil(active_negative_limit * inactive_negative_fraction)` inactive negatives, concatenate, drop duplicate original indexes, and return rows sorted by `target_time`, target descending, and `zone_id`.

- [ ] **Step 4: Run sampling tests**

Run: `C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_spatial_sampling.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
C:\Users\Kevin\.local\bin\rtk.exe git add ghost_ranking_features.py tests/test_spatial_sampling.py
C:\Users\Kevin\.local\bin\rtk.exe git commit -m "feat: sample active windows for spatial risk"
```

### Task 4: Two-Stage Experiment Runner

**Files:**
- Create: `analysis/run_two_stage_experiment.py`
- Test: `tests/test_two_stage_experiment.py`

**Agent Role:** Architect/Builder
**Suggested Model Tier:** high-capability
**Why This Tier:** This task integrates data contracts, model selection, split metadata, probabilities, ranking metrics, and artifact naming across both stages.
**Inputs Needed:** Tasks 1-3, `analysis/run_model_iteration.py`, `analysis/run_zone_ranking_experiment.py`, and the approved design spec.
**Expected Output:** A runner that trains 30m, 1h, and 2h two-stage models and writes the approved artifact contract.
**Review Gate:** Main agent verifies output schemas, split metadata, and fallback behavior before dashboard integration.

- [ ] **Step 1: Write failing runner tests**

Create `tests/test_two_stage_experiment.py`:

```python
import json

import pandas as pd

from analysis.run_two_stage_experiment import (
    activity_target_for_horizon,
    combine_activity_and_spatial_scores,
    horizon_slug,
    write_two_stage_summary,
)


def test_activity_target_for_horizon_names_columns():
    assert activity_target_for_horizon(30) == "activity_next_30m"
    assert activity_target_for_horizon(60) == "activity_next_1h"
    assert activity_target_for_horizon(120) == "activity_next_2h"


def test_combine_activity_and_spatial_scores_adds_final_probability_and_rank():
    spatial = pd.DataFrame(
        {
            "target_time": ["2026-06-01 10:00"] * 2,
            "zone_id": ["b", "a"],
            "spatial_probability": [0.2, 0.8],
            "actual": [0, 1],
        }
    )

    combined = combine_activity_and_spatial_scores(spatial, activity_probability=0.5)

    assert list(combined["zone_id"]) == ["a", "b"]
    assert list(combined["probability"]) == [0.4, 0.1]
    assert list(combined["rank"]) == [1, 2]
    assert combined["activity_probability"].unique().tolist() == [0.5]


def test_write_two_stage_summary_writes_stage_paths(tmp_path):
    metadata = [
        {
            "horizon_minutes": 30,
            "activity_model": {"chosen_model": {"model": "logistic"}, "holdout_metrics": {"average_precision": 0.5}},
            "spatial_model": {"chosen_model": {"model": "lightgbm"}, "holdout_metrics": {"precision_at_20": 0.1}},
            "activity_metadata_path": "analysis/activity_model_metadata_30m.json",
            "spatial_metadata_path": "analysis/spatial_model_metadata_30m.json",
            "predictions_path": "analysis/spatial_zone_predictions_30m_latest.csv",
        }
    ]

    path = tmp_path / "summary.csv"
    rows = write_two_stage_summary(metadata, path)

    assert rows[0]["horizon"] == "30m"
    assert rows[0]["activity_model"] == "logistic"
    assert rows[0]["spatial_model"] == "lightgbm"
    assert path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_two_stage_experiment.py -v`

Expected: FAIL because `analysis.run_two_stage_experiment` is missing.

- [ ] **Step 3: Implement runner**

Create `analysis/run_two_stage_experiment.py` with these public functions: `horizon_slug`, `activity_target_for_horizon`, `combine_activity_and_spatial_scores`, `write_two_stage_summary`, `run_two_stage_horizon`, and `run_two_stage_experiment`. Reuse existing candidate pipelines and `_score_predictions` for spatial scoring where possible. Stage 1 should use activity features, binary metrics, purged folds, and positive-count holdout. Stage 2 should use ranking features, `sample_spatial_training_rows` for training folds/final train, ranking metrics on unsampled validation/holdout rows, and final probabilities multiplied by the latest Stage 1 holdout activity probability for the same `target_time` when available, otherwise by the global Stage 1 holdout mean. Write the artifact names listed in the spec.

- [ ] **Step 4: Run runner tests**

Run: `C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_two_stage_experiment.py -v`

Expected: PASS.

- [ ] **Step 5: Run a small integrated smoke**

Run: `C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_two_stage_splits.py tests/test_activity_features.py tests/test_spatial_sampling.py tests/test_two_stage_experiment.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
C:\Users\Kevin\.local\bin\rtk.exe git add analysis/run_two_stage_experiment.py tests/test_two_stage_experiment.py
C:\Users\Kevin\.local\bin\rtk.exe git commit -m "feat: add two-stage experiment runner"
```

### Task 5: Dashboard and Manifest Integration

**Files:**
- Modify: `analysis/dashboard_service.py`
- Modify: `analysis/build_dashboard_manifest.py`
- Modify: `tests/test_dashboard_service.py`
- Modify: `tests/test_dashboard_manifest.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** This is bounded API/file integration with explicit fallback tests.
**Inputs Needed:** Task 4 artifact schema and existing dashboard tests.
**Expected Output:** Dashboard APIs prefer two-stage artifacts and still work with old single-stage files.
**Review Gate:** Existing dashboard tests plus new preference tests pass.

- [ ] **Step 1: Add failing dashboard tests**

Extend `tests/test_dashboard_service.py` with tests that temporarily point `PATHS["two_stage_summary"]`, `PATHS["spatial_predictions_30m"]`, and `PATHS["activity_predictions_30m"]` at temp files. Assert `/api/model-metrics` returns `model_family == "two_stage"`, activity and spatial model names, and holdout split metadata. Assert `/api/predictions?horizon=30m` returns `activity_probability`, `spatial_probability`, `probability`, and `risk_band`.

- [ ] **Step 2: Add failing manifest test**

Extend `tests/test_dashboard_manifest.py` to assert the manifest has groups named `two_stage_models`, `two_stage_metadata`, and `two_stage_predictions`.

- [ ] **Step 3: Run dashboard/manifest tests to verify failure**

Run: `C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_dashboard_service.py tests/test_dashboard_manifest.py -v`

Expected: FAIL until service and manifest paths are updated.

- [ ] **Step 4: Implement service preference and fallback**

In `analysis/dashboard_service.py`, add paths for `two_stage_summary_latest.csv`, `activity_predictions_<horizon>_latest.csv`, and `spatial_zone_predictions_<horizon>_latest.csv`. Update `read_prediction_rows` to prefer spatial predictions for a horizon when the file exists and has rows. Update `api_model_metrics` to read `two_stage_summary_latest.csv` when present, populate explicit Stage 1 and Stage 2 fields, include `model_family: "two_stage"`, and fall back to current `multi_horizon_summary_latest.csv` otherwise. Keep `/api/grid.geojson` unchanged except that merged prediction properties now include the new probability fields.

- [ ] **Step 5: Implement manifest groups**

In `analysis/build_dashboard_manifest.py`, add artifact groups for `two_stage_models`, `two_stage_metadata`, and `two_stage_predictions` with all 30m/1h/2h activity/spatial artifacts plus `two_stage_summary_latest.csv`.

- [ ] **Step 6: Run dashboard/manifest tests**

Run: `C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_dashboard_service.py tests/test_dashboard_manifest.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
C:\Users\Kevin\.local\bin\rtk.exe git add analysis/dashboard_service.py analysis/build_dashboard_manifest.py tests/test_dashboard_service.py tests/test_dashboard_manifest.py
C:\Users\Kevin\.local\bin\rtk.exe git commit -m "feat: surface two-stage model artifacts in dashboard"
```

### Task 6: Dev Startup and Verification

**Files:**
- Modify: `start-dev.ps1`
- Test: existing pytest suite slice

**Agent Role:** Reviewer/Builder
**Suggested Model Tier:** low-cost for command verification, mid-tier if startup changes fail.
**Why This Tier:** The code change is mechanical and verification commands are explicit.
**Inputs Needed:** Task 4 runner path and current `start-dev.ps1`.
**Expected Output:** `-RetrainModels` runs the two-stage experiment and rebuilds the manifest.
**Review Gate:** Focused tests pass, startup command reaches dashboard readiness, and generated artifacts exist.

- [ ] **Step 1: Add startup behavior**

Modify `start-dev.ps1` so the `if ($RetrainModels)` block runs `analysis/run_two_stage_experiment.py` instead of `analysis/run_multi_horizon_experiment.py`, or add a `-LegacyRetrainModels` switch if keeping the old retrain path is useful. Keep MLflow startup unchanged.

- [ ] **Step 2: Run focused tests**

Run:

```bash
C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_two_stage_splits.py tests/test_activity_features.py tests/test_spatial_sampling.py tests/test_two_stage_experiment.py tests/test_dashboard_service.py tests/test_dashboard_manifest.py tests/test_multi_horizon_iteration.py tests/test_model_iteration.py -v
```

Expected: PASS.

- [ ] **Step 3: Run two-stage training**

Run:

```bash
C:\Users\Kevin\.local\bin\rtk.exe python analysis/run_two_stage_experiment.py
```

Expected: writes `analysis/two_stage_summary_latest.csv`, per-horizon activity metadata, per-horizon spatial metadata, and per-horizon spatial predictions.

- [ ] **Step 4: Rebuild manifest**

Run:

```bash
C:\Users\Kevin\.local\bin\rtk.exe python analysis/build_dashboard_manifest.py
```

Expected: manifest includes two-stage artifact groups with `exists: true` for generated outputs.

- [ ] **Step 5: Smoke dashboard dispatch**

Run:

```bash
C:\Users\Kevin\.local\bin\rtk.exe pytest tests/test_dashboard_service.py::test_model_metrics_endpoint_returns_horizon_rows_and_metadata tests/test_dashboard_service.py::test_grid_geojson_merges_horizon_probability_properties -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
C:\Users\Kevin\.local\bin\rtk.exe git add start-dev.ps1 analysis/two_stage_summary_latest.csv analysis/dashboard_manifest_latest.json
C:\Users\Kevin\.local\bin\rtk.exe git commit -m "chore: wire dev startup to two-stage training"
```

## Self-Review

Spec coverage: Stage 1 activity timing is implemented in Task 2 and Task 4. Stage 2 spatial ranking and active-window sampling are implemented in Task 3 and Task 4. Chronological purged model selection and positive-count holdout are implemented in Task 1 and consumed in Task 4. Dashboard activity probability, top-zone risk, split windows, and separate metrics are implemented in Task 5. Multi-horizon 30m/1h/2h support and artifact contract are implemented in Task 4.

Placeholder scan: No task contains `TBD`, `TODO`, `implement later`, or "similar to Task N". The only implementation descriptions without full module bodies are bounded to named functions and exact schemas where writing the full final module in the plan would duplicate low-level code better verified by tests.

Type consistency: Horizon slugs use `30m`, `1h`, and `2h` across activity targets, spatial targets, artifacts, dashboard paths, and summary rows. Split classes use `train_mask`, `validation_mask`, and `holdout_mask` consistently.

Dependency ordering: Splits are first, features second, sampling third, runner fourth, dashboard fifth, startup last. No task edits a file before the functions it depends on are introduced.

Agent independence: Each task lists exact files, tests, commands, expected outputs, and review gates. Tasks 1-3 can run independently after reading existing modules. Task 4 depends on Tasks 1-3. Task 5 depends on Task 4's schema. Task 6 depends on all implementation tasks.

Cost fit: Utility, feature, sampling, dashboard, and verification tasks are mid-tier or low-cost. The two-stage runner is high-capability because it carries the modeling and artifact contract risk.

Parallel safety: Tasks 1 and 2 can run in parallel because they edit different files. Task 3 should wait for Task 2 only for review context, but does not share files. Task 4 must wait for Tasks 1-3. Task 5 must wait for Task 4. Task 6 must run last.

Review gates: Every task has a concrete test command or artifact check before the next task starts.

Plan complete and saved to `docs/superpowers/plans/2026-06-30-two-stage-zone-risk-model.md`.

Recommended execution: Subagent-Driven. I dispatch focused agents by task, use cheaper tiers for bounded read/review/mechanical work, reserve high-capability reasoning for architecture and integration review, and checkpoint after each diff.

Alternative execution: Inline Execution. I execute the plan in this session with the same checkpoints, but with less parallelism.
