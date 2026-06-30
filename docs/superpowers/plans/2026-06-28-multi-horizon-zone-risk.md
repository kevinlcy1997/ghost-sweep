# Multi-Horizon Zone Risk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan includes cost-aware model routing; use the cheapest suggested tier that can satisfy each task's review gate, and escalate only when the listed escalation trigger occurs.

**Goal:** Build separate 30-minute, 1-hour, and 2-hour Hong Kong zone-risk models and expose multi-horizon simulation results for real locations.

**Architecture:** Keep one leakage-safe feature builder, but make horizon targets explicit with `horizon_minutes`. Train and evaluate one selected model per horizon so weak 30-minute performance cannot be hidden by stronger 2-hour performance. Simulation loads the horizon-specific prediction files and reports risk bands side by side.

**Tech Stack:** Python 3.12, pandas, scikit-learn, LightGBM, H3, MLflow-compatible local artifacts, pytest.

---

## File Structure

- `ghost_ranking_features.py`: add `horizon_minutes`, `target_col`, and horizon label generation while preserving default `alert_next_2h`.
- `analysis/run_model_iteration.py`: parameterize model iteration by horizon, write horizon-specific artifacts, and run all horizons.
- `analysis/run_multi_horizon_experiment.py`: small orchestration CLI that calls the horizon-aware model iteration and writes a side-by-side summary.
- `analysis/simulate_real_location_risk.py`: load prediction files for multiple horizons and report a combined location forecast.
- `tests/test_multi_horizon_features.py`: target-building tests for 30m/1h/2h.
- `tests/test_multi_horizon_iteration.py`: pure selection/artifact naming tests.
- `tests/test_multi_horizon_simulation.py`: simulation contract test showing all horizon results.

## Agent and Model Routing

| Workstream | Agent Role | Suggested Model Tier | Cost Control | Review Gate |
| --- | --- | --- | --- | --- |
| Horizon target builder | Builder | mid-tier | Edit `ghost_ranking_features.py` and focused tests only | Tests prove 30m/1h/2h labels differ correctly and 2h stays backward-compatible |
| Horizon model iteration | Builder | mid-tier | Reuse existing candidate/eval code; no new model libraries | Per-horizon artifacts exist and summary includes all three horizons |
| Simulation integration | Builder | mid-tier | Modify only simulation module and tests | Real-location output shows next 30m, 1h, and 2h for each scenario |
| Verification | Reviewer | low-cost | Run listed commands and inspect generated artifacts | Full tests pass and reports are non-empty |

## Task 1: Add Horizon-Aware Targets

**Files:**
- Modify: `ghost_ranking_features.py`
- Create: `tests/test_multi_horizon_features.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** This is core label behavior and must avoid target leakage, but it follows the existing feature-builder pattern.
**Inputs Needed:** This task, `ghost_ranking_features.py`, existing `_event()` helper in `tests/test_ghost_ranking_features.py`.
**Expected Output:** `build_zone_ranking_training_data(..., horizon_minutes=30, target_col="alert_next_30m")` works, and default calls still produce `alert_next_2h`.
**Review Gate:** Tests prove the same `target_time` can be negative for 30m and positive for 2h when a future event is between those horizons.
**Escalation Trigger:** Escalate to high-capability if changing horizons alters past-only feature windows.

- [ ] **Step 1: Write failing tests**

Create `tests/test_multi_horizon_features.py`:

```python
from ghost_ranking_features import build_zone_ranking_training_data
from tests.test_ghost_ranking_features import _event


def test_horizon_minutes_controls_future_target_window():
    events = [
        _event(22.3154, 114.1698, f"2026-06-{day:02d} 10:00:00")
        for day in range(1, 18)
    ]
    events.append(_event(22.3154, 114.1698, "2026-06-17 10:45:00"))

    rows_30m = build_zone_ranking_training_data(
        events,
        horizon_minutes=30,
        target_col="alert_next_30m",
    )
    rows_2h = build_zone_ranking_training_data(
        events,
        horizon_minutes=120,
        target_col="alert_next_2h",
    )

    target_time = "2026-06-17 10:00:00"
    row_30m = rows_30m.loc[rows_30m["target_time"].astype(str) == target_time].iloc[0]
    row_2h = rows_2h.loc[rows_2h["target_time"].astype(str) == target_time].iloc[0]

    assert row_30m["alert_next_30m"] == 0
    assert row_2h["alert_next_2h"] == 1


def test_default_target_remains_alert_next_2h():
    events = [
        _event(22.3154, 114.1698, f"2026-06-{day:02d} 10:00:00")
        for day in range(1, 18)
    ]

    rows = build_zone_ranking_training_data(events)

    assert "alert_next_2h" in rows.columns
```

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_multi_horizon_features.py -q --basetemp C:\tmp\ghost-sweep-pytest
```

Expected before implementation: failure because `horizon_minutes` and `target_col` are not accepted.

- [ ] **Step 2: Implement the minimal label change**

Change `build_zone_ranking_training_data()` signature in `ghost_ranking_features.py`:

```python
def build_zone_ranking_training_data(
    events: Iterable[dict],
    zone_col: str = "h3_zone",
    forecast_hours: int = 2,
    lookback_days: int = 14,
    resolution: int = DEFAULT_H3_RESOLUTION,
    horizon_minutes: int | None = None,
    target_col: str = "alert_next_2h",
) -> pd.DataFrame:
```

Inside the function:

```python
if horizon_minutes is None:
    horizon_minutes = forecast_hours * 60
future_end = target_dt + timedelta(minutes=horizon_minutes)
```

Replace the hard-coded row value:

```python
target_col: int(future_by_zone[zone_id] > 0),
```

- [ ] **Step 3: Verify green**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_multi_horizon_features.py tests/test_ghost_ranking_features.py -q --basetemp C:\tmp\ghost-sweep-pytest
```

Expected: tests pass.

## Task 2: Add Horizon-Aware Model Iteration

**Files:**
- Modify: `analysis/run_model_iteration.py`
- Create: `analysis/run_multi_horizon_experiment.py`
- Create: `tests/test_multi_horizon_iteration.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** The task reuses existing model-selection logic and only parameterizes target/artifact naming.
**Inputs Needed:** Task 1 output and current `analysis/run_model_iteration.py`.
**Expected Output:** Separate per-horizon selected models, predictions, metadata, and a combined summary report.
**Review Gate:** Test verifies artifact names and target selection for `30m`, `1h`, and `2h`.
**Escalation Trigger:** Escalate to high-capability if model metrics become incomparable across horizons due to target column mismatch.

- [ ] **Step 1: Write failing pure tests**

Create `tests/test_multi_horizon_iteration.py`:

```python
from analysis.run_model_iteration import horizon_slug, target_for_horizon


def test_horizon_slug_formats_minutes_and_hours():
    assert horizon_slug(30) == "30m"
    assert horizon_slug(60) == "1h"
    assert horizon_slug(120) == "2h"


def test_target_for_horizon_names_alert_columns():
    assert target_for_horizon(30) == "alert_next_30m"
    assert target_for_horizon(60) == "alert_next_1h"
    assert target_for_horizon(120) == "alert_next_2h"
```

Run:

```powershell
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_multi_horizon_iteration.py -q --basetemp C:\tmp\ghost-sweep-pytest
```

Expected before implementation: import failure.

- [ ] **Step 2: Parameterize iteration**

Add helpers to `analysis/run_model_iteration.py`:

```python
def horizon_slug(horizon_minutes: int) -> str:
    if horizon_minutes % 60 == 0:
        return f"{horizon_minutes // 60}h"
    return f"{horizon_minutes}m"


def target_for_horizon(horizon_minutes: int) -> str:
    return f"alert_next_{horizon_slug(horizon_minutes)}"
```

Change `evaluate_candidates(df, target_col=TARGET)` and every `frame.loc[..., TARGET]` to use `target_col`.

Change `run_model_iteration(resolution=8, horizon_minutes=120)` to:

```python
slug = horizon_slug(horizon_minutes)
target_col = target_for_horizon(horizon_minutes)
df = build_zone_ranking_training_data(
    events.to_dict("records"),
    forecast_hours=max(1, horizon_minutes // 60),
    horizon_minutes=horizon_minutes,
    target_col=target_col,
    lookback_days=7,
    resolution=resolution,
)
```

Write horizon-specific files:

```python
analysis/model_iteration_summary_{slug}_latest.csv
analysis/model_iteration_report_{slug}.html
analysis/best_iterated_zone_model_{slug}.joblib
analysis/best_iterated_model_metadata_{slug}.json
analysis/iterated_zone_predictions_{slug}_latest.csv
```

Keep the legacy non-suffixed aliases when `horizon_minutes == 120`.

- [ ] **Step 3: Add multi-horizon orchestration**

Create `analysis/run_multi_horizon_experiment.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_model_iteration import ROOT, run_model_iteration

HORIZONS = [30, 60, 120]
SUMMARY_PATH = ROOT / "analysis" / "multi_horizon_summary_latest.csv"
REPORT_PATH = ROOT / "analysis" / "multi_horizon_report.html"


def run_multi_horizon_experiment() -> list[dict]:
    rows = []
    for horizon in HORIZONS:
        metadata = run_model_iteration(horizon_minutes=horizon)
        row = {
            "horizon_minutes": horizon,
            "target": metadata["target_col"],
            "chosen_model": metadata["chosen_model"]["model"],
            **{f"holdout_{k}": v for k, v in metadata["holdout_metrics"].items()},
            "metadata_path": metadata["metadata_path"],
            "predictions_path": metadata["predictions_path"],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_PATH, index=False)
    REPORT_PATH.write_text(df.to_html(index=False), encoding="utf-8")
    return rows


if __name__ == "__main__":
    print(json.dumps(run_multi_horizon_experiment(), indent=2))
```

- [ ] **Step 4: Run the multi-horizon experiment**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe analysis/run_multi_horizon_experiment.py
```

Expected: three horizon metadata files and `analysis/multi_horizon_summary_latest.csv`.

## Task 3: Add Multi-Horizon Location Simulation

**Files:**
- Modify: `analysis/simulate_real_location_risk.py`
- Create: `tests/test_multi_horizon_simulation.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** It extends an existing report script with predictable artifact loading and output shape.
**Inputs Needed:** Task 2 output and current simulation module.
**Expected Output:** Real-location report shows next 30m, 1h, and 2h scores per scenario.
**Review Gate:** Test proves one location returns a `horizons` dictionary with `30m`, `1h`, and `2h`.
**Escalation Trigger:** Escalate to high-capability if horizon predictions are temporally misaligned or impossible to compare.

- [ ] **Step 1: Write failing simulation test**

Create `tests/test_multi_horizon_simulation.py`:

```python
import pandas as pd

from analysis.simulate_real_location_risk import simulate_location_multi_horizon
from ghost_zones import compute_h3_zone, h3_zone_centroid


def _predictions(score):
    zone = compute_h3_zone(22.3154, 114.1698, resolution=8)
    lat, lng = h3_zone_centroid(zone)
    return pd.DataFrame(
        [{
            "target_time": "2026-06-25 17:00:00",
            "zone_id": zone,
            "district": "Mong Kok",
            "region": "Kowloon West",
            "zone_lat": lat,
            "zone_lng": lng,
            "score": score,
            "actual": 1,
        }]
    )


def test_simulate_location_multi_horizon_returns_all_horizons():
    result = simulate_location_multi_horizon(
        lat=22.3154,
        lng=114.1698,
        target_time="2026-06-25 17:00:00",
        predictions_by_horizon={
            "30m": _predictions(0.8),
            "1h": _predictions(0.6),
            "2h": _predictions(0.4),
        },
    )

    assert set(result["horizons"]) == {"30m", "1h", "2h"}
    assert result["horizons"]["30m"]["score"] == 0.8
    assert result["horizons"]["2h"]["score"] == 0.4
```

- [ ] **Step 2: Implement simulation helper**

Add to `analysis/simulate_real_location_risk.py`:

```python
def simulate_location_multi_horizon(
    lat: float,
    lng: float,
    target_time: str | pd.Timestamp,
    predictions_by_horizon: dict[str, pd.DataFrame] | None = None,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> dict[str, Any]:
```

When `predictions_by_horizon` is `None`, load:

```python
analysis/iterated_zone_predictions_30m_latest.csv
analysis/iterated_zone_predictions_1h_latest.csv
analysis/iterated_zone_predictions_2h_latest.csv
```

Return:

```python
{
    "input_lat": lat,
    "input_lng": lng,
    "target_time": str(pd.Timestamp(target_time)),
    "horizons": {
        "30m": simulate_location_risk(...),
        "1h": simulate_location_risk(...),
        "2h": simulate_location_risk(...),
    },
}
```

- [ ] **Step 3: Regenerate simulation artifacts**

Run:

```powershell
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe analysis/simulate_real_location_risk.py
```

Expected: `analysis/real_location_simulation_report.html` includes horizon columns.

## Task 4: Verification and Reasonableness Review

**Files:**
- No planned code edits

**Agent Role:** Reviewer
**Suggested Model Tier:** low-cost
**Why This Tier:** This is deterministic command verification and artifact inspection.
**Inputs Needed:** Completed Tasks 1-3.
**Expected Output:** Tests pass, horizon artifacts exist, and location simulation results are plausible enough to inspect.
**Review Gate:** Main agent records metrics and any caveats before marking complete.
**Escalation Trigger:** Escalate to high-capability if 30m labels are too sparse to train or if simulation contradicts historical active areas.

- [ ] **Step 1: Run focused tests**

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_multi_horizon_features.py tests/test_multi_horizon_iteration.py tests/test_multi_horizon_simulation.py -q --basetemp C:\tmp\ghost-sweep-pytest
```

Expected: all pass.

- [ ] **Step 2: Run full tests**

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest -q --basetemp C:\tmp\ghost-sweep-pytest
```

Expected: all pass.

- [ ] **Step 3: Confirm artifacts**

```powershell
Get-Item analysis/multi_horizon_summary_latest.csv
Get-Item analysis/multi_horizon_report.html
Get-Item analysis/iterated_zone_predictions_30m_latest.csv
Get-Item analysis/iterated_zone_predictions_1h_latest.csv
Get-Item analysis/iterated_zone_predictions_2h_latest.csv
Get-Item analysis/real_location_simulation_report.html
```

Expected: all files exist and have non-zero size.

## Self-Review

Spec coverage: The plan implements separate models for next 30 minutes, 1 hour, and 2 hours, then updates real-location simulation to show horizon-specific risk.

Placeholder scan: No placeholders remain; every task names files, code shape, commands, and expected outputs.

Type consistency: Horizon values are `int` minutes, slugs are strings like `30m`, `1h`, `2h`, targets are `alert_next_<slug>`, and predictions use matching file suffixes.

Dependency ordering: Task 1 creates targets, Task 2 trains/evaluates per horizon, Task 3 simulates locations across horizons, and Task 4 verifies.

Agent independence: Each task has enough file and API context to execute independently after the prior task output.

Cost fit: Builder tasks are mid-tier because they touch modeling behavior. Verification is low-cost. High-capability is reserved for leakage or sparsity surprises.

Parallel safety: Tasks are sequential because each depends on the prior task's artifacts.

Review gates: Each task has focused tests or artifact checks before the next task starts.

## Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-multi-horizon-zone-risk.md`.

Recommended execution: Subagent-Driven. I dispatch focused agents by task, use cheaper tiers for bounded read/review/mechanical work, reserve high-capability reasoning for architecture and integration review, and checkpoint after each diff.

Alternative execution: Inline Execution. I execute the plan in this session with the same checkpoints, but with less parallelism.
