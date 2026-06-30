# Fixed HK Grid Analytics Dashboard Plan

> **For agentic workers:** Execute task-by-task. Keep edits scoped to the listed files. This plan includes cost-aware model routing; use the cheapest suggested tier that can satisfy each task's review gate, and escalate only when the listed escalation trigger occurs.

**Goal:** Create a fixed Hong Kong H3 coverage grid, engineer sparse/zero-history features, retrain/evaluate multi-horizon models on that fixed coverage, and build a Spotfire-like interactive dashboard for data discovery, feature analysis, experiment comparison, and model evaluation traceability.

**Assumptions:** The dashboard audience is the project owner/data scientist. The first implementation should be a static local HTML dashboard backed by generated CSV/JSON/GeoJSON artifacts, not a hosted server. Current H3 resolution remains `8` unless comparison evidence says otherwise. HK coverage can start from a buffered bounding box constrained by known HK event extent and district heuristics, then later improve with land/road polygons.

**Evidence From Parallel Exploration:**
- Data-analysis pass found a long-tailed res8 distribution: `5,126` events, `299` active event zones, median `4` events/zone, Gini `0.714`, top 20 zones `46.6%` of events, peak hour `20:00`, Wednesday strongest.
- Modeling pass found the current builder only uses observed zones: `ghost_ranking_features.py` builds `zones = sorted({event[zone_col] for event in enriched})`, so zero-history HK cells are excluded.
- MLOps/dashboard pass recommends a static file-backed dashboard with a manifest, global filters, cross-filtering, and artifact links to generated reports/MLflow outputs.

## Agent and Model Routing

| Workstream | Agent Role | Suggested Model Tier | Cost Control | Review Gate |
| --- | --- | --- | --- | --- |
| Fixed coverage grid | Builder | mid-tier | Own `ghost_zones.py`, `analysis/build_hk_coverage_grid.py`, grid tests | Grid emits stable HK H3 inventory with zero-history cells |
| Data discovery and feature mart | Builder | mid-tier | Own discovery scripts/artifacts and feature tables only | Derived tables match schemas and no event rows are lost |
| Fixed-grid modeling | Builder | high-capability | Own ranking feature builder/model scripts/tests | Zero-history cells included without target leakage; metrics reported per horizon |
| Dashboard generator | Builder | mid-tier | Own dashboard generator/static assets only | Static dashboard loads manifest and supports filters/cross-filtering |
| Verification | Reviewer | low-cost | Run listed tests/artifact checks only | Full tests pass and dashboard artifacts are non-empty |

## Task 1: Create Canonical Fixed HK H3 Coverage Grid

**Files:**
- Modify: `ghost_zones.py`
- Create: `analysis/build_hk_coverage_grid.py`
- Create: `tests/test_hk_coverage_grid.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** Spatial grid generation is bounded, but the output becomes a core data contract.
**Inputs Needed:** `ghost_zones.py`, `ghost_alerts.db`, existing H3 res8 outputs.
**Expected Output:** `analysis/geo/hk_h3_coverage_res8.csv` and `analysis/geo/hk_h3_coverage_res8.geojson` include observed and zero-history candidate cells.
**Review Gate:** Tests verify observed event zones are a subset of the fixed coverage grid and every grid row has centroid/district/region/coverage fields.
**Escalation Trigger:** Escalate if bounding-box coverage includes excessive non-HK water/outside-HK cells that break model sparsity.

- [ ] **Step 1: Write failing coverage tests**

Create `tests/test_hk_coverage_grid.py`:

```python
import pandas as pd

from analysis.build_hk_coverage_grid import build_hk_coverage_grid
from ghost_zones import assign_zone


def test_fixed_grid_contains_observed_event_zone():
    grid = build_hk_coverage_grid(resolution=8)
    observed = assign_zone(22.3154, 114.1698, resolution=8)["h3_zone"]

    assert observed in set(grid["h3_zone"])
    assert {"h3_zone", "h3_resolution", "zone_lat", "zone_lng", "district", "region", "coverage_source"}.issubset(grid.columns)


def test_fixed_grid_contains_zero_history_cells():
    grid = build_hk_coverage_grid(resolution=8)

    assert "has_observed_history" in grid.columns
    assert grid["has_observed_history"].isin([True, False]).all()
    assert (~grid["has_observed_history"]).any()
```

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_hk_coverage_grid.py -q --basetemp C:\tmp\ghost-sweep-pytest
```

Expected before implementation: import failure.

- [ ] **Step 2: Implement grid builder**

Create `analysis/build_hk_coverage_grid.py`:

```python
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import h3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghost_zones import DEFAULT_H3_RESOLUTION, assign_zone, h3_zone_polygon

DB_PATH = ROOT / "ghost_alerts.db"
OUT_DIR = ROOT / "analysis" / "geo"
CSV_PATH = OUT_DIR / "hk_h3_coverage_res8.csv"
GEOJSON_PATH = OUT_DIR / "hk_h3_coverage_res8.geojson"

HK_BOUNDS = {
    "lat_min": 22.15,
    "lat_max": 22.58,
    "lng_min": 113.82,
    "lng_max": 114.45,
}


def _observed_zones(resolution: int) -> set[str]:
    with sqlite3.connect(DB_PATH) as conn:
        events = pd.read_sql_query("select lat, lng from events", conn)
    return {
        assign_zone(row.lat, row.lng, resolution=resolution)["h3_zone"]
        for row in events.itertuples(index=False)
    }


def build_hk_coverage_grid(resolution: int = DEFAULT_H3_RESOLUTION) -> pd.DataFrame:
    observed = _observed_zones(resolution)
    step = 0.004 if resolution <= 8 else 0.002
    candidate_zones: set[str] = set(observed)
    lat = HK_BOUNDS["lat_min"]
    while lat <= HK_BOUNDS["lat_max"]:
        lng = HK_BOUNDS["lng_min"]
        while lng <= HK_BOUNDS["lng_max"]:
            candidate_zones.add(assign_zone(lat, lng, resolution=resolution)["h3_zone"])
            lng += step
        lat += step

    rows = []
    for zone_id in sorted(candidate_zones):
        centroid = h3.cell_to_latlng(zone_id)
        context = assign_zone(centroid[0], centroid[1], resolution=resolution)
        rows.append(
            {
                **context,
                "coverage_source": "observed" if zone_id in observed else "fixed_grid",
                "has_observed_history": zone_id in observed,
            }
        )
    return pd.DataFrame(rows).sort_values("h3_zone").reset_index(drop=True)


def write_hk_coverage_grid(resolution: int = DEFAULT_H3_RESOLUTION) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = build_hk_coverage_grid(resolution)
    csv_path = OUT_DIR / f"hk_h3_coverage_res{resolution}.csv"
    geojson_path = OUT_DIR / f"hk_h3_coverage_res{resolution}.geojson"
    grid.to_csv(csv_path, index=False)
    features = [
        {
            "type": "Feature",
            "properties": row.drop(labels=["zone_lat", "zone_lng"]).to_dict(),
            "geometry": {"type": "Polygon", "coordinates": [h3_zone_polygon(row["h3_zone"])]},
        }
        for _, row in grid.iterrows()
    ]
    geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")
    return grid
```

- [ ] **Step 3: Verify**

Run:

```powershell
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe analysis/build_hk_coverage_grid.py
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_hk_coverage_grid.py -q --basetemp C:\tmp\ghost-sweep-pytest
```

Expected: coverage CSV/GeoJSON exist and tests pass.

## Task 2: Build Discovery Mart And Feature Interactions

**Files:**
- Create: `analysis/build_fixed_grid_feature_mart.py`
- Create: `tests/test_fixed_grid_feature_mart.py`
- Modify: `analysis/data_discovery/` outputs via script only

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** This is deterministic data aggregation with clear schemas.
**Inputs Needed:** Task 1 grid output, `ghost_alerts.db`, existing discovery artifacts.
**Expected Output:** Derived CSV tables for `zone_daily_profile`, `zone_hour_profile`, `zone_recency_profile`, `zone_neighbor_context`, `zone_sparsity_profile`.
**Review Gate:** Tests prove fixed-grid zero-history zones remain in the mart with zeros/defaults.
**Escalation Trigger:** Escalate if aggregations create future leakage for training rows.

- [ ] **Step 1: Write failing mart tests**

Create `tests/test_fixed_grid_feature_mart.py`:

```python
import pandas as pd

from analysis.build_fixed_grid_feature_mart import build_zone_sparsity_profile


def test_sparsity_profile_preserves_zero_history_zone():
    grid = pd.DataFrame(
        [
            {"h3_zone": "a", "district": "Mong Kok", "region": "Kowloon West", "has_observed_history": True},
            {"h3_zone": "b", "district": "Sha Tin", "region": "New Territories East", "has_observed_history": False},
        ]
    )
    events = pd.DataFrame([{"h3_zone": "a", "create_dt": "2026-06-01 10:00:00"}])

    profile = build_zone_sparsity_profile(grid, events)

    zero = profile.loc[profile["h3_zone"] == "b"].iloc[0]
    assert zero["events"] == 0
    assert zero["coverage_class"] == "zero_history"
```

- [ ] **Step 2: Implement mart builder**

Create functions:

```python
build_zone_daily_profile(grid, events)
build_zone_hour_profile(grid, events)
build_zone_recency_profile(grid, events, as_of)
build_zone_neighbor_context(grid, events)
build_zone_sparsity_profile(grid, events)
write_fixed_grid_feature_mart()
```

Write files:

```text
analysis/data_discovery/zone_daily_profile_res8.csv
analysis/data_discovery/zone_hour_profile_res8.csv
analysis/data_discovery/zone_recency_profile_res8.csv
analysis/data_discovery/zone_neighbor_context_res8.csv
analysis/data_discovery/zone_sparsity_profile_res8.csv
```

- [ ] **Step 3: Verify**

Run:

```powershell
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe analysis/build_fixed_grid_feature_mart.py
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_fixed_grid_feature_mart.py -q --basetemp C:\tmp\ghost-sweep-pytest
```

Expected: mart files exist and zero-history test passes.

## Task 3: Train Fixed-Coverage Multi-Horizon Models

**Files:**
- Modify: `ghost_ranking_features.py`
- Modify: `analysis/run_model_iteration.py`
- Create: `tests/test_fixed_grid_training_rows.py`

**Agent Role:** Builder
**Suggested Model Tier:** high-capability
**Why This Tier:** This changes the model universe and evaluation comparability; leakage and sparse-label risks are real.
**Inputs Needed:** Fixed grid CSV, current multi-horizon model scripts.
**Expected Output:** Training data includes every fixed HK coverage cell at each snapshot, with `is_zero_history`/`has_observed_history` features and horizon metrics.
**Review Gate:** Tests prove observed-only and fixed-grid modes differ, zero-history rows exist, and no label window crosses evaluation boundary.
**Escalation Trigger:** Escalate if class balance becomes too sparse for 30m training or metrics become unstable.

- [ ] **Step 1: Write failing fixed-grid training tests**

Create `tests/test_fixed_grid_training_rows.py`:

```python
import pandas as pd

from ghost_ranking_features import build_zone_ranking_training_data


def test_training_data_can_include_fixed_zero_history_zone():
    events = [
        {"lat": 22.3154, "lng": 114.1698, "create_dt": f"2026-06-{day:02d} 10:00:00", "duration_min": 5, "report_count": 1, "total_upvotes": 0, "total_downvotes": 0}
        for day in range(1, 18)
    ]
    fixed_zones = pd.DataFrame(
        [
            {"h3_zone": "88411cb369fffff", "zone_lat": 22.3154, "zone_lng": 114.1698, "district": "Mong Kok", "region": "Kowloon West", "has_observed_history": True},
            {"h3_zone": "88411d2a9dfffff", "zone_lat": 22.45, "zone_lng": 114.15, "district": "Tai Po", "region": "New Territories East", "has_observed_history": False},
        ]
    )

    rows = build_zone_ranking_training_data(events, fixed_zones=fixed_zones)

    assert "is_zero_history" in rows.columns
    assert rows.loc[rows["zone_id"] == "88411d2a9dfffff", "is_zero_history"].eq(1).any()
```

- [ ] **Step 2: Add fixed-grid mode**

Change `build_zone_ranking_training_data()` signature:

```python
fixed_zones: pd.DataFrame | None = None,
```

If `fixed_zones` is provided:

```python
zones = sorted(set(fixed_zones["h3_zone"]))
zone_context = fixed_zones.set_index("h3_zone").to_dict("index")
```

For each row, use context defaults:

```python
"is_zero_history": int(not zone_context.get(zone_id, {}).get("has_observed_history", True)),
"has_observed_history": int(zone_context.get(zone_id, {}).get("has_observed_history", True)),
```

Add these to `NUMERIC_FEATURES` in `analysis/run_model_iteration.py`.

- [ ] **Step 3: Add fixed-grid model mode**

In `run_model_iteration()`:

```python
fixed_grid_path: Path | None = ROOT / "analysis" / "geo" / "hk_h3_coverage_res8.csv"
fixed_zones = pd.read_csv(fixed_grid_path) if fixed_grid_path and fixed_grid_path.exists() else None
```

Pass `fixed_zones=fixed_zones` into the builder.

Output fixed-grid-suffixed artifacts:

```text
analysis/fixed_grid_model_iteration_summary_30m_latest.csv
analysis/fixed_grid_iterated_zone_predictions_30m_latest.csv
```

or add `coverage_mode=fixed_grid` in metadata and manifest.

## Task 4: Build Traceable Dashboard Manifest

**Files:**
- Create: `analysis/build_dashboard_manifest.py`
- Create: `analysis/dashboard_manifest_latest.json`
- Create: `tests/test_dashboard_manifest.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** This is deterministic artifact inventory and schema validation.
**Inputs Needed:** Outputs from Tasks 1-3 and existing reports.
**Expected Output:** Manifest lists all source artifacts, schemas, timestamps, model run metadata, and dashboard panel inputs.
**Review Gate:** Test verifies all manifest paths exist and required schemas are declared.
**Escalation Trigger:** Escalate if MLflow artifact/run mapping is inconsistent with root artifacts.

- [ ] **Step 1: Write manifest test**

```python
import json
from pathlib import Path


def test_dashboard_manifest_paths_exist():
    manifest = json.loads(Path("analysis/dashboard_manifest_latest.json").read_text())
    for item in manifest["artifacts"]:
        assert Path(item["path"]).exists()
        assert item["kind"] in {"csv", "json", "geojson", "html", "joblib"}
```

- [ ] **Step 2: Implement manifest builder**

Include artifact groups:

```text
coverage_grid
feature_mart
data_discovery
multi_horizon_models
predictions
simulation
mlflow_tracking
reports
```

For each artifact, store:

```json
{"path": "...", "kind": "csv", "rows": 123, "columns": ["..."], "last_modified": "..."}
```

## Task 5: Build Spotfire-Like Static Dashboard

**Files:**
- Create: `analysis/build_spotfire_dashboard.py`
- Create: `analysis/spotfire_dashboard.html`
- Create: `tests/test_spotfire_dashboard.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** Static HTML/JS dashboard generation is scoped and can be tested with text/schema checks.
**Inputs Needed:** Dashboard manifest, feature mart, predictions, GeoJSON, reports.
**Expected Output:** One local HTML page with interactive filters and cross-linked artifacts.
**Review Gate:** HTML contains dashboard sections, embedded data, filter controls, and report links.
**Escalation Trigger:** Escalate if browser rendering or JS cross-filtering fails and needs visual debugging.

- [ ] **Step 1: Write dashboard smoke test**

```python
from pathlib import Path


def test_spotfire_dashboard_contains_required_sections():
    html = Path("analysis/spotfire_dashboard.html").read_text(encoding="utf-8")

    assert "Data Discovery" in html
    assert "Fixed HK Coverage" in html
    assert "Feature Interactions" in html
    assert "Experiment Traceability" in html
    assert "Model Evaluation" in html
    assert "Real Location Simulation" in html
```

- [ ] **Step 2: Implement dashboard generator**

Build a static HTML page with:

```text
Global filter bar: horizon, district, region, coverage class, run/model, target time
KPI cards: zones, zero-history share, event count, top-zone concentration, selected model, AP/lift/AUC
Map panel: SVG/GeoJSON hex grid colored by score or event density
Distribution panel: long-tail histogram, hourly pattern, day-of-week pattern
Feature panel: feature importance and feature interaction scatter/table
Experiment panel: horizon model metrics, artifact links, metadata JSON links
Simulation panel: location scenario table with 30m/1h/2h scores
Detail table: filtered zones with score, actual, counts, district, recency
```

Use vanilla JS embedded in the HTML. Avoid external network dependencies.

## Task 6: Verification

**Files:**
- No planned code edits.

**Agent Role:** Reviewer
**Suggested Model Tier:** low-cost
**Why This Tier:** Deterministic verification.
**Inputs Needed:** Completed Tasks 1-5.
**Expected Output:** Tests pass and dashboard artifacts are ready to open locally.
**Review Gate:** Main agent verifies commands and artifact existence.
**Escalation Trigger:** Escalate if visual/browser QA reveals broken interactions.

- [ ] **Step 1: Run focused tests**

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_hk_coverage_grid.py tests/test_fixed_grid_feature_mart.py tests/test_fixed_grid_training_rows.py tests/test_dashboard_manifest.py tests/test_spotfire_dashboard.py -q --basetemp C:\tmp\ghost-sweep-pytest
```

- [ ] **Step 2: Run full tests**

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest -q --basetemp C:\tmp\ghost-sweep-pytest
```

- [ ] **Step 3: Check artifacts**

```powershell
Get-Item analysis/geo/hk_h3_coverage_res8.csv
Get-Item analysis/data_discovery/zone_sparsity_profile_res8.csv
Get-Item analysis/dashboard_manifest_latest.json
Get-Item analysis/spotfire_dashboard.html
```

## Self-Review

Spec coverage: Covers fixed HK coverage grid, data discovery/features, model updates, MLOps traceability, and interactive Spotfire-like dashboard.

Placeholder scan: No placeholder tasks remain; each task includes file paths, expected behavior, and verification commands.

Type consistency: H3 zones use `h3_zone` in coverage artifacts and `zone_id` in prediction/model rows; manifest records schema mapping explicitly.

Dependency ordering: Grid first, mart second, fixed-grid training third, manifest fourth, dashboard fifth, verification last.

Agent independence: Data mart, dashboard manifest, and dashboard implementation can be delegated after grid contracts are stable. Modeling should wait for grid output.

Cost fit: Discovery/dashboard workers can use mid/low-cost models. Fixed-grid modeling uses high-capability due to leakage/sparsity risk.

Parallel safety: Initial read-only discovery was parallelized. Implementation should split only after Task 1 defines the grid contract.

Review gates: Each task has concrete tests or artifact checks.

## Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-fixed-hk-grid-analytics-dashboard.md`.

Recommended execution: Subagent-Driven. Dispatch focused agents by task after Task 1, use cheaper tiers for bounded data/dashboard work, reserve high-capability reasoning for fixed-grid modeling and leakage review, and checkpoint after each diff.

Alternative execution: Inline Execution. Execute the plan in this session with the same checkpoints, but with less parallelism.
