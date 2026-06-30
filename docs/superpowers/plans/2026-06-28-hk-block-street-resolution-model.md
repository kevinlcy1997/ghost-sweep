# HK Block/Street Resolution Model Plan

> This plan is for agentic workers. Execute task-by-task, keep changes scoped to the listed files, run the listed verification commands, and stop at review gates before continuing. This plan includes cost-aware model routing; use the cheapest suggested tier that can satisfy each task's review gate, and escalate only when the listed escalation trigger occurs.

## Goal

Turn the current Hong Kong H3-resolution-8 zone model into a reproducible block/street-level evaluation workflow. The first implementation milestone is not to blindly switch to smaller cells; it is to compare H3 resolutions 8, 9, and 10 with sparsity and ranking metrics, then generate a higher-resolution local SVG map from the best practical output.

## Current Context

Current H3 behavior is anchored in `ghost_zones.py`, where `DEFAULT_H3_RESOLUTION` is `8`, `compute_h3_zone()` assigns an H3 cell, and `assign_zone()` enriches cleaned events with `h3_zone`, centroid, district, and region fields.

Training data is built in `ghost_ranking_features.py` via `build_zone_ranking_training_data()`. The target is currently `alert_next_2h`, with rows keyed by `zone_id` and `target_time`. Features are past-only counts and district context.

The model runner is `analysis/run_zone_ranking_experiment.py`. It writes `analysis/zone_predictions_latest.csv`, `analysis/zone_feature_ranking_latest.csv`, `analysis/best_zone_model.joblib`, and `ghost_zone_forecast.geojson`, then logs MLflow metrics.

The geography survey is `analysis/survey_hk_geography.py`, which writes `analysis/geo/hk_zone_summary.csv` and `analysis/geo/hk_zone_summary.html`.

The local map is `analysis/make_zone_map.py`, which reads `ghost_zone_forecast.geojson` and writes `analysis/zone_forecast_map.html`.

## Agent and Model Routing

| Workstream | Agent Role | Suggested Model Tier | Cost Control | Review Gate |
| --- | --- | --- | --- | --- |
| Resolution parameterization | Builder | mid-tier | Edit only H3 assignment, training runner, survey, and tests | Tests prove resolution can be selected without changing default behavior |
| Resolution comparison runner | Builder | mid-tier | New analysis script plus focused tests; reuse existing runner logic | CSV/HTML report lists sparsity and ranking metrics for res 8/9/10 |
| Map/report integration | Builder | mid-tier | Modify map generator to consume selected GeoJSON path and expose resolution metadata | Local HTML shows high-resolution polygons and report links |
| Verification and review | Reviewer | low-cost | Run listed commands only | Exact command results and residual risks are recorded |

## Technical Strategy

Use H3 resolution as an experiment parameter everywhere zone IDs are produced. Keep resolution 8 as the default so existing outputs and tests remain compatible. Add a comparison script that runs the same leakage-safe target construction across H3 resolutions 8, 9, and 10, records sparsity diagnostics, trains the existing candidate models, and selects the resolution with the best balance of `precision_at_20`, `top_decile_lift`, active-zone count, and one-off-zone rate.

Do not fetch OSM road-network data in this milestone. Street-segment snapping should be the next milestone if H3 res 9/10 still cannot produce actionable block-level precision without excessive sparsity.

## Task 1: Parameterize H3 Resolution

**Files:**
- Modify: `ghost_zones.py`
- Modify: `ghost_ranking_features.py`
- Modify: `analysis/survey_hk_geography.py`
- Modify: `analysis/run_zone_ranking_experiment.py`
- Modify: `tests/test_ghost_zones.py`
- Modify: `tests/test_ghost_ranking_features.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** The task edits existing modules with clear patterns and a small public API change.
**Inputs Needed:** This task section and the listed files.
**Expected Output:** Resolution is configurable, default behavior stays resolution 8, and tests cover resolution 9/10 assignment.
**Review Gate:** Main agent verifies that old callers still work and no target leakage is introduced.
**Escalation Trigger:** Escalate to high-capability if changing resolution alters timestamp target semantics or breaks cleaned-data assumptions.

- [ ] **Step 1: Write failing tests for configurable resolution**

Add tests equivalent to:

```python
from ghost_zones import compute_h3_zone, assign_zone


def test_compute_h3_zone_accepts_resolution():
    res8 = compute_h3_zone(22.3154, 114.1698, resolution=8)
    res9 = compute_h3_zone(22.3154, 114.1698, resolution=9)

    assert res8 != res9
    assert len(res9) >= len(res8)


def test_assign_zone_accepts_resolution():
    row = assign_zone(22.3154, 114.1698, resolution=9)

    assert row["h3_resolution"] == 9
    assert row["h3_zone"] == compute_h3_zone(22.3154, 114.1698, resolution=9)
```

Run:

```powershell
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_ghost_zones.py -q --basetemp .pytest_tmp
```

Expected result before implementation: failure because `resolution` is not accepted or `h3_resolution` is missing.

- [ ] **Step 2: Implement resolution parameters**

Update `ghost_zones.py` so the public functions accept a resolution while preserving defaults:

```python
DEFAULT_H3_RESOLUTION = 8


def compute_h3_zone(lat: float, lng: float, resolution: int = DEFAULT_H3_RESOLUTION) -> str:
    return h3.latlng_to_cell(lat, lng, resolution)


def assign_zone(lat: float, lng: float, resolution: int = DEFAULT_H3_RESOLUTION) -> dict[str, object]:
    zone_id = compute_h3_zone(float(lat), float(lng), resolution=resolution)
    zone_lat, zone_lng = h3_zone_centroid(zone_id)
    district, region = district_for_point(float(lat), float(lng))
    return {
        "h3_zone": zone_id,
        "h3_resolution": resolution,
        "zone_lat": zone_lat,
        "zone_lng": zone_lng,
        "district": district,
        "region": region,
    }
```

Update `ghost_ranking_features.py` so `build_zone_ranking_training_data(..., resolution=DEFAULT_H3_RESOLUTION)` passes `resolution` into `assign_zone()` and carries `h3_resolution` into the output rows.

- [ ] **Step 3: Update analysis scripts**

Update `analysis/survey_hk_geography.py` to expose:

```python
def build_zone_summary(resolution: int = DEFAULT_H3_RESOLUTION) -> pd.DataFrame:
    ...
```

Write outputs using resolution-specific names:

```python
csv_path = OUT_DIR / f"hk_zone_summary_res{resolution}.csv"
html_path = OUT_DIR / f"hk_zone_summary_res{resolution}.html"
```

Also keep the existing latest aliases:

```python
summary.to_csv(OUT_DIR / "hk_zone_summary.csv", index=False)
html_path_latest = OUT_DIR / "hk_zone_summary.html"
html_path_latest.write_text(html, encoding="utf-8")
```

Update `analysis/run_zone_ranking_experiment.py` so `run_experiment(resolution: int = DEFAULT_H3_RESOLUTION)` passes resolution into the training-data builder, logs `h3_resolution`, and writes resolution-specific artifacts:

```python
predictions_path = OUTPUT_DIR / f"zone_predictions_res{resolution}_latest.csv"
geojson_path = ROOT / f"ghost_zone_forecast_res{resolution}.geojson"
report_path = OUTPUT_DIR / f"zone_ranking_report_res{resolution}_{timestamp}.html"
mlflow.log_param("zone_type", f"h3_resolution_{resolution}")
mlflow.log_param("h3_resolution", resolution)
```

Also keep current aliases for the selected/default run:

```python
predictions.to_csv(PREDICTIONS_PATH, index=False)
GEOJSON_PATH.write_text(json.dumps(feature_collection), encoding="utf-8")
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_ghost_zones.py tests/test_ghost_ranking_features.py -q --basetemp .pytest_tmp
```

Expected result: all selected tests pass.

## Task 2: Add Resolution Comparison Experiment

**Files:**
- Create: `analysis/run_resolution_comparison.py`
- Create: `tests/test_resolution_comparison.py`
- Modify: `analysis/run_zone_ranking_experiment.py`

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** This is a new orchestration script that should reuse existing experiment functions.
**Inputs Needed:** Task 1 output, current MLflow runner, and ranking metrics.
**Expected Output:** A repeatable script creates `analysis/resolution_comparison_latest.csv` and `analysis/resolution_comparison_report.html`.
**Review Gate:** Main agent checks that the script reuses the existing leakage-safe builder and does not duplicate modeling logic unnecessarily.
**Escalation Trigger:** Escalate to high-capability if model outputs differ in a way that suggests label leakage or invalid train/test splits.

- [ ] **Step 1: Extract reusable runner return value**

Modify `analysis/run_zone_ranking_experiment.py` so `run_experiment()` returns a dictionary like:

```python
return {
    "resolution": resolution,
    "best_model": best_model_name,
    "metrics": best_metrics,
    "predictions_path": str(predictions_path),
    "geojson_path": str(geojson_path),
    "report_path": str(report_path),
    "training_rows": int(len(df)),
    "active_zones": int(df["zone_id"].nunique()),
    "positive_rate": float(df["alert_next_2h"].mean()),
}
```

Keep CLI behavior unchanged:

```python
if __name__ == "__main__":
    run_experiment()
```

- [ ] **Step 2: Write comparison helper tests**

Add tests for pure helper functions in `tests/test_resolution_comparison.py`:

```python
from analysis.run_resolution_comparison import choose_practical_resolution


def test_choose_practical_resolution_penalizes_sparse_one_off_cells():
    rows = [
        {"resolution": 8, "precision_at_20": 0.80, "top_decile_lift": 4.0, "one_off_zone_rate": 0.10, "active_zones": 100},
        {"resolution": 9, "precision_at_20": 0.85, "top_decile_lift": 4.2, "one_off_zone_rate": 0.25, "active_zones": 250},
        {"resolution": 10, "precision_at_20": 0.90, "top_decile_lift": 4.4, "one_off_zone_rate": 0.80, "active_zones": 900},
    ]

    chosen = choose_practical_resolution(rows)

    assert chosen["resolution"] == 9
```

Run:

```powershell
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_resolution_comparison.py -q --basetemp .pytest_tmp
```

Expected result before implementation: failure because the module does not exist.

- [ ] **Step 3: Implement comparison script**

Create `analysis/run_resolution_comparison.py` with:

```python
RESOLUTIONS = [8, 9, 10]


def choose_practical_resolution(rows: list[dict[str, object]]) -> dict[str, object]:
    candidates = [row for row in rows if float(row["one_off_zone_rate"]) <= 0.60]
    if not candidates:
        candidates = rows
    return max(
        candidates,
        key=lambda row: (
            float(row["precision_at_20"]),
            float(row["top_decile_lift"]),
            -float(row["one_off_zone_rate"]),
        ),
    )
```

For each resolution, call `build_zone_summary(resolution=resolution)` and `run_experiment(resolution=resolution, write_latest_alias=False)` if Task 1 adds that flag. Build a row with:

```python
{
    "resolution": resolution,
    "active_zones": active_zones,
    "median_events_per_zone": median_events_per_zone,
    "one_off_zone_rate": one_off_zone_rate,
    "positive_rate": positive_rate,
    "precision_at_20": precision_at_20,
    "precision_at_50": precision_at_50,
    "top_decile_lift": top_decile_lift,
    "average_precision": average_precision,
    "roc_auc": roc_auc,
    "geojson_path": geojson_path,
    "report_path": report_path,
}
```

Write:

```python
analysis/resolution_comparison_latest.csv
analysis/resolution_comparison_report.html
analysis/resolution_choice_latest.json
```

- [ ] **Step 4: Run comparison**

Run:

```powershell
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe analysis/run_resolution_comparison.py
```

Expected result: CSV, HTML, and JSON choice files are created. The chosen resolution is printed with the selected metrics.

## Task 3: Generate Selected High-Resolution Map

**Files:**
- Modify: `analysis/make_zone_map.py`
- Modify: `analysis/zone_model_visual_explainer.html` only if the report link needs to be updated manually

**Agent Role:** Builder
**Suggested Model Tier:** mid-tier
**Why This Tier:** The map generator is local HTML/SVG generation with existing code structure.
**Inputs Needed:** Task 2 output, especially `analysis/resolution_choice_latest.json`.
**Expected Output:** `analysis/zone_forecast_map.html` uses the chosen resolution GeoJSON and displays resolution metadata.
**Review Gate:** Main agent opens or statically checks the HTML to confirm polygons and metadata are embedded.
**Escalation Trigger:** Escalate to high-capability if the selected output is too dense for the SVG to render or coordinate projection breaks.

- [ ] **Step 1: Add selected-input logic**

Update `analysis/make_zone_map.py` so it reads `analysis/resolution_choice_latest.json` when available:

```python
CHOICE_PATH = ROOT / "analysis" / "resolution_choice_latest.json"


def resolve_geojson_path() -> Path:
    if CHOICE_PATH.exists():
        choice = json.loads(CHOICE_PATH.read_text(encoding="utf-8"))
        path = ROOT / choice["geojson_path"]
        if path.exists():
            return path
    return GEOJSON_PATH
```

Use the resolved path in `main()`, and include selected resolution, active zones, one-off-zone rate, and precision metrics in the HTML header.

- [ ] **Step 2: Regenerate selected map**

Run:

```powershell
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe analysis/make_zone_map.py
```

Expected result: `analysis/zone_forecast_map.html` is updated and references the selected resolution output.

## Task 4: Full Verification

**Files:**
- No planned code edits

**Agent Role:** Reviewer
**Suggested Model Tier:** low-cost
**Why This Tier:** This task runs deterministic commands and summarizes pass/fail evidence.
**Inputs Needed:** Completed Tasks 1-3.
**Expected Output:** Passing tests and generated artifacts listed below.
**Review Gate:** Main agent checks command output before marking the goal complete.
**Escalation Trigger:** Escalate to mid-tier if a test fails and the failure has an obvious local fix; escalate to high-capability if model metrics suggest leakage or invalid comparisons.

- [ ] **Step 1: Run full test suite**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe -m pytest -q --basetemp .pytest_tmp
```

Expected result: all tests pass.

- [ ] **Step 2: Confirm artifacts**

Confirm these files exist and are non-empty:

```powershell
Get-Item analysis/resolution_comparison_latest.csv
Get-Item analysis/resolution_comparison_report.html
Get-Item analysis/resolution_choice_latest.json
Get-Item analysis/zone_forecast_map.html
```

Expected result: each file has a non-zero length.

## Self-Review

Spec coverage: The plan addresses the prior recommendation to compare H3 res 8/9/10 before committing to true road-segment snapping. It keeps the current `alert_next_2h` target and existing MLflow/ranking workflow.

Placeholder scan: No task uses TBD, TODO, or unspecified edge-case work. Every task lists files, commands, expected outputs, and review gates.

Type consistency: Resolution is consistently passed as `resolution: int`; output files use `res{resolution}`; choice rows use dictionary fields consumed by the map.

Dependency ordering: Task 1 enables resolution selection, Task 2 compares resolutions, Task 3 maps the selected output, and Task 4 verifies everything.

Agent independence: Each task has enough context and named files for a fresh worker. Task 2 depends on Task 1 output, and Task 3 depends on Task 2 output explicitly.

Cost fit: All builder work is mid-tier because it touches real code and tests but follows existing patterns. Verification is low-cost. High-capability is reserved for leakage, target semantics, or unexpected modeling failures.

Parallel safety: These tasks should run sequentially because Task 2 depends on Task 1 and Task 3 depends on Task 2. Within Task 4, artifact checks can run after tests.

Review gates: Each task has concrete pass/fail commands or artifact checks before the next task begins.

## Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-hk-block-street-resolution-model.md`.

Recommended execution: Subagent-Driven. I dispatch focused agents by task, use cheaper tiers for bounded read/review/mechanical work, reserve high-capability reasoning for architecture and integration review, and checkpoint after each diff.

Alternative execution: Inline Execution. I execute the plan in this session with the same checkpoints, but with less parallelism.
