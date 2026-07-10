# Current Model Data Analysis HTML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static dated HTML report at `analysis\reports\2026-07-10\current-model-data-analysis.html` that packages the current-model data analysis as a dashboard summary followed by a full write-up.

**Architecture:** Add one focused report generator script that reuses the existing feature builders and target helpers to compute the already-approved analysis snapshot, renders a self-contained HTML document with inline CSS, and writes it into the dated reports folder. Keep the output static and single-file: no web server, no live reload, no external charting libraries, and no new experiment logic.

**Tech Stack:** Python, pandas, existing project analysis helpers, pathlib, json, html escaping, pytest.

---

## File Map

- **Create:** `analysis/build_current_model_data_analysis_report.py`
  - Gather the raw-feed, reconstructed-geo, Stage 1, and Stage 2 statistics
  - Format the dashboard summary and full narrative
  - Write the final static HTML file into the dated reports folder

- **Create:** `tests/test_current_model_data_analysis_report.py`
  - Verify the dated output path, the HTML structure, and key embedded strings
  - Use a tiny synthetic fixture so the test stays fast and deterministic

- **Modify:** `WORKLOG.md`
  - Record the implementation-plan milestone now
  - Record the implementation milestone when execution completes

- **Modify:** `WORKLOG_INDEX.md`
  - Add the implementation-plan milestone now
  - Add the implementation milestone later if execution adds a new top-level `##` entry

## Ground Rules

- Do **not** build a live page or call any local web service.
- Do **not** pull data out of chat text; recompute the approved analysis snapshot from the same project inputs.
- Do **not** add chart libraries or a front-end framework for a one-page report.
- Do **not** add a reporting framework, template engine, or reusable component system.
- Do **not** rerun model training; only recompute the lightweight profiling already used for the analysis write-up.
- Keep the output convention fixed at `analysis\reports\YYYY-MM-DD\current-model-data-analysis.html`.

---

### Task 1: Add a focused report-generator test

**Files:**
- Create: `tests/test_current_model_data_analysis_report.py`
- Test: `tests/test_current_model_data_analysis_report.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_current_model_data_analysis_report.py` with this content:

```python
from pathlib import Path

from analysis.build_current_model_data_analysis_report import (
    build_report_payload,
    output_path_for_date,
    render_report_html,
    write_report,
)


def test_render_report_writes_static_dated_html(tmp_path: Path):
    payload = build_report_payload(
        raw={
            "total_alerts": 10236,
            "date_range": ["2026-06-13 09:58:34", "2026-07-10 14:32:03"],
            "field_presence": {"lat": 10236, "lng": 10236, "region": 0, "district": 0},
            "last_7_daily": [
                ["2026-07-04", 18],
                ["2026-07-05", 639],
                ["2026-07-06", 556],
            ],
            "hour_top5": [[16, 919], [20, 901]],
            "dow": {"Wed": 1986, "Thu": 1662},
        },
        reconstructed_geo={
            "usable_events": 10236,
            "unique_zones": 880,
            "unique_regions": 5,
            "unique_districts": 18,
            "top_regions": [["Kowloon West", 3688], ["New Territories North", 2724]],
            "top_districts": [["Yau Tsim Mong", 2180], ["Yuen Long", 1329]],
        },
        stage1={
            "30m": {"positive_rate": 0.5, "rows": 486, "holdout_base_rate": 0.7463},
            "60m": {"positive_rate": 0.5556, "rows": 486, "holdout_base_rate": 0.7812},
            "120m": {"positive_rate": 0.6358, "rows": 486, "holdout_base_rate": 0.9259},
        },
        stage2={
            "feature_quality": {
                "rows": 279840,
                "unique_target_times": 318,
                "unique_zones": 880,
                "null_rate_top10": [["region", 0.2566]],
                "zero_share_top10": [["zone_event_count_1h", 0.9928]],
            },
            "30m": {
                "positive_rate": 0.0037,
                "sampled_neg_to_pos_ratio": 5.1,
                "rows": 279840,
                "positives": 1027,
            },
            "60m": {
                "positive_rate": 0.0073,
                "sampled_neg_to_pos_ratio": 5.1,
                "rows": 279840,
                "positives": 2031,
            },
            "120m": {
                "positive_rate": 0.0136,
                "sampled_neg_to_pos_ratio": 5.1,
                "rows": 279840,
                "positives": 3804,
            },
        },
        writeup={
            "executive_read": "The current model still fits the data, but Stage 1 is less selective in the hotter regime.",
            "what_so_what_now_what": {
                "what": "The raw feed lost admin text fields, but coordinates still support enrichment.",
                "so_what": "The accepted design still works, but DB refresh and geo backfill are now hard dependencies.",
                "now_what": "Refresh the DB, rerun the baseline, and keep the coordinate backfill path intact.",
            },
        },
    )

    html_text = render_report_html(payload)
    out_path = write_report(payload, root_dir=tmp_path, report_date="2026-07-10")

    assert output_path_for_date(tmp_path, "2026-07-10") == out_path
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == html_text
    assert "<title>Current Model Data Analysis</title>" in html_text
    assert "Stage 1 30m positive rate" in html_text
    assert "Stage 2 30m positive rate" in html_text
    assert "What / So What / Now What" in html_text
    assert "Yau Tsim Mong" in html_text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_current_model_data_analysis_report.py -q -p no:cacheprovider --basetemp .pytest_tmp_current_model_html_red
```

Expected: FAIL with `ModuleNotFoundError` or import failure because `analysis.build_current_model_data_analysis_report` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `analysis/build_current_model_data_analysis_report.py` with this starting implementation:

```python
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "analysis" / "reports"
REPORT_FILENAME = "current-model-data-analysis.html"


def output_path_for_date(root_dir: Path, report_date: str) -> Path:
    return Path(root_dir) / "analysis" / "reports" / report_date / REPORT_FILENAME


def build_report_payload(**sections):
    return sections


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_report_html(payload: dict) -> str:
    raw = payload["raw"]
    reconstructed = payload["reconstructed_geo"]
    stage1_30m = payload["stage1"]["30m"]
    stage2_30m = payload["stage2"]["30m"]
    writeup = payload["writeup"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Current Model Data Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f6f8fb; color: #172033; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card, .panel {{ background: #fff; border: 1px solid #dbe4ef; border-radius: 10px; padding: 16px; }}
    .metric {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #e6edf5; }}
    h1, h2, h3 {{ margin-top: 0; }}
  </style>
</head>
<body>
<main>
  <section class="panel">
    <h1>Current Model Data Analysis</h1>
    <p>Static snapshot of the current accepted model design: Stage 1 city-activity gating plus Stage 2 spatial ranking.</p>
  </section>
  <section class="grid">
    <div class="card"><div>Raw alert count</div><div class="metric">{raw["total_alerts"]}</div></div>
    <div class="card"><div>Usable enriched events</div><div class="metric">{reconstructed["usable_events"]}</div></div>
    <div class="card"><div>Active H3 zones</div><div class="metric">{reconstructed["unique_zones"]}</div></div>
    <div class="card"><div>Stage 1 30m positive rate</div><div class="metric">{_pct(stage1_30m["positive_rate"])}</div></div>
    <div class="card"><div>Stage 2 30m positive rate</div><div class="metric">{_pct(stage2_30m["positive_rate"])}</div></div>
    <div class="card"><div>Stage 2 sampled neg:pos</div><div class="metric">{stage2_30m["sampled_neg_to_pos_ratio"]:.1f}:1</div></div>
  </section>
  <section class="panel">
    <h2>Dashboard summary</h2>
    <p>Top district: {escape(reconstructed["top_districts"][0][0])}</p>
    <p>Top region: {escape(reconstructed["top_regions"][0][0])}</p>
    <p>Most complete raw fields: lat/lng/create_dt; district/region missing in raw payload.</p>
  </section>
  <section class="panel">
    <h2>Full write-up</h2>
    <h3>Executive read</h3>
    <p>{escape(writeup["executive_read"])}</p>
    <h3>What / So What / Now What</h3>
    <p><strong>What:</strong> {escape(writeup["what_so_what_now_what"]["what"])}</p>
    <p><strong>So What:</strong> {escape(writeup["what_so_what_now_what"]["so_what"])}</p>
    <p><strong>Now What:</strong> {escape(writeup["what_so_what_now_what"]["now_what"])}</p>
  </section>
</main>
</body>
</html>"""


def write_report(payload: dict, root_dir: Path | None = None, report_date: str = "2026-07-10") -> Path:
    root = Path(root_dir) if root_dir is not None else ROOT
    out_path = output_path_for_date(root, report_date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report_html(payload), encoding="utf-8")
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_current_model_data_analysis_report.py -q -p no:cacheprovider --basetemp .pytest_tmp_current_model_html_green
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add analysis\build_current_model_data_analysis_report.py tests\test_current_model_data_analysis_report.py
git commit -m "feat: scaffold current model analysis html report"
```

---

### Task 2: Replace the scaffold payload with real project analysis data

**Files:**
- Modify: `analysis/build_current_model_data_analysis_report.py`
- Test: `tests/test_current_model_data_analysis_report.py`

- [ ] **Step 1: Write the failing real-data payload test**

Append this test to `tests/test_current_model_data_analysis_report.py`:

```python
import json

from analysis.build_current_model_data_analysis_report import collect_report_data


def test_collect_report_data_uses_project_inputs(tmp_path: Path):
    source = tmp_path / "ghost_alerts.json"
    source.write_text(
        json.dumps(
            {
                "alerts": {
                    "1": {
                        "lat": 22.3154,
                        "lng": 114.1698,
                        "create_dt": "2026-07-09 10:00:00",
                        "address": "Central",
                        "name": "A",
                    },
                    "2": {
                        "lat": 22.3160,
                        "lng": 114.1703,
                        "create_dt": "2026-07-10 11:00:00",
                        "address": "Wan Chai",
                        "name": "B",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    payload = collect_report_data(alerts_path=source)

    assert payload["raw"]["total_alerts"] == 2
    assert payload["reconstructed_geo"]["usable_events"] == 2
    assert "30m" in payload["stage1"]
    assert "30m" in payload["stage2"]
    assert payload["current_model_design"]["stage2_numeric_feature_count"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_current_model_data_analysis_report.py::test_collect_report_data_uses_project_inputs -q -p no:cacheprovider --basetemp .pytest_tmp_current_model_html_real_red
```

Expected: FAIL because `collect_report_data()` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

In `analysis/build_current_model_data_analysis_report.py`, replace the scaffold-only logic with these imports:

```python
import json
from collections import Counter
from datetime import datetime
from statistics import mean

import pandas as pd

from analysis.run_model_iteration import target_for_horizon
from analysis.run_zone_ranking_experiment import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from analysis.two_stage_splits import make_positive_count_holdout
from ghost_activity_features import activity_target_for_horizon, build_activity_training_data
from ghost_ranking_features import (
    build_zone_ranking_training_data,
    enrich_events_with_zones,
    sample_spatial_training_rows,
)
```

Add this function below `build_report_payload()`:

```python
def collect_report_data(alerts_path: Path | None = None) -> dict:
    source = Path(alerts_path) if alerts_path is not None else ROOT / "ghost_alerts.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing alerts source: {source}")

    data = json.loads(source.read_text(encoding="utf-8"))
    alerts = data.get("alerts")
    if not isinstance(alerts, dict) or not alerts:
        raise ValueError("Expected ghost_alerts.json to contain a non-empty alerts dictionary.")

    raw_events = list(alerts.values())
    enriched = enrich_events_with_zones(raw_events)
    if not enriched:
        raise ValueError("No enriched events were produced from the current raw alerts.")

    fmt = "%Y-%m-%d %H:%M:%S"
    raw_dts = [datetime.strptime(event["create_dt"], fmt) for event in raw_events if event.get("create_dt")]
    raw_dates = Counter(dt.date().isoformat() for dt in raw_dts)
    reconstructed_regions = Counter(str(event.get("region") or "Unknown") for event in enriched)
    reconstructed_districts = Counter(str(event.get("district") or "Unknown") for event in enriched)
    zone_counts = Counter(str(event.get("h3_zone") or "Unknown") for event in enriched)

    payload = {
        "raw": {
            "total_alerts": len(raw_events),
            "date_range": [str(min(raw_dts)), str(max(raw_dts))],
            "field_presence": {
                key: sum(1 for event in raw_events if event.get(key) not in (None, "", [], {}))
                for key in ["lat", "lng", "create_dt", "region", "district", "sub_district", "title", "name", "address"]
            },
            "last_7_daily": sorted(raw_dates.items())[-7:],
            "hour_top5": Counter(dt.hour for dt in raw_dts).most_common(5),
            "dow": dict(sorted(Counter(dt.strftime("%a") for dt in raw_dts).items())),
        },
        "reconstructed_geo": {
            "usable_events": len(enriched),
            "unique_zones": len(zone_counts),
            "unique_regions": len(reconstructed_regions),
            "unique_districts": len(reconstructed_districts),
            "top_regions": reconstructed_regions.most_common(8),
            "top_districts": reconstructed_districts.most_common(12),
            "top_zones": zone_counts.most_common(12),
        },
        "current_model_design": {
            "stage2_numeric_feature_count": len(NUMERIC_FEATURES),
            "stage2_categorical_feature_count": len(CATEGORICAL_FEATURES),
            "stage2_numeric_features": NUMERIC_FEATURES,
            "stage2_categorical_features": CATEGORICAL_FEATURES,
        },
        "stage1": {},
        "stage2": {},
    }

    for horizon in (30, 60, 120):
        activity = build_activity_training_data(raw_events, horizon_minutes=horizon)
        activity_target = activity_target_for_horizon(horizon)
        holdout = make_positive_count_holdout(activity, target_col=activity_target, min_positives=50)
        payload["stage1"][f"{horizon}m"] = {
            "rows": int(len(activity)),
            "positives": int(activity[activity_target].sum()),
            "positive_rate": float(activity[activity_target].mean()),
            "avg_future_count": float(activity["event_count_next_horizon"].mean()),
            "holdout_rows": int(holdout.metadata["holdout_rows"]),
            "holdout_positives": int(holdout.metadata["holdout_positives"]),
            "holdout_base_rate": float(holdout.metadata["holdout_base_rate"]),
        }

    stage2_30m = build_zone_ranking_training_data(
        raw_events,
        horizon_minutes=30,
        target_col=target_for_horizon(30),
    )
    payload["stage2"]["feature_quality"] = {
        "rows": int(len(stage2_30m)),
        "unique_target_times": int(stage2_30m["target_time"].nunique()),
        "unique_zones": int(stage2_30m["zone_id"].nunique()),
        "district_cardinality": int(stage2_30m["district"].nunique(dropna=False)),
        "region_cardinality": int(stage2_30m["region"].nunique(dropna=False)),
        "null_rate_top10": sorted(
            [
                (column, float(stage2_30m[column].isna().mean()))
                for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES
                if column in stage2_30m.columns
            ],
            key=lambda item: item[1],
            reverse=True,
        )[:10],
    }

    for horizon in (30, 60, 120):
        target = target_for_horizon(horizon)
        spatial = stage2_30m if horizon == 30 else build_zone_ranking_training_data(
            raw_events,
            horizon_minutes=horizon,
            target_col=target,
        )
        sampled = sample_spatial_training_rows(spatial, target_col=target)
        holdout = make_positive_count_holdout(spatial, target_col=target, min_positives=50)
        payload["stage2"][f"{horizon}m"] = {
            "rows": int(len(spatial)),
            "positives": int(spatial[target].sum()),
            "positive_rate": float(spatial[target].mean()),
            "sampled_rows": int(len(sampled)),
            "sampled_positives": int(sampled[target].sum()),
            "sampled_positive_rate": float(sampled[target].mean()),
            "sampled_neg_to_pos_ratio": float((len(sampled) - sampled[target].sum()) / max(sampled[target].sum(), 1)),
            "holdout_rows": int(holdout.metadata["holdout_rows"]),
            "holdout_positives": int(holdout.metadata["holdout_positives"]),
            "holdout_base_rate": float(holdout.metadata["holdout_base_rate"]),
        }

    payload["writeup"] = {
        "executive_read": "The current model still fits the data, but Stage 1 is less selective in the hotter regime.",
        "what_so_what_now_what": {
            "what": "The raw feed lost admin text fields, but coordinates still support enrichment.",
            "so_what": "The accepted design still works, but DB refresh and geo backfill are now hard dependencies.",
            "now_what": "Refresh the DB, rerun the baseline, and keep the coordinate backfill path intact.",
        },
    }
    return payload
```

Update the header section of `render_report_html()` so it references `payload["current_model_design"]` and the Stage 1 / Stage 2 sections. Also add a small helper to render summary rows:

```python
def _table_rows(rows: list[list | tuple]) -> str:
    return "".join(
        f"<tr><td>{escape(str(left))}</td><td>{escape(str(right))}</td></tr>"
        for left, right in rows
    )
```

Add the dashboard panels under `<section class="panel">`:

```python
  <section class="panel">
    <h2>Raw feed completeness</h2>
    <table>
      <thead><tr><th>Field</th><th>Present rows</th></tr></thead>
      <tbody>{_table_rows(payload["raw"]["field_presence"].items())}</tbody>
    </table>
  </section>
  <section class="panel">
    <h2>Reconstructed geography</h2>
    <table>
      <thead><tr><th>District</th><th>Alerts</th></tr></thead>
      <tbody>{_table_rows(payload["reconstructed_geo"]["top_districts"][:8])}</tbody>
    </table>
  </section>
  <section class="panel">
    <h2>Stage 1 label balance</h2>
    <table>
      <thead><tr><th>Horizon</th><th>Positive rate</th><th>Holdout base rate</th></tr></thead>
      <tbody>
        <tr><td>30m</td><td>{_pct(payload["stage1"]["30m"]["positive_rate"])}</td><td>{_pct(payload["stage1"]["30m"]["holdout_base_rate"])}</td></tr>
        <tr><td>1h</td><td>{_pct(payload["stage1"]["60m"]["positive_rate"])}</td><td>{_pct(payload["stage1"]["60m"]["holdout_base_rate"])}</td></tr>
        <tr><td>2h</td><td>{_pct(payload["stage1"]["120m"]["positive_rate"])}</td><td>{_pct(payload["stage1"]["120m"]["holdout_base_rate"])}</td></tr>
      </tbody>
    </table>
  </section>
  <section class="panel">
    <h2>Stage 2 label balance</h2>
    <table>
      <thead><tr><th>Horizon</th><th>Positive rate</th><th>Sampled neg:pos</th></tr></thead>
      <tbody>
        <tr><td>30m</td><td>{_pct(payload["stage2"]["30m"]["positive_rate"])}</td><td>{payload["stage2"]["30m"]["sampled_neg_to_pos_ratio"]:.1f}:1</td></tr>
        <tr><td>1h</td><td>{_pct(payload["stage2"]["60m"]["positive_rate"])}</td><td>{payload["stage2"]["60m"]["sampled_neg_to_pos_ratio"]:.1f}:1</td></tr>
        <tr><td>2h</td><td>{_pct(payload["stage2"]["120m"]["positive_rate"])}</td><td>{payload["stage2"]["120m"]["sampled_neg_to_pos_ratio"]:.1f}:1</td></tr>
      </tbody>
    </table>
  </section>
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_current_model_data_analysis_report.py::test_collect_report_data_uses_project_inputs -q -p no:cacheprovider --basetemp .pytest_tmp_current_model_html_real_green
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add analysis\build_current_model_data_analysis_report.py tests\test_current_model_data_analysis_report.py
git commit -m "feat: compute current model analysis report data"
```

---

### Task 3: Add the report CLI entrypoint and verify the dated artifact

**Files:**
- Modify: `analysis/build_current_model_data_analysis_report.py`
- Modify: `tests/test_current_model_data_analysis_report.py`

- [ ] **Step 1: Write the failing CLI smoke test**

Append this test to `tests/test_current_model_data_analysis_report.py`:

```python
from analysis.build_current_model_data_analysis_report import main


def test_main_writes_default_dated_report(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("analysis.build_current_model_data_analysis_report.ROOT", tmp_path)
    source = tmp_path / "ghost_alerts.json"
    source.write_text(
        '{"alerts":{"1":{"lat":22.3154,"lng":114.1698,"create_dt":"2026-07-09 10:00:00","address":"Central","name":"A"}}}',
        encoding="utf-8",
    )

    output = main(["--report-date", "2026-07-10", "--alerts-path", str(source)])

    assert output == tmp_path / "analysis" / "reports" / "2026-07-10" / "current-model-data-analysis.html"
    assert output.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_current_model_data_analysis_report.py::test_main_writes_default_dated_report -q -p no:cacheprovider --basetemp .pytest_tmp_current_model_html_cli_red
```

Expected: FAIL because `main()` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

At the bottom of `analysis/build_current_model_data_analysis_report.py`, add:

```python
import argparse


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-date", default="2026-07-10")
    parser.add_argument("--alerts-path", default=str(ROOT / "ghost_alerts.json"))
    args = parser.parse_args(argv)

    payload = collect_report_data(alerts_path=Path(args.alerts_path))
    output = write_report(payload, root_dir=ROOT, report_date=args.report_date)
    print(output)
    return output


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and generate the real report**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_current_model_data_analysis_report.py -q -p no:cacheprovider --basetemp .pytest_tmp_current_model_html_full
.venv\Scripts\python.exe analysis\build_current_model_data_analysis_report.py --report-date 2026-07-10
```

Expected:

- pytest PASS
- the script prints `analysis\reports\2026-07-10\current-model-data-analysis.html`

Then verify the file contains the required sections:

```powershell
Select-String -Path analysis\reports\2026-07-10\current-model-data-analysis.html -Pattern "Current Model Data Analysis","Stage 1 30m positive rate","Stage 2 30m positive rate","What / So What / Now What"
```

Expected: all four strings are found.

- [ ] **Step 5: Commit**

```powershell
git add analysis\build_current_model_data_analysis_report.py tests\test_current_model_data_analysis_report.py analysis\reports\2026-07-10\current-model-data-analysis.html
git commit -m "feat: generate current model analysis html report"
```

---

### Task 4: Record the implementation milestone

**Files:**
- Modify: `WORKLOG.md`
- Modify: `WORKLOG_INDEX.md`

- [ ] **Step 1: Add the implementation worklog entry**

Append a new top-level `##` entry to `WORKLOG.md` using this exact structure:

```markdown
## 2026-07-10 Current model data analysis HTML implementation

Current objective:
- Generate the static dated HTML report for the current-model data analysis.

Files inspected:
- `analysis/build_current_model_data_analysis_report.py`
- `tests/test_current_model_data_analysis_report.py`
- `analysis/reports/2026-07-10/current-model-data-analysis.html`

Files changed:
- `analysis/build_current_model_data_analysis_report.py`
- `tests/test_current_model_data_analysis_report.py`
- `analysis/reports/2026-07-10/current-model-data-analysis.html`
- `WORKLOG.md`
- `WORKLOG_INDEX.md`

Commands run:
- `.venv\Scripts\python.exe -m pytest tests\test_current_model_data_analysis_report.py -q -p no:cacheprovider --basetemp .pytest_tmp_current_model_html_full`
- `.venv\Scripts\python.exe analysis\build_current_model_data_analysis_report.py --report-date 2026-07-10`
- `Select-String -Path analysis\reports\2026-07-10\current-model-data-analysis.html -Pattern "Current Model Data Analysis","Stage 1 30m positive rate","Stage 2 30m positive rate","What / So What / Now What"`

Test results:
- Report test suite passed.
- The dated HTML file was created and contained the required dashboard and write-up sections.

Blockers:
- None.

Next steps:
- Share the generated report path with the user.
```

- [ ] **Step 2: Update the worklog index**

Add a matching line to `WORKLOG_INDEX.md`:

```markdown
- `2026-07-10 Current model data analysis HTML implementation` — Logged the static report generator, dated HTML artifact creation, and final verification for the current-model analysis page. `WORKLOG.md` lines <fill after write>
```

- [ ] **Step 3: Run diff check**

Run:

```powershell
git --no-pager diff --check -- WORKLOG.md WORKLOG_INDEX.md
```

Expected: no output.

- [ ] **Step 4: Commit**

```powershell
git add WORKLOG.md WORKLOG_INDEX.md
git commit -m "docs: log current model html report"
```

---

## Spec Coverage Check

- **Static single-file HTML report:** covered by Tasks 1-3.
- **Dated folder under `analysis\reports\YYYY-MM-DD`:** covered by Tasks 1 and 3.
- **Dashboard summary first, write-up below:** covered by Tasks 1 and 2.
- **Reuse existing analysis logic rather than chat text:** covered by Task 2.
- **Standalone local file with no external dependencies:** covered by Tasks 1 and 3.
- **Worklog continuity:** covered by Task 4.

## Placeholder Scan

- No `TODO`, `TBD`, or “implement later” placeholders remain.
- The only deliberate fill-in is the worklog line span in Task 4 Step 2, because it is unknowable until the entry is written; compute it immediately after writing the entry before committing.

## Type Consistency Check

- `collect_report_data()` returns the same payload shape that `render_report_html()` consumes.
- `output_path_for_date()` and `write_report()` both target `analysis\reports\YYYY-MM-DD\current-model-data-analysis.html`.
- Horizon keys are consistently `30m`, `60m`, and `120m` in the plan, matching the proposed implementation.
