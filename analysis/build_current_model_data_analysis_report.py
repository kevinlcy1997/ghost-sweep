from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.run_model_iteration import target_for_horizon
from analysis.run_zone_ranking_experiment import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from analysis.two_stage_splits import make_positive_count_holdout
from ghost_activity_features import activity_target_for_horizon, build_activity_training_data
from ghost_ranking_features import (
    build_zone_ranking_training_data,
    enrich_events_with_zones,
    sample_spatial_training_rows,
)

REPORTS_DIR = ROOT / "analysis" / "reports"
REPORT_FILENAME = "current-model-data-analysis.html"


def output_path_for_date(root_dir: Path, report_date: str) -> Path:
    return Path(root_dir) / "analysis" / "reports" / report_date / REPORT_FILENAME


def build_report_payload(**sections):
    return sections


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _table_rows(rows) -> str:
    return "".join(
        f"<tr><td>{escape(str(left))}</td><td>{escape(str(right))}</td></tr>"
        for left, right in rows
    )


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).mean())


def _safe_sum(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _safe_holdout(frame: pd.DataFrame, target_col: str) -> dict[str, float | int]:
    if frame.empty or target_col not in frame:
        return {
            "holdout_rows": 0,
            "holdout_positives": 0,
            "holdout_base_rate": 0.0,
        }
    holdout = make_positive_count_holdout(frame, target_col=target_col, min_positives=50)
    return {
        "holdout_rows": int(holdout.metadata["holdout_rows"]),
        "holdout_positives": int(holdout.metadata["holdout_positives"]),
        "holdout_base_rate": float(holdout.metadata["holdout_base_rate"]),
    }


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
            "date_range": [str(min(raw_dts)), str(max(raw_dts))] if raw_dts else ["", ""],
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
        holdout_stats = _safe_holdout(activity, activity_target)
        payload["stage1"][f"{horizon}m"] = {
            "rows": int(len(activity)),
            "positives": _safe_sum(activity, activity_target),
            "positive_rate": _safe_mean(activity, activity_target),
            "avg_future_count": _safe_mean(activity, "event_count_next_horizon"),
            **holdout_stats,
        }

    stage2_30m = build_zone_ranking_training_data(
        raw_events,
        horizon_minutes=30,
        target_col=target_for_horizon(30),
    )

    if stage2_30m.empty:
        payload["stage2"]["feature_quality"] = {
            "rows": 0,
            "unique_target_times": 0,
            "unique_zones": 0,
            "district_cardinality": 0,
            "region_cardinality": 0,
            "null_rate_top10": [],
        }
    else:
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
            "zero_share_top10": sorted(
                [
                    (
                        column,
                        float(
                            (
                                pd.to_numeric(stage2_30m[column], errors="coerce")
                                .fillna(0.0)
                                .eq(0)
                            ).mean()
                        ),
                    )
                    for column in NUMERIC_FEATURES
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
        sampled = (
            sample_spatial_training_rows(spatial, target_col=target)
            if not spatial.empty and target in spatial
            else spatial.iloc[0:0].copy()
        )
        holdout_stats = _safe_holdout(spatial, target)
        positives = _safe_sum(spatial, target)
        sampled_positives = _safe_sum(sampled, target)
        sampled_rows = int(len(sampled))
        sampled_neg_to_pos_ratio = float((sampled_rows - sampled_positives) / max(sampled_positives, 1))
        payload["stage2"][f"{horizon}m"] = {
            "rows": int(len(spatial)),
            "positives": positives,
            "positive_rate": _safe_mean(spatial, target),
            "sampled_rows": sampled_rows,
            "sampled_positives": sampled_positives,
            "sampled_positive_rate": _safe_mean(sampled, target),
            "sampled_neg_to_pos_ratio": sampled_neg_to_pos_ratio,
            **holdout_stats,
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


def render_report_html(payload: dict) -> str:
    raw = payload["raw"]
    reconstructed = payload["reconstructed_geo"]
    model_design = payload.get(
        "current_model_design",
        {
            "stage2_numeric_feature_count": 0,
            "stage2_categorical_feature_count": 0,
        },
    )
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
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card, .panel {{ background: #fff; border: 1px solid #dbe4ef; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
    .metric {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #e6edf5; vertical-align: top; }}
    th {{ background: #f2f6fb; }}
    h1, h2, h3 {{ margin-top: 0; }}
    .muted {{ color: #5f6f85; }}
  </style>
</head>
<body>
<main>
  <section class="panel">
    <h1>Current Model Data Analysis</h1>
    <p>Static snapshot of the current accepted model design: Stage 1 city-activity gating plus Stage 2 spatial ranking.</p>
    <p class="muted">Stage 2 feature contract: {model_design["stage2_numeric_feature_count"]} numeric features and {model_design["stage2_categorical_feature_count"]} categorical features.</p>
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
    <h2>Raw feed completeness</h2>
    <table>
      <thead><tr><th>Field</th><th>Present rows</th></tr></thead>
      <tbody>{_table_rows(raw["field_presence"].items())}</tbody>
    </table>
  </section>
  <section class="panel">
    <h2>Reconstructed geography</h2>
    <table>
      <thead><tr><th>District</th><th>Alerts</th></tr></thead>
      <tbody>{_table_rows(reconstructed["top_districts"][:8])}</tbody>
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
