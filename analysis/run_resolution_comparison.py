from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from run_zone_ranking_experiment import ROOT, run_experiment
except ModuleNotFoundError:
    from analysis.run_zone_ranking_experiment import ROOT, run_experiment


OUTPUT_DIR = ROOT / "analysis"
CSV_PATH = OUTPUT_DIR / "resolution_comparison_latest.csv"
HTML_PATH = OUTPUT_DIR / "resolution_comparison_report.html"
CHOICE_PATH = OUTPUT_DIR / "resolution_choice_latest.json"
RESOLUTIONS = [8, 9, 10]


def _metric(result: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = result.get("metrics", {}).get(name, default)
    if value is None:
        return default
    return float(value)


def choose_practical_resolution(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def _row_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "resolution": int(result["resolution"]),
        "best_model": result["best_model"],
        "training_rows": int(result["training_rows"]),
        "active_zones": int(result["active_zones"]),
        "median_events_per_zone": float(result["median_events_per_zone"]),
        "one_off_zone_rate": float(result["one_off_zone_rate"]),
        "positive_rate": float(result["positive_rate"]),
        "precision_at_10": _metric(result, "precision_at_10"),
        "precision_at_20": _metric(result, "precision_at_20"),
        "precision_at_50": _metric(result, "precision_at_50"),
        "recall_at_20": _metric(result, "recall_at_20"),
        "top_decile_lift": _metric(result, "top_decile_lift"),
        "average_precision": _metric(result, "average_precision"),
        "roc_auc": _metric(result, "roc_auc"),
        "geojson_path": result["geojson_path"],
        "report_path": result["report_path"],
    }


def _write_report(rows: list[dict[str, Any]], chosen: dict[str, Any]) -> None:
    df = pd.DataFrame(rows)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HK H3 Resolution Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    h1 {{ margin-bottom: 8px; }}
    .summary {{ margin: 18px 0; padding: 14px; border: 1px solid #d9e2ef; border-radius: 8px; background: #f7f9fc; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ef; padding: 8px; text-align: right; }}
    th {{ background: #eef3f9; }}
    td:nth-child(1), td:nth-child(2), td:nth-last-child(1), td:nth-last-child(2) {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>HK H3 Resolution Comparison</h1>
  <div class="summary">
    <strong>Chosen practical resolution:</strong> H3 res {int(chosen["resolution"])}.
    Precision@20 is {float(chosen["precision_at_20"]):.3f},
    top-decile lift is {float(chosen["top_decile_lift"]):.3f}, and
    one-off-zone rate is {float(chosen["one_off_zone_rate"]):.3f}.
  </div>
  {df.to_html(index=False)}
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")


def run_resolution_comparison(
    resolutions: list[int] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for resolution in resolutions or RESOLUTIONS:
        result = run_experiment(resolution=resolution, write_latest_alias=False)
        rows.append(_row_from_result(result))

    chosen = choose_practical_resolution(rows)
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    CHOICE_PATH.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    _write_report(rows, chosen)
    return {"rows": rows, "chosen": chosen}


def main() -> None:
    result = run_resolution_comparison()
    chosen = result["chosen"]
    print(
        "Chosen H3 res "
        f"{chosen['resolution']} with precision_at_20={chosen['precision_at_20']:.3f}, "
        f"top_decile_lift={chosen['top_decile_lift']:.3f}, "
        f"one_off_zone_rate={chosen['one_off_zone_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
