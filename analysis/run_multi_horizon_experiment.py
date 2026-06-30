from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.run_model_iteration import run_model_iteration


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
            "median_precision_at_20": metadata["chosen_model"]["median_precision_at_20"],
            "median_precision_at_50": metadata["chosen_model"]["median_precision_at_50"],
            "median_precision_at_100": metadata["chosen_model"]["median_precision_at_100"],
            "median_average_precision": metadata["chosen_model"]["median_average_precision"],
            "median_top_decile_lift": metadata["chosen_model"]["median_top_decile_lift"],
            "median_district_hit_rate_at_50": metadata["chosen_model"][
                "median_district_hit_rate_at_50"
            ],
            "median_region_hit_rate_at_50": metadata["chosen_model"][
                "median_region_hit_rate_at_50"
            ],
            "median_brier_score": metadata["chosen_model"]["median_brier_score"],
            "median_expected_calibration_error": metadata["chosen_model"][
                "median_expected_calibration_error"
            ],
            **{f"holdout_{key}": value for key, value in metadata["holdout_metrics"].items()},
            "metadata_path": metadata["metadata_path"],
            "predictions_path": metadata["predictions_path"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_PATH, index=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Ghost Sweep Multi-Horizon Risk</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ef; padding: 8px; text-align: right; }}
    th {{ background: #eef3f9; }}
    td:nth-child(2), td:nth-child(3), td:nth-last-child(1), td:nth-last-child(2) {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Ghost Sweep Multi-Horizon Risk</h1>
  <p>Separate selected models for next 30 minutes, 1 hour, and 2 hours.</p>
  {df.to_html(index=False)}
</body>
</html>
"""
    REPORT_PATH.write_text(html, encoding="utf-8")
    return rows


def main() -> None:
    print(json.dumps(run_multi_horizon_experiment(), indent=2))
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
