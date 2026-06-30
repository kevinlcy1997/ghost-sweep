"""Inventory analysis, modeling, and dashboard artifacts for traceability."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghost_zones import DEFAULT_H3_RESOLUTION

MANIFEST_PATH = ROOT / "analysis" / "dashboard_manifest_latest.json"
RESOLUTION = DEFAULT_H3_RESOLUTION
COVERAGE_MODE = "road_access"

ARTIFACT_GROUPS = {
    "coverage_grid": [
        f"analysis/geo/hk_h3_coverage_res{RESOLUTION}.csv",
        f"analysis/geo/hk_h3_coverage_res{RESOLUTION}.geojson",
    ],
    "road_coverage_grid": [
        f"analysis/geo/hk_drivable_roads.geojson",
        f"analysis/geo/hk_h3_road_coverage_res{RESOLUTION}.csv",
        f"analysis/geo/hk_h3_road_coverage_res{RESOLUTION}.geojson",
    ],
    "feature_mart": [
        f"analysis/data_discovery/zone_sparsity_profile_road_res{RESOLUTION}.csv",
        f"analysis/data_discovery/zone_hour_profile_road_res{RESOLUTION}.csv",
        f"analysis/data_discovery/zone_daily_profile_road_res{RESOLUTION}.csv",
        f"analysis/data_discovery/zone_recency_profile_road_res{RESOLUTION}.csv",
        f"analysis/data_discovery/zone_neighbor_context_road_res{RESOLUTION}.csv",
    ],
    "data_discovery": [
        "analysis/data_discovery/hex_grid_discovery_summary_res8.json",
        "analysis/data_discovery/hex_zone_event_distribution_res8.csv",
        "analysis/data_discovery/hourly_distribution.csv",
        "analysis/data_discovery/day_of_week_distribution.csv",
    ],
    "multi_horizon_models": [
        "analysis/multi_horizon_summary_latest.csv",
        "analysis/multi_horizon_report.html",
        "analysis/model_iteration_summary_latest.csv",
        "analysis/model_iteration_summary_30m_latest.csv",
        "analysis/model_iteration_summary_1h_latest.csv",
        "analysis/model_iteration_summary_2h_latest.csv",
        "analysis/model_iteration_report.html",
        "analysis/model_iteration_report_30m.html",
        "analysis/model_iteration_report_1h.html",
        "analysis/model_iteration_report_2h.html",
    ],
    "two_stage_models": [
        "analysis/two_stage_summary_latest.csv",
        "analysis/two_stage_report.html",
        "analysis/activity_model_folds_30m_latest.csv",
        "analysis/activity_model_folds_1h_latest.csv",
        "analysis/activity_model_folds_2h_latest.csv",
        "analysis/spatial_model_folds_30m_latest.csv",
        "analysis/spatial_model_folds_1h_latest.csv",
        "analysis/spatial_model_folds_2h_latest.csv",
        "analysis/best_activity_model_30m.joblib",
        "analysis/best_activity_model_1h.joblib",
        "analysis/best_activity_model_2h.joblib",
        "analysis/best_spatial_zone_model_30m.joblib",
        "analysis/best_spatial_zone_model_1h.joblib",
        "analysis/best_spatial_zone_model_2h.joblib",
    ],
    "two_stage_metadata": [
        "analysis/activity_model_metadata_30m.json",
        "analysis/activity_model_metadata_1h.json",
        "analysis/activity_model_metadata_2h.json",
        "analysis/spatial_model_metadata_30m.json",
        "analysis/spatial_model_metadata_1h.json",
        "analysis/spatial_model_metadata_2h.json",
    ],
    "two_stage_predictions": [
        "analysis/activity_predictions_30m_latest.csv",
        "analysis/activity_predictions_1h_latest.csv",
        "analysis/activity_predictions_2h_latest.csv",
        "analysis/spatial_zone_predictions_30m_latest.csv",
        "analysis/spatial_zone_predictions_1h_latest.csv",
        "analysis/spatial_zone_predictions_2h_latest.csv",
    ],
    "model_metadata": [
        "analysis/best_iterated_model_metadata.json",
        "analysis/best_iterated_model_metadata_30m.json",
        "analysis/best_iterated_model_metadata_1h.json",
        "analysis/best_iterated_model_metadata_2h.json",
    ],
    "model_joblibs": [
        "analysis/best_experiment_model.joblib",
        "analysis/best_iterated_zone_model.joblib",
        "analysis/best_iterated_zone_model_30m.joblib",
        "analysis/best_iterated_zone_model_1h.joblib",
        "analysis/best_iterated_zone_model_2h.joblib",
    ],
    "predictions": [
        "analysis/iterated_zone_predictions_latest.csv",
        "analysis/iterated_zone_predictions_30m_latest.csv",
        "analysis/iterated_zone_predictions_1h_latest.csv",
        "analysis/iterated_zone_predictions_2h_latest.csv",
    ],
    "simulation": [
        "analysis/real_location_simulation_report.html",
    ],
    "mlflow_tracking": [
        "analysis/mlruns",
    ],
    "reports": [
        "analysis/zone_model_visual_explainer.html",
        "analysis/zone_forecast_map.html",
        "analysis/spotfire_dashboard.html",
    ],
}


def count_rows(path: Path) -> int | None:
    if not path.exists() or path.is_dir() or path.suffix.lower() != ".csv":
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def artifact_entry(relative_path: str) -> dict[str, object]:
    path = ROOT / relative_path
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": relative_path.replace("\\", "/"),
        "exists": exists,
        "type": "directory" if exists and path.is_dir() else path.suffix.lower().lstrip("."),
        "size_bytes": stat.st_size if stat and path.is_file() else None,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
        if stat
        else None,
        "row_count": count_rows(path) if exists else None,
    }


def build_manifest() -> dict[str, object]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "ghost-sweep",
        "coverage_mode": COVERAGE_MODE,
        "h3_resolution": RESOLUTION,
        "artifact_groups": {
            group: [artifact_entry(path) for path in paths]
            for group, paths in ARTIFACT_GROUPS.items()
        },
    }


def write_manifest(path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    output = write_manifest(args.output)
    print(f"Wrote dashboard manifest to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
