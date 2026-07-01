import pandas as pd

from analysis.run_zone_ranking_experiment import NUMERIC_FEATURES
from analysis.run_two_stage_experiment import (
    activity_target_for_horizon,
    combine_activity_and_spatial_scores,
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
            "activity_model": {
                "chosen_model": {"model": "logistic"},
                "holdout_metrics": {"average_precision": 0.5},
            },
            "spatial_model": {
                "chosen_model": {"model": "lightgbm"},
                "holdout_metrics": {"precision_at_20": 0.1},
            },
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


def test_spatial_feature_list_includes_context_feature_pack():
    expected = {
        "ring2_event_count_24h",
        "distance_to_nearest_event_24h_m",
        "zone_24h_share_of_district",
        "nearest_road_m",
    }

    assert expected <= set(NUMERIC_FEATURES)
