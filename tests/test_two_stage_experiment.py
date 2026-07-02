import pandas as pd

from analysis.run_zone_ranking_experiment import NUMERIC_FEATURES
from ghost_zones import compute_h3_zone
import h3
from analysis.run_two_stage_experiment import (
    _candidate_models,
    _evaluate_activity_candidates,
    activity_target_for_horizon,
    combine_activity_and_spatial_scores,
    _effective_lookback_hours,
    _neighbor_hit_metrics,
    _operational_spatial_metrics,
    _prepare_ranker_training_frame,
    _select_model,
    write_two_stage_summary,
)


def test_activity_target_for_horizon_names_columns():
    assert activity_target_for_horizon(30) == "activity_next_30m"
    assert activity_target_for_horizon(60) == "activity_next_1h"
    assert activity_target_for_horizon(120) == "activity_next_2h"


def test_effective_lookback_hours_fits_short_event_history():
    events = [
        {"create_dt": "2026-06-13 09:00:00"},
        {"create_dt": "2026-06-14 15:00:00"},
    ]

    assert _effective_lookback_hours(events) == 29


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


def test_neighbor_hit_metrics_counts_adjacent_top_ranked_zone():
    zone = compute_h3_zone(22.3154, 114.1698, resolution=8)
    neighbor = next(iter(h3.grid_ring(zone, 1)))
    frame = pd.DataFrame(
        {
            "target_time": ["2026-06-01 10:00"] * 2,
            "zone_id": [zone, neighbor],
            "actual": [0, 1],
        }
    )
    metrics = _neighbor_hit_metrics(frame, "actual", pd.Series([0.9, 0.1]).to_numpy())

    assert metrics["neighbor_hit_rate_at_20"] == 1.0
    assert metrics["neighbor_hit_rate_at_50"] == 1.0


def test_select_spatial_model_prefers_top_k_before_decile_lift():
    summary = pd.DataFrame(
        [
            {
                "model": "lift_only",
                "median_precision_at_50": 0.0,
                "median_precision_at_100": 0.0,
                "median_top_decile_lift": 5.0,
                "median_average_precision": 0.01,
            },
            {
                "model": "top_k",
                "median_precision_at_50": 0.02,
                "median_precision_at_100": 0.02,
                "median_top_decile_lift": 2.0,
                "median_average_precision": 0.01,
            },
        ]
    )

    assert _select_model(summary, "spatial")["model"] == "top_k"


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


def test_ranker_group_sizes_sort_by_target_time_and_drop_empty_positive_groups():
    frame = pd.DataFrame(
        [
            {"target_time": "2026-06-01 11:00", "actual": 0, "score_feature": 0.1},
            {"target_time": "2026-06-01 10:00", "actual": 1, "score_feature": 0.9},
            {"target_time": "2026-06-01 10:00", "actual": 0, "score_feature": 0.8},
            {"target_time": "2026-06-01 12:00", "actual": 0, "score_feature": 0.7},
        ]
    )

    sorted_frame, group_sizes = _prepare_ranker_training_frame(frame, "actual")

    assert list(sorted_frame["target_time"]) == [
        pd.Timestamp("2026-06-01 10:00"),
        pd.Timestamp("2026-06-01 10:00"),
    ]
    assert group_sizes == [2]


def test_candidate_models_include_lightgbm_ranker_neighbor():
    candidates = {candidate.name: candidate for candidate in _candidate_models()}

    assert candidates["lightgbm_ranker_neighbor"].kind == "ranker"


def test_evaluate_activity_candidates_skips_ranker_candidates():
    frame = pd.DataFrame(
        {
            "target_time": pd.date_range("2026-06-01 00:00:00", periods=12, freq="h"),
            "activity_next_30m": [0, 1] * 6,
            "city_event_count_1h": range(12),
            "city_event_count_3h": range(1, 13),
            "city_event_count_24h": range(2, 14),
            "city_event_count_7d": range(3, 15),
            "active_districts_24h": [1] * 12,
            "active_regions_24h": [1] * 12,
            "city_3h_to_24h_ratio": [0.5] * 12,
            "city_24h_to_7d_ratio": [0.25] * 12,
            "hour": list(range(12)),
            "day_of_week": [0] * 12,
            "is_weekend": [0] * 12,
            "hour_sin": [0.0] * 12,
            "hour_cos": [1.0] * 12,
            "dow_sin": [0.0] * 12,
            "dow_cos": [1.0] * 12,
            "hour_bucket": ["morning"] * 12,
        }
    )

    folds, _ = _evaluate_activity_candidates(frame, "activity_next_30m", 30)

    assert "lightgbm_ranker_neighbor" not in set(folds["model"])


def test_operational_hit_metrics_include_group_precision_and_recall():
    frame = pd.DataFrame(
        {
            "target_time": ["t1", "t1", "t2", "t2"],
            "zone_id": ["a", "b", "c", "d"],
            "actual": [1, 0, 0, 1],
        }
    )
    scores = pd.Series([0.9, 0.8, 0.7, 0.6]).to_numpy()

    metrics = _operational_spatial_metrics(frame, "actual", scores)

    assert metrics["group_precision_at_50"] == 0.5
    assert metrics["group_recall_at_50"] == 1.0
