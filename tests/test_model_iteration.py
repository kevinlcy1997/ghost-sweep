import pandas as pd

from analysis.run_model_iteration import (
    format_prediction_artifact,
    make_walk_forward_splits,
    select_satisfactory_model,
)


def test_make_walk_forward_splits_keeps_train_before_test():
    df = pd.DataFrame(
        {
            "target_time": pd.date_range("2026-06-01", periods=10, freq="h"),
            "value": range(10),
        }
    )

    splits = make_walk_forward_splits(df, n_splits=3)

    assert len(splits) == 3
    for train_mask, test_mask in splits:
        assert df.loc[train_mask, "target_time"].max() < df.loc[test_mask, "target_time"].min()
        assert train_mask.sum() > 0
        assert test_mask.sum() > 0


def test_select_satisfactory_model_prefers_stable_precision_and_ap():
    rows = [
        {
            "model": "spiky",
            "median_precision_at_20": 0.95,
            "median_precision_at_50": 0.15,
            "median_precision_at_100": 0.10,
            "median_average_precision": 0.20,
            "median_top_decile_lift": 3.0,
            "precision_at_20_std": 0.40,
            "median_brier_score": 0.40,
        },
        {
            "model": "stable",
            "median_precision_at_20": 0.90,
            "median_precision_at_50": 0.20,
            "median_precision_at_100": 0.18,
            "median_average_precision": 0.45,
            "median_top_decile_lift": 5.0,
            "precision_at_20_std": 0.05,
            "median_brier_score": 0.18,
        },
    ]

    chosen = select_satisfactory_model(rows)

    assert chosen["model"] == "stable"


def test_select_satisfactory_model_prefers_practical_sparse_ranking_over_p20_spike():
    rows = [
        {
            "model": "p20_spike",
            "median_precision_at_20": 1.0,
            "median_precision_at_50": 0.04,
            "median_precision_at_100": 0.02,
            "median_average_precision": 0.06,
            "median_top_decile_lift": 1.2,
            "precision_at_20_std": 0.48,
            "median_brier_score": 0.30,
        },
        {
            "model": "useful_ranker",
            "median_precision_at_20": 0.0,
            "median_precision_at_50": 0.08,
            "median_precision_at_100": 0.06,
            "median_average_precision": 0.12,
            "median_top_decile_lift": 4.5,
            "precision_at_20_std": 0.00,
            "median_brier_score": 0.10,
        },
    ]

    chosen = select_satisfactory_model(rows)

    assert chosen["model"] == "useful_ranker"


def test_format_prediction_artifact_adds_probability_rank_and_risk_band():
    base = pd.DataFrame(
        {
            "target_time": ["2026-06-29 10:00:00", "2026-06-29 10:00:00"],
            "zone_id": ["b", "a"],
            "district": ["Yau Tsim Mong", "Central"],
            "region": ["Kowloon West", "Hong Kong Island"],
            "zone_lat": [22.3, 22.2],
            "zone_lng": [114.1, 114.2],
        }
    )

    predictions = format_prediction_artifact(base, [0.01, 0.2], [0, 1])

    assert list(predictions["zone_id"]) == ["a", "b"]
    assert list(predictions["rank"]) == [1, 2]
    assert list(predictions["probability"]) == [0.2, 0.01]
    assert list(predictions["score"]) == [0.2, 0.01]
    assert list(predictions["risk_band"]) == ["critical", "low"]
    assert list(predictions["actual"]) == [1, 0]
