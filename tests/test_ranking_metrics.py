import pandas as pd

from ghost_ranking_metrics import (
    group_precision_at_k,
    group_recall_at_k,
    near_miss_hit_rate_at_k,
)


def test_near_miss_hit_rate_at_k_counts_neighbor_matches_by_group():
    predictions = pd.DataFrame(
        [
            {"target_time": "t1", "zone_id": "a", "score": 0.9, "actual": 0},
            {"target_time": "t1", "zone_id": "b", "score": 0.1, "actual": 1},
            {"target_time": "t2", "zone_id": "c", "score": 0.8, "actual": 0},
            {"target_time": "t2", "zone_id": "d", "score": 0.7, "actual": 0},
        ]
    )

    result = near_miss_hit_rate_at_k(
        predictions,
        k=1,
        neighbor_lookup={"a": {"b"}, "c": set()},
        score_col="score",
        label_col="actual",
    )

    assert result == 0.5


def test_group_precision_at_k_averages_target_time_precision():
    predictions = pd.DataFrame(
        [
            {"target_time": "t1", "zone_id": "a", "score": 0.9, "actual": 1},
            {"target_time": "t1", "zone_id": "b", "score": 0.8, "actual": 0},
            {"target_time": "t2", "zone_id": "c", "score": 0.7, "actual": 0},
            {"target_time": "t2", "zone_id": "d", "score": 0.6, "actual": 1},
        ]
    )

    result = group_precision_at_k(
        predictions,
        k=1,
        score_col="score",
        label_col="actual",
        group_col="target_time",
    )

    assert result == 0.5


def test_group_recall_at_k_counts_positive_groups_without_hits_as_zero():
    predictions = pd.DataFrame(
        [
            {"target_time": "t1", "zone_id": "a", "score": 0.9, "actual": 1},
            {"target_time": "t1", "zone_id": "b", "score": 0.8, "actual": 0},
            {"target_time": "t2", "zone_id": "c", "score": 0.7, "actual": 0},
            {"target_time": "t2", "zone_id": "d", "score": 0.6, "actual": 1},
            {"target_time": "t3", "zone_id": "e", "score": 0.5, "actual": 0},
        ]
    )

    result = group_recall_at_k(
        predictions,
        k=1,
        score_col="score",
        label_col="actual",
        group_col="target_time",
    )

    assert result == 0.5
