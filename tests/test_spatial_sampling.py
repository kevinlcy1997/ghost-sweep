import pandas as pd

from ghost_ranking_features import sample_spatial_training_rows


def test_sample_spatial_training_rows_keeps_all_positive_zones():
    df = pd.DataFrame(
        {
            "target_time": ["2026-06-01 10:00"] * 5,
            "zone_id": ["a", "b", "c", "d", "e"],
            "zone_event_count_24h": [2, 1, 9, 0, 5],
            "district_event_count_24h": [3, 3, 7, 0, 7],
            "alert_next_1h": [1, 1, 0, 0, 0],
        }
    )

    sampled = sample_spatial_training_rows(df, "alert_next_1h", negative_ratio=1, random_state=7)

    assert set(sampled.loc[sampled["alert_next_1h"] == 1, "zone_id"]) == {"a", "b"}
    assert len(sampled) == 4
    assert sampled["target_time"].nunique() == 1


def test_sample_spatial_training_rows_limits_inactive_negatives():
    df = pd.DataFrame(
        {
            "target_time": ["2026-06-01 10:00"] * 3 + ["2026-06-01 11:00"] * 4,
            "zone_id": list("abcdefg"),
            "zone_event_count_24h": [0, 0, 0, 3, 2, 1, 0],
            "district_event_count_24h": [0, 0, 0, 4, 4, 1, 0],
            "alert_next_1h": [0, 0, 0, 1, 0, 0, 0],
        }
    )

    sampled = sample_spatial_training_rows(
        df,
        "alert_next_1h",
        negative_ratio=2,
        inactive_negative_fraction=0.5,
        random_state=11,
    )

    inactive = sampled[sampled["target_time"] == "2026-06-01 10:00"]
    assert len(inactive) <= 1
    assert sampled["alert_next_1h"].sum() == 1
