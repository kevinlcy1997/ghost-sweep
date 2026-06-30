import pandas as pd

from analysis.two_stage_splits import make_positive_count_holdout, make_purged_rolling_splits


def test_make_purged_rolling_splits_keeps_horizon_gap():
    df = pd.DataFrame({"target_time": pd.date_range("2026-06-01", periods=16, freq="h")})

    splits = make_purged_rolling_splits(df, horizon_minutes=120, n_splits=3)

    assert len(splits) == 3
    for split in splits:
        train_end = df.loc[split.train_mask, "target_time"].max()
        validation_start = df.loc[split.validation_mask, "target_time"].min()
        assert validation_start - train_end >= pd.Timedelta(minutes=120)
        assert split.metadata["purge_minutes"] == 120
        assert split.metadata["train_rows"] > 0
        assert split.metadata["validation_rows"] > 0


def test_make_positive_count_holdout_expands_back_to_minimum_positives():
    df = pd.DataFrame(
        {
            "target_time": pd.date_range("2026-06-01", periods=8, freq="h"),
            "target": [0, 1, 0, 1, 0, 0, 0, 1],
        }
    )

    split = make_positive_count_holdout(df, "target", min_positives=2)

    holdout = df.loc[split.holdout_mask]
    assert holdout["target"].sum() == 2
    assert holdout["target_time"].min() == pd.Timestamp("2026-06-01 03:00:00")
    assert split.metadata["met_min_positives"] is True
    assert split.metadata["holdout_positives"] == 2
    assert split.metadata["train_rows"] == 3


def test_make_positive_count_holdout_reports_fallback_when_threshold_not_met():
    df = pd.DataFrame(
        {
            "target_time": pd.date_range("2026-06-01", periods=4, freq="h"),
            "target": [0, 0, 1, 0],
        }
    )

    split = make_positive_count_holdout(df, "target", min_positives=5)

    assert split.metadata["met_min_positives"] is False
    assert split.metadata["holdout_positives"] == 1
    assert split.holdout_mask.all()
    assert split.train_mask.sum() == 0
