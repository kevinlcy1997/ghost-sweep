import pandas as pd

from ghost_activity_features import build_activity_training_data


def _event(create_dt: str, lat: float = 22.302, lng: float = 114.172) -> dict:
    return {"create_dt": create_dt, "lat": lat, "lng": lng, "name": "ticket"}


def test_build_activity_training_data_keeps_no_event_hours_and_future_labels():
    rows = build_activity_training_data(
        [
            _event("2026-06-01 00:10:00"),
            _event("2026-06-01 03:30:00"),
        ],
        horizon_minutes=60,
        lookback_hours=1,
    )

    assert list(rows["target_time"]) == [
        pd.Timestamp("2026-06-01 01:00:00"),
        pd.Timestamp("2026-06-01 02:00:00"),
        pd.Timestamp("2026-06-01 03:00:00"),
    ]
    assert (
        rows.loc[
            rows["target_time"] == pd.Timestamp("2026-06-01 02:00:00"),
            "activity_next_1h",
        ].item()
        == 0
    )
    assert (
        rows.loc[
            rows["target_time"] == pd.Timestamp("2026-06-01 03:00:00"),
            "activity_next_1h",
        ].item()
        == 1
    )
    assert (
        rows.loc[
            rows["target_time"] == pd.Timestamp("2026-06-01 01:00:00"),
            "city_event_count_1h",
        ].item()
        == 1
    )


def test_build_activity_training_data_uses_requested_horizon_target_name():
    rows = build_activity_training_data(
        [_event("2026-06-01 00:10:00"), _event("2026-06-01 01:20:00")],
        horizon_minutes=30,
        lookback_hours=1,
    )

    assert "activity_next_30m" in rows.columns
    assert rows["activity_next_30m"].isin([0, 1]).all()
