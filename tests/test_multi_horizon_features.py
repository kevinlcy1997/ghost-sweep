from ghost_ranking_features import build_zone_ranking_training_data
from tests.test_ghost_ranking_features import _event


def test_horizon_minutes_controls_future_target_window():
    events = [
        _event(22.3154, 114.1698, f"2026-06-{day:02d} 10:00:00")
        for day in range(1, 17)
    ]
    events.append(_event(22.3154, 114.1698, "2026-06-17 10:45:00"))

    rows_30m = build_zone_ranking_training_data(
        events,
        horizon_minutes=30,
        target_col="alert_next_30m",
    )
    rows_2h = build_zone_ranking_training_data(
        events,
        horizon_minutes=120,
        target_col="alert_next_2h",
    )

    target_time = "2026-06-17 10:00:00"
    row_30m = rows_30m.loc[rows_30m["target_time"].astype(str) == target_time].iloc[0]
    row_2h = rows_2h.loc[rows_2h["target_time"].astype(str) == target_time].iloc[0]

    assert row_30m["alert_next_30m"] == 0
    assert row_2h["alert_next_2h"] == 1


def test_default_target_remains_alert_next_2h():
    events = [
        _event(22.3154, 114.1698, f"2026-06-{day:02d} 10:00:00")
        for day in range(1, 18)
    ]

    rows = build_zone_ranking_training_data(events)

    assert "alert_next_2h" in rows.columns
