import pandas as pd

from ghost_ranking_features import build_zone_ranking_training_data
from ghost_zones import compute_h3_zone


def _event(lat, lng, create_dt):
    return {
        "lat": lat,
        "lng": lng,
        "create_dt": create_dt,
        "duration_min": 5,
        "report_count": 1,
        "total_upvotes": 0,
        "total_downvotes": 0,
    }


def test_zone_ranking_target_uses_future_window_without_feature_leakage():
    lat, lng = 22.3154, 114.1698
    zone = compute_h3_zone(lat, lng)
    events = [
        _event(lat, lng, "2026-06-01 09:00:00"),
        _event(lat, lng, "2026-06-02 10:00:00"),
        _event(lat, lng, "2026-06-02 11:00:00"),
    ]

    df = build_zone_ranking_training_data(events, lookback_days=1, forecast_hours=2)
    row = df[
        (df["target_time"] == pd.Timestamp("2026-06-02 09:00:00")) & (df["zone_id"] == zone)
    ].iloc[0]

    assert row["zone_event_count_24h"] == 1
    assert row["event_count_next_2h"] == 1
    assert row["alert_next_2h"] == 1
def test_build_zone_ranking_training_data_accepts_resolution():
    events = [
        _event(22.3154, 114.1698, f"2026-06-{day:02d} 10:00:00")
        for day in range(1, 18)
    ]

    rows = build_zone_ranking_training_data(events, resolution=9)

    assert set(rows["h3_resolution"]) == {9}
