import h3
import pandas as pd

from ghost_ranking_features import (
    add_engineered_ranking_features,
    build_zone_ranking_training_data,
)
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


def test_build_zone_ranking_training_data_adds_police_zone_and_res8_context():
    events = [
        _event(22.3154, 114.1698, f"2026-06-{day:02d} 10:00:00")
        for day in range(1, 18)
    ]

    rows = build_zone_ranking_training_data(events, resolution=9)

    assert "police_zone" in rows.columns
    assert "res8_zone" in rows.columns
    assert set(rows["police_zone"]) == set(rows["region"])
    assert set(rows["res8_zone"].map(h3.get_resolution)) == {8}


def test_add_engineered_ranking_features_adds_police_zone_and_res8_relative_features():
    zone_a = compute_h3_zone(22.3154, 114.1698, resolution=9)
    zone_b = compute_h3_zone(22.3160, 114.1703, resolution=9)
    target_time = pd.Timestamp("2026-06-02 09:00:00")
    frame = pd.DataFrame(
        {
            "target_time": [target_time, target_time],
            "zone_id": [zone_a, zone_b],
            "zone_lat": [22.3154, 22.3160],
            "zone_lng": [114.1698, 114.1703],
            "district": ["Kowloon City", "Kowloon City"],
            "region": ["Kowloon West", "Kowloon West"],
            "police_zone": ["Kowloon West", "Kowloon West"],
            "res8_zone": [str(h3.cell_to_parent(zone_a, 8)), str(h3.cell_to_parent(zone_b, 8))],
            "hour": [9, 9],
            "day_of_week": [0, 0],
            "zone_event_count_3h": [2, 1],
            "zone_event_count_24h": [3, 1],
            "zone_event_count_7d": [5, 2],
            "district_event_count_3h": [3, 3],
            "district_event_count_24h": [4, 4],
            "district_active_zones_24h": [2, 2],
            "zone_same_hour_rate": [0.8, 0.2],
            "district_same_hour_rate": [0.5, 0.5],
        }
    )

    enhanced = add_engineered_ranking_features(frame)
    row_a = enhanced.loc[enhanced["zone_id"] == zone_a].iloc[0]

    assert row_a["police_zone_event_count_24h"] == 4
    assert row_a["res8_event_count_24h"] == 4
    assert row_a["zone_24h_share_of_police_zone"] == 0.75
    assert row_a["zone_7d_rank_in_res8"] == 1.0
    assert row_a["zone_same_hour_percentile_in_police_zone"] == 1.0
