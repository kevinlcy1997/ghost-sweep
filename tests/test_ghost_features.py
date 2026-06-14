# tests/test_ghost_features.py
import pytest
from datetime import datetime
from ghost_features import build_features


def test_build_features_basic():
    events = [
        {"event_id": 1, "lat": 22.315, "lng": 114.170, "create_dt": "2026-06-13 11:00:00",
         "end_dt": "2026-06-13 11:05:00", "duration_min": 5, "report_count": 3,
         "total_upvotes": 5, "total_downvotes": 0,
         "grid_cell": "22.315_114.170", "district": "Mong Kok", "region": "Kowloon West"},
        {"event_id": 2, "lat": 22.315, "lng": 114.170, "create_dt": "2026-06-13 14:00:00",
         "end_dt": "2026-06-13 14:02:00", "duration_min": 2, "report_count": 1,
         "total_upvotes": 1, "total_downvotes": 0,
         "grid_cell": "22.315_114.170", "district": "Mong Kok", "region": "Kowloon West"},
    ]
    target_dt = datetime(2026, 6, 13, 15, 0, 0)
    df = build_features(events, target_dt)
    assert len(df) > 0
    expected_cols = {"hour", "day_of_week", "is_weekend", "cell_historical_freq",
                    "cell_24h_count", "district", "region"}
    assert expected_cols.issubset(set(df.columns))


def test_features_hour_extraction():
    events = [
        {"event_id": 1, "lat": 22.315, "lng": 114.170, "create_dt": "2026-06-13 11:00:00",
         "end_dt": "2026-06-13 11:00:00", "duration_min": 0, "report_count": 1,
         "total_upvotes": 1, "total_downvotes": 0,
         "grid_cell": "22.315_114.170", "district": "Mong Kok", "region": "Kowloon West"},
    ]
    target_dt = datetime(2026, 6, 13, 15, 0, 0)
    df = build_features(events, target_dt)
    assert (df["hour"] == 15).all()
