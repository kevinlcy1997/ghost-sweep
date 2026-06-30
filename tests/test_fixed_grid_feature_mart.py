from datetime import datetime

from analysis.build_fixed_grid_feature_mart import (
    build_zone_daily_profile,
    build_zone_hour_profile,
    build_zone_neighbor_context,
    build_zone_recency_profile,
    build_zone_sparsity_profile,
    load_observed_events,
)


def test_sparsity_profile_keeps_zero_history_cells():
    grid = [
        {"h3_zone": "88411cb369fffff", "district": "Mong Kok", "region": "Kowloon West"},
        {"h3_zone": "88411c95a9fffff", "district": "Sha Tin", "region": "New Territories"},
    ]
    events = [
        {
            "h3_zone": "88411cb369fffff",
            "district": "Mong Kok",
            "region": "Kowloon West",
            "event_time": datetime(2026, 6, 1, 10),
            "hour": 10,
            "day_of_week": 0,
        }
    ]

    rows = build_zone_sparsity_profile(grid, events)

    active = next(row for row in rows if row["h3_zone"] == "88411cb369fffff")
    empty = next(row for row in rows if row["h3_zone"] == "88411c95a9fffff")
    assert active["event_count"] == 1
    assert active["has_observed_history"] == 1
    assert empty["event_count"] == 0
    assert empty["is_zero_history"] == 1


def test_time_profiles_create_complete_zone_combinations():
    grid = [{"h3_zone": "88411cb369fffff"}]
    events = [
        {
            "h3_zone": "88411cb369fffff",
            "event_time": datetime(2026, 6, 3, 20),
            "hour": 20,
            "day_of_week": 2,
        }
    ]

    hour_rows = build_zone_hour_profile(grid, events)
    day_rows = build_zone_daily_profile(grid, events)

    assert len(hour_rows) == 24
    assert len(day_rows) == 7
    assert next(row for row in hour_rows if row["hour"] == 20)["event_count"] == 1
    assert next(row for row in day_rows if row["day_of_week"] == 2)["event_count"] == 1


def test_recency_and_neighbor_context_are_zone_level():
    grid = [
        {"h3_zone": "88411cb369fffff"},
        {"h3_zone": "88411cb36dfffff"},
    ]
    events = [
        {
            "h3_zone": "88411cb369fffff",
            "event_time": datetime(2026, 6, 3, 20),
            "hour": 20,
            "day_of_week": 2,
        }
    ]

    recency = build_zone_recency_profile(grid, events)
    neighbors = build_zone_neighbor_context(grid, events)

    assert len(recency) == 2
    assert len(neighbors) == 2
    assert any(row["last_event_time"] for row in recency)
    assert all("neighbor_event_count" in row for row in neighbors)


def test_load_observed_events_reads_project_database_schema():
    events = load_observed_events()

    assert len(events) > 0
    assert len(events) >= 5500
    assert {"h3_zone", "event_time", "hour", "day_of_week"} <= set(events[0])
