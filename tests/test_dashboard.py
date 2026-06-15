# tests/test_dashboard.py
"""Tests for dashboard generator compute_stats and generate_html."""

import pytest
from generate_dashboard import compute_stats, deduplicate_alerts


def _make_alerts():
    """Create test alerts across multiple districts."""
    from datetime import datetime
    base = [
        {"id": "1", "lat": 22.30, "lng": 114.17, "address": "Addr A1", "district": "Yau Tsim", "region": "Kowloon West",
         "create_dt": "2026-06-13 10:00:00", "dt": datetime(2026, 6, 13, 10, 0), "hour": 10, "dow": "Saturday", "upvote": 1, "downvote": 0, "count": 1},
        {"id": "2", "lat": 22.30, "lng": 114.17, "address": "Addr A1", "district": "Yau Tsim", "region": "Kowloon West",
         "create_dt": "2026-06-13 11:00:00", "dt": datetime(2026, 6, 13, 11, 0), "hour": 11, "dow": "Saturday", "upvote": 0, "downvote": 0, "count": 1},
        {"id": "3", "lat": 22.30, "lng": 114.17, "address": "Addr A2", "district": "Yau Tsim", "region": "Kowloon West",
         "create_dt": "2026-06-13 12:00:00", "dt": datetime(2026, 6, 13, 12, 0), "hour": 12, "dow": "Saturday", "upvote": 2, "downvote": 0, "count": 1},
        {"id": "4", "lat": 22.45, "lng": 114.03, "address": "Addr B1", "district": "Yuen Long", "region": "New Territories North",
         "create_dt": "2026-06-13 09:00:00", "dt": datetime(2026, 6, 13, 9, 0), "hour": 9, "dow": "Saturday", "upvote": 0, "downvote": 0, "count": 1},
        {"id": "5", "lat": 22.45, "lng": 114.03, "address": "Addr B1", "district": "Yuen Long", "region": "New Territories North",
         "create_dt": "2026-06-13 10:30:00", "dt": datetime(2026, 6, 13, 10, 30), "hour": 10, "dow": "Saturday", "upvote": 0, "downvote": 0, "count": 1},
        {"id": "6", "lat": 22.45, "lng": 114.03, "address": "Addr B2", "district": "Yuen Long", "region": "New Territories North",
         "create_dt": "2026-06-14 08:00:00", "dt": datetime(2026, 6, 14, 8, 0), "hour": 8, "dow": "Sunday", "upvote": 3, "downvote": 0, "count": 1},
    ]
    return base


def test_top_addresses_grouped_by_district():
    """Top hotspot addresses should be grouped by district."""
    alerts = _make_alerts()
    stats = compute_stats(alerts)
    assert "top_by_district" in stats
    assert "Yau Tsim" in stats["top_by_district"]
    assert "Yuen Long" in stats["top_by_district"]


def test_top_addresses_limited_to_5_per_district():
    """Each district should have at most 5 top addresses."""
    alerts = _make_alerts()
    stats = compute_stats(alerts)
    for district, addrs in stats["top_by_district"].items():
        assert len(addrs) <= 5


def test_top_addresses_include_recent_records():
    """Each top address should include up to 5 recent activity records."""
    alerts = _make_alerts()
    stats = compute_stats(alerts)
    for district, addrs in stats["top_by_district"].items():
        for entry in addrs:
            assert "address" in entry
            assert "count" in entry
            assert "recent" in entry
            assert len(entry["recent"]) <= 5


def test_recent_alerts_grouped_by_district():
    """Latest alerts should be grouped by district."""
    alerts = _make_alerts()
    stats = compute_stats(alerts)
    assert "recent_by_district" in stats
    assert isinstance(stats["recent_by_district"], dict)
    assert "Yau Tsim" in stats["recent_by_district"]
    assert "Yuen Long" in stats["recent_by_district"]


def test_recent_by_district_sorted_by_time():
    """Alerts within each district should be sorted most recent first."""
    alerts = _make_alerts()
    stats = compute_stats(alerts)
    for district, records in stats["recent_by_district"].items():
        for i in range(len(records) - 1):
            assert records[i]["create_dt"] >= records[i + 1]["create_dt"]
