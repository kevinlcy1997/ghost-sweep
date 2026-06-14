# tests/test_ghost_clean.py
import pytest
from ghost_clean import consolidate_events


def test_single_alert_becomes_one_event():
    sightings = [{
        "alert_record_id": "1", "lat": 22.3154, "lng": 114.1698,
        "address": "37 Dundas St", "create_dt": "2026-06-13 11:37:50",
        "upvote": 3, "downvote": 0,
    }]
    events = consolidate_events(sightings)
    assert len(events) == 1
    assert events[0]["report_count"] == 1
    assert events[0]["lat"] == 22.3154


def test_nearby_alerts_merged():
    sightings = [
        {"alert_record_id": "1", "lat": 22.31540, "lng": 114.16980,
         "address": "37 Dundas St", "create_dt": "2026-06-13 11:37:50",
         "upvote": 3, "downvote": 0},
        {"alert_record_id": "2", "lat": 22.31542, "lng": 114.16982,
         "address": "37 Dundas St", "create_dt": "2026-06-13 11:40:00",
         "upvote": 1, "downvote": 0},
    ]
    events = consolidate_events(sightings)
    assert len(events) == 1
    assert events[0]["report_count"] == 2
    assert events[0]["total_upvotes"] == 4


def test_far_alerts_separate():
    sightings = [
        {"alert_record_id": "1", "lat": 22.3154, "lng": 114.1698,
         "address": "37 Dundas St", "create_dt": "2026-06-13 11:37:50",
         "upvote": 1, "downvote": 0},
        {"alert_record_id": "2", "lat": 22.3200, "lng": 114.1700,
         "address": "Argyle St", "create_dt": "2026-06-13 11:38:00",
         "upvote": 1, "downvote": 0},
    ]
    events = consolidate_events(sightings)
    assert len(events) == 2


def test_time_gap_separates():
    sightings = [
        {"alert_record_id": "1", "lat": 22.3154, "lng": 114.1698,
         "address": "37 Dundas St", "create_dt": "2026-06-13 11:00:00",
         "upvote": 1, "downvote": 0},
        {"alert_record_id": "2", "lat": 22.3154, "lng": 114.1698,
         "address": "37 Dundas St", "create_dt": "2026-06-13 11:20:00",
         "upvote": 1, "downvote": 0},
    ]
    events = consolidate_events(sightings)
    assert len(events) == 2


def test_centroid_computed():
    sightings = [
        {"alert_record_id": "1", "lat": 22.3100, "lng": 114.1700,
         "address": "A St", "create_dt": "2026-06-13 11:00:00",
         "upvote": 1, "downvote": 0},
        {"alert_record_id": "2", "lat": 22.3100, "lng": 114.17001,
         "address": "A St", "create_dt": "2026-06-13 11:01:00",
         "upvote": 2, "downvote": 0},
    ]
    events = consolidate_events(sightings)
    assert len(events) == 1
    assert abs(events[0]["lat"] - 22.3100) < 0.0001
    assert events[0]["address"] == "A St"
