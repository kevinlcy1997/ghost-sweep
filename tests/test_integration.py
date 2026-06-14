# tests/test_integration.py
"""End-to-end test: JSON migration -> clean -> features."""

import json
import os
import tempfile
import random
import pytest
from datetime import datetime

from ghost_db import GhostDB
from ghost_clean import consolidate_events
from ghost_districts import assign_district_to_events
from ghost_features import build_features


def _make_test_alerts(n=50):
    random.seed(42)
    alerts = {}
    base_lat, base_lng = 22.315, 114.170
    for i in range(n):
        alert_id = str(1000 + i)
        hour = random.randint(8, 20)
        day = random.randint(1, 13)
        alerts[alert_id] = {
            "alert_record_id": alert_id,
            "lat": base_lat + random.uniform(-0.01, 0.01),
            "lng": base_lng + random.uniform(-0.01, 0.01),
            "address": f"Street {i}",
            "alert_type": "alert",
            "create_dt": f"2026-06-{day:02d} {hour:02d}:{random.randint(0,59):02d}:00",
            "upvote": str(random.randint(0, 5)),
            "downvote": str(random.randint(0, 1)),
        }
    return alerts


def test_full_pipeline():
    alerts = _make_test_alerts(50)
    data = {"alerts": alerts, "meta": {"total_alerts": 50}}

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as jf:
        json.dump(data, jf)
        json_path = jf.name
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as df:
        db_path = df.name

    try:
        db = GhostDB(db_path)
        count = db.migrate_from_json(json_path)
        assert count == 50

        sightings = db.get_unprocessed_sightings()
        assert len(sightings) > 0
        events = consolidate_events(sightings)
        assert len(events) > 0
        assert len(events) <= 50

        events = assign_district_to_events(events)
        assert all(ev["district"] != "" for ev in events)

        db.insert_events(events)
        stored = db.get_all_events()
        assert len(stored) == len(events)

        target_dt = datetime(2026, 6, 13, 15, 0, 0)
        features_df = build_features(stored, target_dt)
        assert len(features_df) > 0
        assert "district" in features_df.columns
        assert "hour" in features_df.columns
        db.close()
    finally:
        os.unlink(json_path)
        os.unlink(db_path)
