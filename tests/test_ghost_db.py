# tests/test_ghost_db.py
import os
import tempfile
import pytest
from ghost_db import GhostDB


def test_create_tables():
    """DB should create sightings, events, and poll_cycles tables."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = GhostDB(db_path)
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {row[0] for row in tables}
        assert "sightings" in table_names
        assert "events" in table_names
        assert "poll_cycles" in table_names
        db.close()
    finally:
        os.unlink(db_path)


def test_insert_sighting():
    """Should insert a sighting record and retrieve it."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = GhostDB(db_path)
        db.insert_sightings([{
            "alert_record_id": "12345",
            "lat": 22.3154,
            "lng": 114.1698,
            "address": "37 Dundas St",
            "alert_type": "alert",
            "create_dt": "2026-06-13 11:37:50",
            "upvote": 3,
            "downvote": 0,
        }])
        rows = db.execute("SELECT * FROM sightings WHERE alert_record_id='12345'").fetchall()
        assert len(rows) == 1
        db.close()
    finally:
        os.unlink(db_path)


def test_insert_sighting_duplicate_ignored():
    """Duplicate alert_record_id should be ignored (upsert)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = GhostDB(db_path)
        record = {
            "alert_record_id": "99999",
            "lat": 22.30,
            "lng": 114.15,
            "address": "Test St",
            "alert_type": "alert",
            "create_dt": "2026-06-13 10:00:00",
            "upvote": 1,
            "downvote": 0,
        }
        db.insert_sightings([record])
        db.insert_sightings([record])  # duplicate
        rows = db.execute("SELECT * FROM sightings WHERE alert_record_id='99999'").fetchall()
        assert len(rows) == 1
        db.close()
    finally:
        os.unlink(db_path)


def test_insert_poll_cycle():
    """Should record a poll cycle."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = GhostDB(db_path)
        db.insert_poll_cycle(
            timestamp="2026-06-13T16:05:00Z",
            total_alerts=87,
            new_alerts=12,
            duration_sec=32.5,
        )
        rows = db.execute("SELECT * FROM poll_cycles").fetchall()
        assert len(rows) == 1
        db.close()
    finally:
        os.unlink(db_path)
