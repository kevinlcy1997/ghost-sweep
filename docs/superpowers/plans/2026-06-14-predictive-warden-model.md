# Ghost Sweep Predictive Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a spatiotemporal prediction pipeline that forecasts traffic warden activity per ~500m grid cell using LightGBM on historically collected alert data.

**Architecture:** Raw alerts flow from `ghost_listener.py` → SQLite `sightings` table → event consolidation (20m/15min clustering) → `events` table → feature engineering → LightGBM model → JSON forecast with street-level recent events.

**Tech Stack:** Python 3.11+, SQLite, LightGBM, scikit-learn, pandas, shapely

---

## File Structure

| File | Responsibility |
|------|---------------|
| `ghost_db.py` | SQLite schema (sightings, events, poll_cycles), insert/query operations, JSON migration |
| `ghost_utils.py` | Shared utility: `compute_grid_cell()` function used by db and clean modules |
| `ghost_clean.py` | Spatiotemporal clustering (20m + 15min), spam filtering, event consolidation |
| `ghost_districts.py` | District boundary lookup (point-in-polygon or fallback), cell→district cache |
| `ghost_features.py` | Build ML feature vectors from events table for (cell, time_window) pairs |
| `ghost_model.py` | LightGBM train/evaluate/serialize, data gate check, metrics |
| `ghost_predict.py` | CLI entry point: train, forecast, stats, districts commands |
| `ghost_listener.py` | Minor addition: call `ghost_db.insert_sightings()` after each poll cycle |
| `requirements.txt` | New dependencies |
| `tests/test_ghost_db.py` | Tests for DB operations |
| `tests/test_ghost_clean.py` | Tests for event clustering |
| `tests/test_ghost_districts.py` | Tests for district assignment |
| `tests/test_ghost_features.py` | Tests for feature computation |
| `tests/test_ghost_model.py` | Tests for model train/predict |
| `ghost_utils.py` | Shared: `compute_grid_cell()` function |

---

### Task 1: Project Setup & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
cryptography
requests
lightgbm
scikit-learn
pandas
shapely
pytest
```

- [ ] **Step 1b: Create ghost_utils.py**

```python
# ghost_utils.py
"""Shared utilities for Ghost Sweep modules."""


def compute_grid_cell(lat: float, lng: float) -> str:
    """Snap lat/lng to 0.005° grid cell (~500m)."""
    cell_lat = round(lat / 0.005) * 0.005
    cell_lng = round(lng / 0.005) * 0.005
    return f"{cell_lat:.3f}_{cell_lng:.3f}"
```

- [ ] **Step 2: Install dependencies**

Run: `py -m pip install -r requirements.txt`
Expected: All packages install successfully

- [ ] **Step 3: Create tests directory**

Create `tests/__init__.py` (empty file to make tests discoverable)

- [ ] **Step 4: Verify pytest works**

Run: `py -m pytest tests/ -v`
Expected: "no tests ran" (clean baseline)

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/__init__.py
git commit -m "feat: add ML dependencies and test directory"
```

---

### Task 2: SQLite Database Layer (`ghost_db.py`)

**Files:**
- Create: `ghost_db.py`
- Create: `tests/test_ghost_db.py`

- [ ] **Step 1: Write the failing test for schema creation**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/test_ghost_db.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'ghost_db'"

- [ ] **Step 3: Implement ghost_db.py**

```python
# ghost_db.py
"""SQLite database layer for Ghost Sweep alert data."""

import sqlite3
from datetime import datetime, timezone


class GhostDB:
    """Manages the ghost_alerts.db SQLite database."""

    def __init__(self, db_path: str = "ghost_alerts.db"):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sightings (
                alert_record_id TEXT PRIMARY KEY,
                lat REAL NOT NULL,
                lng REAL NOT NULL,
                address TEXT,
                alert_type TEXT,
                create_dt TEXT,
                upvote INTEGER DEFAULT 0,
                downvote INTEGER DEFAULT 0,
                grid_cell TEXT,
                district TEXT,
                region TEXT,
                first_seen TEXT,
                last_seen TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL,
                lng REAL NOT NULL,
                address TEXT,
                create_dt TEXT,
                end_dt TEXT,
                duration_min REAL,
                report_count INTEGER DEFAULT 1,
                total_upvotes INTEGER DEFAULT 0,
                total_downvotes INTEGER DEFAULT 0,
                grid_cell TEXT,
                district TEXT,
                region TEXT
            );

            CREATE TABLE IF NOT EXISTS poll_cycles (
                cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_alerts INTEGER,
                new_alerts INTEGER,
                duration_sec REAL
            );

            CREATE INDEX IF NOT EXISTS idx_sightings_cell_dt
                ON sightings(grid_cell, create_dt);
            CREATE INDEX IF NOT EXISTS idx_sightings_district_dt
                ON sightings(district, create_dt);
            CREATE INDEX IF NOT EXISTS idx_sightings_dt
                ON sightings(create_dt);
            CREATE INDEX IF NOT EXISTS idx_events_cell_dt
                ON events(grid_cell, create_dt);
            CREATE INDEX IF NOT EXISTS idx_events_district_dt
                ON events(district, create_dt);
        """)
        self._conn.commit()

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def insert_sightings(self, records: list[dict]):
        """Insert sighting records, ignoring duplicates."""
        for rec in records:
            grid_cell = self._compute_grid_cell(rec.get("lat", 0), rec.get("lng", 0))
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute("""
                INSERT OR IGNORE INTO sightings
                (alert_record_id, lat, lng, address, alert_type, create_dt,
                 upvote, downvote, grid_cell, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(rec.get("alert_record_id", "")),
                rec.get("lat", 0),
                rec.get("lng", 0),
                rec.get("address", ""),
                rec.get("alert_type", ""),
                rec.get("create_dt", ""),
                int(rec.get("upvote", 0)),
                int(rec.get("downvote", 0)),
                grid_cell,
                now,
                now,
            ))
        self._conn.commit()

    def insert_poll_cycle(self, timestamp: str, total_alerts: int,
                          new_alerts: int, duration_sec: float):
        """Record a completed poll cycle."""
        self._conn.execute("""
            INSERT INTO poll_cycles (timestamp, total_alerts, new_alerts, duration_sec)
            VALUES (?, ?, ?, ?)
        """, (timestamp, total_alerts, new_alerts, duration_sec))
        self._conn.commit()

    def get_unprocessed_sightings(self) -> list[dict]:
        """Get sightings not yet assigned to an event (no matching event)."""
        rows = self._conn.execute("""
            SELECT s.* FROM sightings s
            WHERE s.lat != 0 AND s.lng != 0
              AND s.lat BETWEEN 22.15 AND 22.56
              AND s.lng BETWEEN 113.83 AND 114.41
              AND NOT (s.upvote = 0 AND s.downvote >= 3)
            ORDER BY s.create_dt
        """).fetchall()
        return [dict(row) for row in rows]

    def insert_events(self, events: list[dict]):
        """Insert consolidated event records."""
        for ev in events:
            self._conn.execute("""
                INSERT INTO events
                (lat, lng, address, create_dt, end_dt, duration_min,
                 report_count, total_upvotes, total_downvotes, grid_cell, district, region)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ev["lat"], ev["lng"], ev.get("address", ""),
                ev["create_dt"], ev.get("end_dt", ev["create_dt"]),
                ev.get("duration_min", 0),
                ev.get("report_count", 1),
                ev.get("total_upvotes", 0),
                ev.get("total_downvotes", 0),
                ev.get("grid_cell", ""),
                ev.get("district", ""),
                ev.get("region", ""),
            ))
        self._conn.commit()

    def get_events_since(self, since_dt: str) -> list[dict]:
        """Get events created after a given datetime string."""
        rows = self._conn.execute(
            "SELECT * FROM events WHERE create_dt >= ? ORDER BY create_dt",
            (since_dt,)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_all_events(self) -> list[dict]:
        """Get all events ordered by create_dt."""
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY create_dt"
        ).fetchall()
        return [dict(row) for row in rows]

    def count_days_collected(self) -> int:
        """Count distinct days with at least one poll cycle."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT date(timestamp)) FROM poll_cycles"
        ).fetchone()
        return row[0] if row else 0

    @staticmethod
    def _compute_grid_cell(lat: float, lng: float) -> str:
        from ghost_utils import compute_grid_cell
        return compute_grid_cell(lat, lng)

    def close(self):
        self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/test_ghost_db.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ghost_db.py tests/test_ghost_db.py
git commit -m "feat: SQLite database layer with sightings, events, poll_cycles tables"
```

---

### Task 3: JSON Migration

**Files:**
- Create: `tests/test_migration.py`
- Modify: `ghost_db.py` (add `migrate_from_json` method)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_migration.py
import json
import os
import tempfile
import pytest
from ghost_db import GhostDB


def test_migrate_from_json():
    """Should import alerts from ghost_alerts.json into sightings table."""
    alerts_data = {
        "alerts": {
            "5233453": {
                "alert_record_id": "5233453",
                "lat": 22.315428,
                "lng": 114.169763,
                "address": "37 Dundas St, Mong Kok",
                "alert_type": "alert",
                "create_dt": "2026-06-13 11:37:50",
                "upvote": "3",
                "downvote": "0",
            },
            "5233500": {
                "alert_record_id": "5233500",
                "lat": 22.280100,
                "lng": 114.158200,
                "address": "Nathan Rd, Tsim Sha Tsui",
                "alert_type": "alert",
                "create_dt": "2026-06-13 12:00:00",
                "upvote": "1",
                "downvote": "0",
            },
        },
        "meta": {"total_alerts": 2},
    }

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as jf:
        json.dump(alerts_data, jf)
        json_path = jf.name

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as df:
        db_path = df.name

    try:
        db = GhostDB(db_path)
        count = db.migrate_from_json(json_path)
        assert count == 2
        rows = db.execute("SELECT * FROM sightings").fetchall()
        assert len(rows) == 2
        db.close()
    finally:
        os.unlink(json_path)
        os.unlink(db_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_migration.py -v`
Expected: FAIL with "AttributeError: 'GhostDB' object has no attribute 'migrate_from_json'"

- [ ] **Step 3: Add migrate_from_json to ghost_db.py**

Add this method to the `GhostDB` class:

```python
    def migrate_from_json(self, json_path: str) -> int:
        """Import alerts from ghost_alerts.json into sightings table. Returns count imported."""
        import json as json_mod
        with open(json_path, "r", encoding="utf-8") as f:
            data = json_mod.load(f)
        alerts = data.get("alerts", {})
        records = []
        for alert_id, rec in alerts.items():
            records.append({
                "alert_record_id": str(rec.get("alert_record_id", alert_id)),
                "lat": float(rec.get("lat", 0)),
                "lng": float(rec.get("lng", 0)),
                "address": rec.get("address", ""),
                "alert_type": rec.get("alert_type", ""),
                "create_dt": rec.get("create_dt", ""),
                "upvote": int(rec.get("upvote", 0)),
                "downvote": int(rec.get("downvote", 0)),
            })
        self.insert_sightings(records)
        return len(records)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_migration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ghost_db.py tests/test_migration.py
git commit -m "feat: JSON-to-SQLite migration for existing ghost_alerts.json"
```

---

### Task 4: Data Cleaning — Event Consolidation (`ghost_clean.py`)

**Files:**
- Create: `ghost_clean.py`
- Create: `tests/test_ghost_clean.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ghost_clean.py
import pytest
from ghost_clean import consolidate_events


def test_single_alert_becomes_one_event():
    """A single alert should produce one event."""
    sightings = [{
        "alert_record_id": "1",
        "lat": 22.3154,
        "lng": 114.1698,
        "address": "37 Dundas St",
        "create_dt": "2026-06-13 11:37:50",
        "upvote": 3,
        "downvote": 0,
    }]
    events = consolidate_events(sightings)
    assert len(events) == 1
    assert events[0]["report_count"] == 1
    assert events[0]["lat"] == 22.3154


def test_nearby_alerts_merged():
    """Two alerts within 20m and 15min should merge into one event."""
    sightings = [
        {
            "alert_record_id": "1",
            "lat": 22.31540,
            "lng": 114.16980,
            "address": "37 Dundas St",
            "create_dt": "2026-06-13 11:37:50",
            "upvote": 3,
            "downvote": 0,
        },
        {
            "alert_record_id": "2",
            "lat": 22.31542,  # ~2m away
            "lng": 114.16982,
            "address": "37 Dundas St",
            "create_dt": "2026-06-13 11:40:00",  # 2 min later
            "upvote": 1,
            "downvote": 0,
        },
    ]
    events = consolidate_events(sightings)
    assert len(events) == 1
    assert events[0]["report_count"] == 2
    assert events[0]["total_upvotes"] == 4


def test_far_alerts_separate():
    """Two alerts >20m apart should be separate events."""
    sightings = [
        {
            "alert_record_id": "1",
            "lat": 22.3154,
            "lng": 114.1698,
            "address": "37 Dundas St",
            "create_dt": "2026-06-13 11:37:50",
            "upvote": 1,
            "downvote": 0,
        },
        {
            "alert_record_id": "2",
            "lat": 22.3200,  # ~500m away
            "lng": 114.1700,
            "address": "Argyle St",
            "create_dt": "2026-06-13 11:38:00",
            "upvote": 1,
            "downvote": 0,
        },
    ]
    events = consolidate_events(sightings)
    assert len(events) == 2


def test_time_gap_separates():
    """Two alerts at same location but >15min apart should be separate events."""
    sightings = [
        {
            "alert_record_id": "1",
            "lat": 22.3154,
            "lng": 114.1698,
            "address": "37 Dundas St",
            "create_dt": "2026-06-13 11:00:00",
            "upvote": 1,
            "downvote": 0,
        },
        {
            "alert_record_id": "2",
            "lat": 22.3154,
            "lng": 114.1698,
            "address": "37 Dundas St",
            "create_dt": "2026-06-13 11:20:00",  # 20 min later
            "upvote": 1,
            "downvote": 0,
        },
    ]
    events = consolidate_events(sightings)
    assert len(events) == 2


def test_centroid_computed():
    """Event centroid should be average of merged alert coordinates."""
    sightings = [
        {
            "alert_record_id": "1",
            "lat": 22.3100,
            "lng": 114.1700,
            "address": "A St",
            "create_dt": "2026-06-13 11:00:00",
            "upvote": 1,
            "downvote": 0,
        },
        {
            "alert_record_id": "2",
            "lat": 22.3100,  # within 20m
            "lng": 114.17001,
            "address": "A St",
            "create_dt": "2026-06-13 11:01:00",
            "upvote": 2,
            "downvote": 0,
        },
    ]
    events = consolidate_events(sightings)
    assert len(events) == 1
    assert abs(events[0]["lat"] - 22.3100) < 0.0001
    assert events[0]["address"] == "A St"  # highest upvote address
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/test_ghost_clean.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'ghost_clean'"

- [ ] **Step 3: Implement ghost_clean.py**

```python
# ghost_clean.py
"""Data cleaning: consolidate raw sightings into unique warden events."""

import math
from datetime import datetime


# Thresholds
SPATIAL_THRESHOLD_M = 20  # meters
TEMPORAL_THRESHOLD_MIN = 15  # minutes


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance in meters between two lat/lng points."""
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_dt(dt_str: str) -> datetime:
    """Parse datetime string (flexible formats)."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return datetime.strptime(dt_str[:19], "%Y-%m-%dT%H:%M:%S")


def consolidate_events(sightings: list[dict]) -> list[dict]:
    """
    Consolidate raw sightings into events using greedy spatiotemporal clustering.

    Two alerts merge if within 20m AND 15min of the cluster's latest report.
    Returns list of event dicts ready for DB insertion.
    """
    if not sightings:
        return []

    # Sort by create_dt
    sorted_sightings = sorted(sightings, key=lambda s: s.get("create_dt", ""))

    clusters: list[dict] = []  # Each: {lats, lngs, alerts, latest_dt, addresses_votes}

    for sight in sorted_sightings:
        lat = float(sight.get("lat", 0))
        lng = float(sight.get("lng", 0))
        dt = _parse_dt(sight.get("create_dt", "2000-01-01 00:00:00"))

        merged = False
        for cluster in clusters:
            # Check temporal distance against cluster's latest report
            time_diff = (dt - cluster["latest_dt"]).total_seconds() / 60.0
            if time_diff > TEMPORAL_THRESHOLD_MIN:
                continue

            # Check spatial distance against cluster centroid
            c_lat = sum(cluster["lats"]) / len(cluster["lats"])
            c_lng = sum(cluster["lngs"]) / len(cluster["lngs"])
            dist = _haversine_m(lat, lng, c_lat, c_lng)
            if dist <= SPATIAL_THRESHOLD_M:
                # Merge into this cluster
                cluster["lats"].append(lat)
                cluster["lngs"].append(lng)
                cluster["alerts"].append(sight)
                cluster["latest_dt"] = dt
                merged = True
                break

        if not merged:
            clusters.append({
                "lats": [lat],
                "lngs": [lng],
                "alerts": [sight],
                "latest_dt": dt,
            })

    # Convert clusters to event records
    events = []
    for cluster in clusters:
        alerts = cluster["alerts"]
        centroid_lat = sum(cluster["lats"]) / len(cluster["lats"])
        centroid_lng = sum(cluster["lngs"]) / len(cluster["lngs"])

        # Address from alert with highest upvotes
        best_alert = max(alerts, key=lambda a: int(a.get("upvote", 0)))
        address = best_alert.get("address", "")

        create_dt = alerts[0].get("create_dt", "")
        end_dt = alerts[-1].get("create_dt", create_dt)

        dt_start = _parse_dt(create_dt)
        dt_end = _parse_dt(end_dt)
        duration_min = (dt_end - dt_start).total_seconds() / 60.0

        total_up = sum(int(a.get("upvote", 0)) for a in alerts)
        total_down = sum(int(a.get("downvote", 0)) for a in alerts)

        grid_cell = _compute_grid_cell(centroid_lat, centroid_lng)

        events.append({
            "lat": centroid_lat,
            "lng": centroid_lng,
            "address": address,
            "create_dt": create_dt,
            "end_dt": end_dt,
            "duration_min": duration_min,
            "report_count": len(alerts),
            "total_upvotes": total_up,
            "total_downvotes": total_down,
            "grid_cell": grid_cell,
            "district": "",  # Filled later by ghost_districts.py
            "region": "",
        })

    return events


def _compute_grid_cell(lat: float, lng: float) -> str:
    from ghost_utils import compute_grid_cell
    return compute_grid_cell(lat, lng)
```

- [ ] **Step 4: Create `ghost_utils.py` with shared grid cell function**

```python
# ghost_utils.py
"""Shared utilities for Ghost Sweep modules."""


def compute_grid_cell(lat: float, lng: float) -> str:
    """Snap lat/lng to 0.005° grid cell (~500m)."""
    cell_lat = round(lat / 0.005) * 0.005
    cell_lng = round(lng / 0.005) * 0.005
    return f"{cell_lat:.3f}_{cell_lng:.3f}"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `py -m pytest tests/test_ghost_clean.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add ghost_utils.py ghost_clean.py tests/test_ghost_clean.py
git commit -m "feat: event consolidation with 20m/15min spatiotemporal clustering"
```

---

### Task 5: District Assignment (`ghost_districts.py`)

**Files:**
- Create: `ghost_districts.py`
- Create: `tests/test_ghost_districts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ghost_districts.py
import pytest
from ghost_districts import get_district


def test_mong_kok_district():
    """A coordinate near Mong Kok (not exact station) should map to Mong Kok."""
    # Langham Place, ~200m from station
    district, region = get_district(22.3182, 114.1685)
    assert district == "Mong Kok"
    assert region == "Kowloon West"


def test_central_district():
    """A coordinate in Central (not exact station) should map to Central."""
    # IFC Mall, ~300m from Central station
    district, region = get_district(22.2855, 114.1580)
    assert district == "Central"
    assert region == "Hong Kong Island"


def test_ocean_fallback():
    """A point in open water should map to nearest district."""
    district, region = get_district(22.30, 114.00)
    # Should return something (nearest), not empty
    assert district != ""
    assert region != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_ghost_districts.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'ghost_districts'"

- [ ] **Step 3: Implement ghost_districts.py (fallback: station proximity)**

```python
# ghost_districts.py
"""Police district assignment for grid cells.

Uses a fallback approach: maps coordinates to nearest police station.
When GeoJSON boundary data is available, upgrades to point-in-polygon.
"""

import math
from functools import lru_cache

# Police station coordinates (approximate centroids per district)
# Source: HK Police Force website station addresses, geocoded
DISTRICT_STATIONS: dict[str, dict] = {
    "Eastern": {"lat": 22.2870, "lng": 114.2190, "region": "Hong Kong Island"},
    "Wan Chai": {"lat": 22.2780, "lng": 114.1720, "region": "Hong Kong Island"},
    "Central": {"lat": 22.2816, "lng": 114.1585, "region": "Hong Kong Island"},
    "Western": {"lat": 22.2870, "lng": 114.1420, "region": "Hong Kong Island"},
    "Wong Tai Sin": {"lat": 22.3420, "lng": 114.1930, "region": "Kowloon East"},
    "Kwun Tong": {"lat": 22.3130, "lng": 114.2250, "region": "Kowloon East"},
    "Tseung Kwan O": {"lat": 22.3170, "lng": 114.2590, "region": "Kowloon East"},
    "Sau Mau Ping": {"lat": 22.3290, "lng": 114.2320, "region": "Kowloon East"},
    "Yau Tsim": {"lat": 22.2980, "lng": 114.1720, "region": "Kowloon West"},
    "Mong Kok": {"lat": 22.3193, "lng": 114.1694, "region": "Kowloon West"},
    "Sham Shui Po": {"lat": 22.3310, "lng": 114.1590, "region": "Kowloon West"},
    "Kowloon City": {"lat": 22.3280, "lng": 114.1870, "region": "Kowloon West"},
    "Tai Po": {"lat": 22.4510, "lng": 114.1680, "region": "New Territories North"},
    "Tuen Mun": {"lat": 22.3910, "lng": 113.9770, "region": "New Territories North"},
    "Yuen Long": {"lat": 22.4440, "lng": 114.0220, "region": "New Territories North"},
    "Border": {"lat": 22.5030, "lng": 114.1280, "region": "New Territories North"},
    "Tsuen Wan": {"lat": 22.3710, "lng": 114.1140, "region": "New Territories South"},
    "Kwai Tsing": {"lat": 22.3560, "lng": 114.1300, "region": "New Territories South"},
    "Sha Tin": {"lat": 22.3810, "lng": 114.1880, "region": "New Territories South"},
    "Airport": {"lat": 22.3080, "lng": 113.9185, "region": "New Territories South"},
    "Lantau": {"lat": 22.2660, "lng": 113.9430, "region": "New Territories South"},
}


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance in meters."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@lru_cache(maxsize=4096)
def get_district(lat: float, lng: float) -> tuple[str, str]:
    """
    Map a coordinate to (district, region).
    Uses nearest-station fallback. Cache results for repeated lookups.
    """
    best_dist = float("inf")
    best_name = ""
    best_region = ""
    for name, info in DISTRICT_STATIONS.items():
        d = _haversine_m(lat, lng, info["lat"], info["lng"])
        if d < best_dist:
            best_dist = d
            best_name = name
            best_region = info["region"]
    return best_name, best_region


def assign_district_to_events(events: list[dict]) -> list[dict]:
    """Fill in district and region fields for a list of event dicts."""
    for ev in events:
        district, region = get_district(ev["lat"], ev["lng"])
        ev["district"] = district
        ev["region"] = region
    return events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/test_ghost_districts.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ghost_districts.py tests/test_ghost_districts.py
git commit -m "feat: district assignment via nearest-station proximity lookup"
```

---

### Task 6: Feature Engineering (`ghost_features.py`)

**Files:**
- Create: `ghost_features.py`
- Create: `tests/test_ghost_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ghost_features.py
import pytest
from datetime import datetime
from ghost_features import build_features


def test_build_features_basic():
    """Should produce a DataFrame with expected columns."""
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
    """Hour feature should match the target prediction time."""
    events = [
        {"event_id": 1, "lat": 22.315, "lng": 114.170, "create_dt": "2026-06-13 11:00:00",
         "end_dt": "2026-06-13 11:00:00", "duration_min": 0, "report_count": 1,
         "total_upvotes": 1, "total_downvotes": 0,
         "grid_cell": "22.315_114.170", "district": "Mong Kok", "region": "Kowloon West"},
    ]
    target_dt = datetime(2026, 6, 13, 15, 0, 0)
    df = build_features(events, target_dt)
    assert (df["hour"] == 15).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/test_ghost_features.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'ghost_features'"

- [ ] **Step 3: Implement ghost_features.py**

```python
# ghost_features.py
"""Feature engineering for Ghost Sweep predictive model."""

from datetime import datetime, timedelta
import pandas as pd


def build_features(events: list[dict], target_dt: datetime,
                   window_hours: int = 1) -> pd.DataFrame:
    """
    Build feature vectors for all active cells at a given prediction time.

    Args:
        events: List of event dicts (from DB or test data)
        target_dt: The time we're predicting FOR (start of forecast window)
        window_hours: Forecast window size (default 1h)

    Returns:
        DataFrame with one row per active cell, columns are features + cell metadata
    """
    if not events:
        return pd.DataFrame()

    df_events = pd.DataFrame(events)
    df_events["create_dt_parsed"] = pd.to_datetime(df_events["create_dt"])

    # Find all unique cells that have ever had activity
    active_cells = df_events[["grid_cell", "district", "region"]].drop_duplicates("grid_cell")

    # Filter events before target_dt (can't use future data)
    past_events = df_events[df_events["create_dt_parsed"] < target_dt].copy()

    # Temporal features (same for all cells at this prediction time)
    hour = target_dt.hour
    day_of_week = target_dt.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    month = target_dt.month

    # Time boundaries
    dt_24h_ago = target_dt - timedelta(hours=24)
    dt_7d_ago = target_dt - timedelta(days=7)

    rows = []
    for _, cell_row in active_cells.iterrows():
        cell_id = cell_row["grid_cell"]
        district = cell_row["district"]
        region = cell_row["region"]

        # Cell-level events
        cell_events = past_events[past_events["grid_cell"] == cell_id]
        cell_24h = cell_events[cell_events["create_dt_parsed"] >= dt_24h_ago]
        cell_7d = cell_events[cell_events["create_dt_parsed"] >= dt_7d_ago]

        # Cell features
        cell_historical_freq = len(cell_events)
        cell_7d_count = len(cell_7d)
        cell_24h_count = len(cell_24h)

        # Last seen
        if len(cell_events) > 0:
            last_event_dt = cell_events["create_dt_parsed"].max()
            cell_last_seen_hours_ago = (target_dt - last_event_dt).total_seconds() / 3600
        else:
            cell_last_seen_hours_ago = 9999

        # Upvote ratio
        total_up = cell_events["total_upvotes"].sum()
        total_down = cell_events["total_downvotes"].sum()
        upvote_ratio = total_up / (total_up + total_down) if (total_up + total_down) > 0 else 0.5

        # Avg report count
        avg_report_count = cell_events["report_count"].mean() if len(cell_events) > 0 else 0

        # Streak: events on consecutive days
        if len(cell_events) > 0:
            event_days = cell_events["create_dt_parsed"].dt.date.unique()
            event_days_sorted = sorted(event_days)
            streak = 0
            if len(event_days_sorted) >= 2:
                for i in range(len(event_days_sorted) - 1, 0, -1):
                    diff = (event_days_sorted[i] - event_days_sorted[i-1]).days
                    if diff == 1:
                        streak = 1
                        break
        else:
            streak = 0

        # District-level features
        district_events = past_events[past_events["district"] == district]
        district_24h = district_events[district_events["create_dt_parsed"] >= dt_24h_ago]
        district_24h_count = len(district_24h)

        # District historical rate (events per day)
        if len(district_events) > 0:
            first_event = district_events["create_dt_parsed"].min()
            days_span = max((target_dt - first_event).days, 1)
            district_historical_rate = len(district_events) / days_span
        else:
            district_historical_rate = 0

        # District active cells in 24h
        district_active_cells = district_24h["grid_cell"].nunique()

        # District hour rate
        district_at_hour = district_events[
            district_events["create_dt_parsed"].dt.hour == hour
        ]
        if len(district_events) > 0:
            first_event = district_events["create_dt_parsed"].min()
            days_span = max((target_dt - first_event).days, 1)
            district_hour_rate = len(district_at_hour) / days_span
        else:
            district_hour_rate = 0

        # Cross features: hour and dow_hour rates (global)
        all_at_hour = past_events[past_events["create_dt_parsed"].dt.hour == hour]
        if len(past_events) > 0:
            first_global = past_events["create_dt_parsed"].min()
            global_days = max((target_dt - first_global).days, 1)
            hour_historical_rate = len(all_at_hour) / global_days
        else:
            hour_historical_rate = 0

        all_at_dow_hour = past_events[
            (past_events["create_dt_parsed"].dt.hour == hour) &
            (past_events["create_dt_parsed"].dt.weekday() == day_of_week)
        ]
        dow_weeks = max((target_dt - past_events["create_dt_parsed"].min()).days // 7, 1) if len(past_events) > 0 else 1
        dow_hour_rate = len(all_at_dow_hour) / dow_weeks

        # Neighbor 24h count (approximate: cells within 0.005° of center)
        # For simplicity, check cells with lat ±0.005 and lng ±0.005
        cell_parts = cell_id.split("_")
        if len(cell_parts) == 2:
            c_lat, c_lng = float(cell_parts[0]), float(cell_parts[1])
            neighbor_cells = []
            for dlat in (-0.005, 0, 0.005):
                for dlng in (-0.005, 0, 0.005):
                    if dlat == 0 and dlng == 0:
                        continue
                    n_cell = f"{c_lat + dlat:.3f}_{c_lng + dlng:.3f}"
                    neighbor_cells.append(n_cell)
            neighbor_events = past_events[
                (past_events["grid_cell"].isin(neighbor_cells)) &
                (past_events["create_dt_parsed"] >= dt_24h_ago)
            ]
            neighbor_24h_count = len(neighbor_events)
        else:
            neighbor_24h_count = 0

        rows.append({
            "grid_cell": cell_id,
            "district": district,
            "region": region,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "month": month,
            "cell_historical_freq": cell_historical_freq,
            "cell_7d_count": cell_7d_count,
            "cell_24h_count": cell_24h_count,
            "cell_last_seen_hours_ago": cell_last_seen_hours_ago,
            "neighbor_24h_count": neighbor_24h_count,
            "streak_active": streak,
            "upvote_ratio": upvote_ratio,
            "avg_report_count": avg_report_count,
            "district_24h_count": district_24h_count,
            "district_historical_rate": district_historical_rate,
            "district_active_cells": district_active_cells,
            "district_hour_rate": district_hour_rate,
            "hour_historical_rate": hour_historical_rate,
            "dow_hour_rate": dow_hour_rate,
        })

    return pd.DataFrame(rows)


def build_training_data(events: list[dict], window_hours: int = 1) -> pd.DataFrame:
    """
    Build training dataset: features + labels for all historical time windows.

    For each unique hour in the data, build features and label whether
    each active cell had an event in the following window_hours.
    """
    if not events:
        return pd.DataFrame()

    df_events = pd.DataFrame(events)
    df_events["create_dt_parsed"] = pd.to_datetime(df_events["create_dt"])

    # Get all unique hours in the data (rounded to hour)
    all_times = df_events["create_dt_parsed"].dt.floor("h").unique()
    all_times = sorted(all_times)

    all_rows = []
    for t in all_times:
        target_dt = t.to_pydatetime()
        window_end = target_dt + timedelta(hours=window_hours)

        # Build features at this time
        features_df = build_features(events, target_dt, window_hours)
        if features_df.empty:
            continue

        # Build labels: did each cell have an event in [target_dt, window_end)?
        future_events = df_events[
            (df_events["create_dt_parsed"] >= target_dt) &
            (df_events["create_dt_parsed"] < window_end)
        ]
        active_cells_in_window = set(future_events["grid_cell"].unique())
        features_df["has_warden"] = features_df["grid_cell"].isin(active_cells_in_window).astype(int)

        all_rows.append(features_df)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/test_ghost_features.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ghost_features.py tests/test_ghost_features.py
git commit -m "feat: feature engineering with temporal, cell, district, and cross features"
```

---

### Task 7: Model Training (`ghost_model.py`)

**Files:**
- Create: `ghost_model.py`
- Create: `tests/test_ghost_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ghost_model.py
import pytest
import pandas as pd
from ghost_model import GhostModel


def test_data_gate_blocks_training():
    """Should refuse to train with insufficient data."""
    model = GhostModel()
    result = model.check_data_gate(days_collected=5)
    assert result["ready"] is False
    assert result["days_needed"] == 14


def test_data_gate_allows_training():
    """Should allow training with sufficient data."""
    model = GhostModel()
    result = model.check_data_gate(days_collected=14)
    assert result["ready"] is True


def test_train_and_predict():
    """Should train on synthetic data and produce predictions."""
    import numpy as np

    rng = np.random.default_rng(42)  # deterministic
    model = GhostModel()

    # Synthetic training data with a learnable signal:
    # has_warden correlates with high cell_24h_count
    n = 200
    cell_24h = rng.integers(0, 10, n)
    has_warden = (cell_24h >= 5).astype(int)  # clear signal

    df = pd.DataFrame({
        "grid_cell": [f"22.{i%10:03d}_114.{i%10:03d}" for i in range(n)],
        "district": ["Mong Kok"] * n,
        "region": ["Kowloon West"] * n,
        "hour": [i % 24 for i in range(n)],
        "day_of_week": [i % 7 for i in range(n)],
        "is_weekend": [1 if i % 7 >= 5 else 0 for i in range(n)],
        "month": [6] * n,
        "cell_historical_freq": rng.integers(1, 50, n),
        "cell_7d_count": rng.integers(0, 10, n),
        "cell_24h_count": cell_24h,
        "cell_last_seen_hours_ago": rng.uniform(0, 100, n),
        "neighbor_24h_count": rng.integers(0, 10, n),
        "streak_active": rng.integers(0, 2, n),
        "upvote_ratio": rng.uniform(0.5, 1.0, n),
        "avg_report_count": rng.uniform(1, 5, n),
        "district_24h_count": rng.integers(5, 30, n),
        "district_historical_rate": rng.uniform(1, 10, n),
        "district_active_cells": rng.integers(1, 10, n),
        "district_hour_rate": rng.uniform(0, 3, n),
        "hour_historical_rate": rng.uniform(0, 5, n),
        "dow_hour_rate": rng.uniform(0, 2, n),
        "has_warden": has_warden,
    })

    metrics = model.train(df)
    assert "auc_roc" in metrics
    assert metrics["auc_roc"] >= 0.6  # learnable signal should give decent AUC
    assert metrics["n_estimators"] > 0

    # Predict
    pred_df = df.drop(columns=["has_warden"]).head(10)
    predictions = model.predict(pred_df)
    assert len(predictions) == 10
    assert all(0 <= p <= 1 for p in predictions)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/test_ghost_model.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'ghost_model'"

- [ ] **Step 3: Implement ghost_model.py**

```python
# ghost_model.py
"""LightGBM model for Ghost Sweep warden activity prediction."""

import os
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib

log = logging.getLogger("ghost_model")

FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend", "month",
    "cell_historical_freq", "cell_7d_count", "cell_24h_count",
    "cell_last_seen_hours_ago", "neighbor_24h_count", "streak_active",
    "upvote_ratio", "avg_report_count",
    "district_24h_count", "district_historical_rate",
    "district_active_cells", "district_hour_rate",
    "hour_historical_rate", "dow_hour_rate",
]

CATEGORICAL_COLS = ["district", "region"]
LABEL_COL = "has_warden"
MODEL_DIR = Path("models")


class GhostModel:
    """LightGBM-based warden activity predictor."""

    def __init__(self, model_path: str | None = None):
        self._model: lgb.Booster | None = None
        if model_path and os.path.exists(model_path):
            self._model = joblib.load(model_path)

    def check_data_gate(self, days_collected: int) -> dict:
        """Check if enough data has been collected for training."""
        min_days = 14
        return {
            "ready": days_collected >= min_days,
            "days_collected": days_collected,
            "days_needed": min_days,
            "days_remaining": max(0, min_days - days_collected),
        }

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the model on a labeled DataFrame.
        Uses 80/10/10 time-based split.
        Returns evaluation metrics dict.
        """
        # Prepare features
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

        X = df[feature_cols + cat_cols].copy()
        y = df[LABEL_COL]

        # Encode categoricals
        for col in cat_cols:
            X[col] = X[col].astype("category")

        # Time-based split (data assumed to be in chronological order)
        n = len(X)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        # Create LightGBM datasets
        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain)

        # Train
        params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "is_unbalance": True,
            "verbose": -1,
            "seed": 42,
        }

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        self._model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        # Evaluate on test set
        y_pred_prob = self._model.predict(X_test)
        y_pred_bin = (y_pred_prob >= 0.5).astype(int)

        metrics = {
            "auc_roc": float(roc_auc_score(y_test, y_pred_prob)) if y_test.nunique() > 1 else 0.0,
            "precision": float(precision_score(y_test, y_pred_bin, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_bin, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_bin, zero_division=0)),
            "train_size": train_end,
            "val_size": val_end - train_end,
            "test_size": n - val_end,
            "n_estimators": self._model.num_trees(),
        }

        # Feature importance
        importance = self._model.feature_importance(importance_type="gain")
        feat_names = self._model.feature_name()
        top_features = sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)[:10]
        metrics["top_features"] = [{"name": n, "importance": float(v)} for n, v in top_features]

        # Save model
        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / "model_latest.joblib"
        joblib.dump(self._model, model_path)
        metrics["model_path"] = str(model_path)

        log.info("Model trained: AUC=%.4f, F1=%.4f, trees=%d",
                 metrics["auc_roc"], metrics["f1"], metrics["n_estimators"])
        return metrics

    def predict(self, df: pd.DataFrame) -> list[float]:
        """Predict warden probability for each row."""
        if self._model is None:
            raise RuntimeError("No trained model loaded. Run train() first.")

        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

        X = df[feature_cols + cat_cols].copy()
        for col in cat_cols:
            X[col] = X[col].astype("category")

        predictions = self._model.predict(X)
        return predictions.tolist()

    def save(self, path: str | None = None):
        """Save model to disk."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        path = path or str(MODEL_DIR / "model_latest.joblib")
        Path(path).parent.mkdir(exist_ok=True)
        joblib.dump(self._model, path)

    def load(self, path: str):
        """Load model from disk."""
        self._model = joblib.load(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/test_ghost_model.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ghost_model.py tests/test_ghost_model.py
git commit -m "feat: LightGBM model with data gate, train, predict, and metrics"
```

> **Note:** The spec mentions "weekly automatic retrain." This is deferred to Phase 2 — for now, retraining is manual via `python ghost_predict.py train`. A scheduler (cron/Windows Task Scheduler) can be added later.

---

### Task 8: CLI Interface (`ghost_predict.py`)

**Files:**
- Create: `ghost_predict.py`

- [ ] **Step 1: Implement ghost_predict.py**

```python
# ghost_predict.py
"""CLI entry point for Ghost Sweep prediction system."""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from ghost_db import GhostDB
from ghost_clean import consolidate_events
from ghost_districts import assign_district_to_events
from ghost_features import build_features, build_training_data
from ghost_model import GhostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ghost_predict")

DB_PATH = "ghost_alerts.db"
MODEL_PATH = "models/model_latest.joblib"
FORECAST_PATH = "ghost_forecast.json"


def cmd_clean(args):
    """Run event consolidation on raw sightings."""
    db = GhostDB(DB_PATH)
    log.info("Fetching raw sightings for cleaning...")
    sightings = db.get_unprocessed_sightings()
    log.info("  %d raw sightings", len(sightings))

    events = consolidate_events(sightings)
    events = assign_district_to_events(events)
    log.info("  %d events after consolidation", len(events))

    # Clear existing events and re-insert (full rebuild)
    db.execute("DELETE FROM events")
    db.insert_events(events)
    log.info("  Events table rebuilt with %d records", len(events))
    db.close()


def cmd_train(args):
    """Train the prediction model."""
    db = GhostDB(DB_PATH)

    # Data gate check
    days = db.count_days_collected()
    model = GhostModel()
    gate = model.check_data_gate(days)
    if not gate["ready"]:
        log.warning(
            "Insufficient data: %d/%d days collected. Need %d more days.",
            gate["days_collected"], gate["days_needed"], gate["days_remaining"],
        )
        db.close()
        sys.exit(1)

    # Build training data from events
    events = db.get_all_events()
    db.close()

    if not events:
        log.error("No events in database. Run 'clean' first.")
        sys.exit(1)

    log.info("Building training data from %d events...", len(events))
    train_df = build_training_data(events)
    log.info("  Training samples: %d", len(train_df))

    metrics = model.train(train_df)
    log.info("Training complete:")
    log.info("  AUC-ROC:   %.4f", metrics["auc_roc"])
    log.info("  Precision: %.4f", metrics["precision"])
    log.info("  Recall:    %.4f", metrics["recall"])
    log.info("  F1:        %.4f", metrics["f1"])
    log.info("  Trees:     %d", metrics["n_estimators"])
    log.info("  Model saved to: %s", metrics["model_path"])

    if metrics.get("top_features"):
        log.info("  Top features:")
        for f in metrics["top_features"][:5]:
            log.info("    %s: %.1f", f["name"], f["importance"])


def cmd_forecast(args):
    """Generate a forecast for the next N hours."""
    hours = args.hours

    if not Path(MODEL_PATH).exists():
        log.error("No trained model found. Run 'train' first.")
        sys.exit(1)

    db = GhostDB(DB_PATH)
    events = db.get_all_events()
    db.close()

    model = GhostModel(MODEL_PATH)
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # tz-naive for pandas compat

    log.info("Generating %dh forecast at %s ...", hours, now.isoformat())
    features_df = build_features(events, now, hours)

    if features_df.empty:
        log.warning("No active cells to predict.")
        return

    predictions = model.predict(features_df)
    features_df["probability"] = predictions

    # Assign risk tiers
    features_df["risk"] = features_df["probability"].apply(
        lambda p: "high" if p >= 0.7 else ("medium" if p >= 0.4 else "low")
    )

    # Sort by probability descending
    features_df = features_df.sort_values("probability", ascending=False)

    # Build output with recent events
    dt_24h_ago = now - timedelta(hours=24)
    recent_events_by_cell = {}
    for ev in events:
        if ev.get("create_dt", "") >= dt_24h_ago.strftime("%Y-%m-%d %H:%M:%S"):
            cell = ev.get("grid_cell", "")
            if cell not in recent_events_by_cell:
                recent_events_by_cell[cell] = []
            recent_events_by_cell[cell].append({
                "lat": ev["lat"],
                "lng": ev["lng"],
                "address": ev.get("address", ""),
                "create_dt": ev.get("create_dt", ""),
                "report_count": ev.get("report_count", 1),
            })

    cells_output = []
    for _, row in features_df.iterrows():
        cell_id = row["grid_cell"]
        parts = cell_id.split("_")
        lat = float(parts[0]) if len(parts) == 2 else 0
        lng = float(parts[1]) if len(parts) == 2 else 0

        cell_out = {
            "cell": cell_id,
            "lat": lat,
            "lng": lng,
            "district": row.get("district", ""),
            "region": row.get("region", ""),
            "probability": round(row["probability"], 4),
            "risk": row["risk"],
        }
        if row["risk"] == "high" and cell_id in recent_events_by_cell:
            cell_out["recent_events"] = recent_events_by_cell[cell_id][:5]

        cells_output.append(cell_out)

    forecast = {
        "generated_at": now.isoformat(),
        "forecast_window": f"{hours}h",
        "model_version": Path(MODEL_PATH).stat().st_mtime if Path(MODEL_PATH).exists() else "",
        "cells": cells_output,
    }

    with open(FORECAST_PATH, "w", encoding="utf-8") as f:
        json.dump(forecast, f, ensure_ascii=False, indent=2)

    high_count = sum(1 for c in cells_output if c["risk"] == "high")
    med_count = sum(1 for c in cells_output if c["risk"] == "medium")
    log.info("Forecast saved to %s", FORECAST_PATH)
    log.info("  %d cells scored: %d high, %d medium", len(cells_output), high_count, med_count)


def cmd_stats(args):
    """Show model and data statistics."""
    db = GhostDB(DB_PATH)
    sighting_count = db.execute("SELECT COUNT(*) FROM sightings").fetchone()[0]
    event_count = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    days = db.count_days_collected()
    cycles = db.execute("SELECT COUNT(*) FROM poll_cycles").fetchone()[0]

    print(f"Ghost Sweep — Data Statistics")
    print(f"{'─' * 40}")
    print(f"Days collected:   {days}")
    print(f"Poll cycles:      {cycles}")
    print(f"Raw sightings:    {sighting_count}")
    print(f"Cleaned events:   {event_count}")
    print(f"Model exists:     {'Yes' if Path(MODEL_PATH).exists() else 'No'}")

    model = GhostModel()
    gate = model.check_data_gate(days)
    if gate["ready"]:
        print(f"Training gate:    READY")
    else:
        print(f"Training gate:    {gate['days_remaining']} more days needed")

    db.close()


def cmd_districts(args):
    """Show per-district activity summary."""
    db = GhostDB(DB_PATH)
    rows = db.execute("""
        SELECT district, region, COUNT(*) as event_count
        FROM events
        WHERE district != ''
        GROUP BY district, region
        ORDER BY event_count DESC
    """).fetchall()
    db.close()

    if not rows:
        print("No events in database yet.")
        return

    print(f"{'District':<20} {'Region':<25} {'Events':>8}")
    print(f"{'─' * 20} {'─' * 25} {'─' * 8}")
    for row in rows:
        print(f"{row[0]:<20} {row[1]:<25} {row[2]:>8}")


def cmd_migrate(args):
    """Migrate ghost_alerts.json into SQLite."""
    json_path = args.json_file
    db = GhostDB(DB_PATH)
    count = db.migrate_from_json(json_path)
    log.info("Migrated %d alerts from %s into %s", count, json_path, DB_PATH)
    db.close()


def main():
    parser = argparse.ArgumentParser(description="Ghost Sweep Prediction CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("clean", help="Run event consolidation")
    subparsers.add_parser("train", help="Train the prediction model")

    forecast_p = subparsers.add_parser("forecast", help="Generate forecast")
    forecast_p.add_argument("--hours", type=int, default=1, help="Forecast window in hours")

    subparsers.add_parser("stats", help="Show data statistics")
    subparsers.add_parser("districts", help="Per-district activity summary")

    migrate_p = subparsers.add_parser("migrate", help="Import ghost_alerts.json to SQLite")
    migrate_p.add_argument("json_file", nargs="?", default="ghost_alerts.json")

    args = parser.parse_args()

    commands = {
        "clean": cmd_clean,
        "train": cmd_train,
        "forecast": cmd_forecast,
        "stats": cmd_stats,
        "districts": cmd_districts,
        "migrate": cmd_migrate,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `py ghost_predict.py --help`
Expected: Shows usage with subcommands: clean, train, forecast, stats, districts, migrate

- [ ] **Step 3: Commit**

```bash
git add ghost_predict.py
git commit -m "feat: CLI entry point with train, forecast, stats, districts, migrate commands"
```

---

### Task 9: Integrate SQLite Write into `ghost_listener.py`

**Files:**
- Modify: `ghost_listener.py`

- [ ] **Step 1: Add import at top of ghost_listener.py**

After the existing imports (around line 145), add:

```python
from ghost_db import GhostDB
```

- [ ] **Step 2: Initialize DB in main()**

In the `main()` function, after `store = load_store(args.output)`, add:

```python
    db = GhostDB()
```

- [ ] **Step 3: Track new alerts and write only new ones to SQLite**

In `main()`, before the while loop, add a tracking set:

```python
    _inserted_ids: set[str] = set()
```

In the main loop, after `save_store(store, args.output)`, add:

```python
            # Write only NEW alerts to SQLite (avoid O(n) re-insert every cycle)
            new_alerts = [
                rec for rid, rec in store.get("alerts", {}).items()
                if str(rid) not in _inserted_ids
            ]
            if new_alerts:
                db.insert_sightings(new_alerts)
                _inserted_ids.update(str(r.get("alert_record_id", "")) for r in new_alerts)
            db.insert_poll_cycle(
                timestamp=store["meta"]["last_poll"],
                total_alerts=len(store["alerts"]),
                new_alerts=len(new_alerts),
                duration_sec=0,
            )
```

- [ ] **Step 4: Test the integration**

Run: `py ghost_listener.py --once`
Expected: Completes one poll cycle, creates/updates both `ghost_alerts.json` and `ghost_alerts.db`

Verify: `py -c "from ghost_db import GhostDB; db=GhostDB(); print(db.execute('SELECT COUNT(*) FROM sightings').fetchone()[0])"`
Expected: Shows count > 0

- [ ] **Step 5: Commit**

```bash
git add ghost_listener.py
git commit -m "feat: write to SQLite after each poll cycle for ML pipeline"
```

---

### Task 10: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end test: JSON migration → clean → features → model."""

import json
import os
import tempfile
import pytest
import numpy as np

from ghost_db import GhostDB
from ghost_clean import consolidate_events
from ghost_districts import assign_district_to_events
from ghost_features import build_features
from ghost_model import GhostModel
from datetime import datetime


def _make_test_alerts(n=50):
    """Generate synthetic alerts for testing the full pipeline."""
    import random
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
    """Test: migrate → clean → district assign → feature build."""
    # Create test JSON
    alerts = _make_test_alerts(50)
    data = {"alerts": alerts, "meta": {"total_alerts": 50}}

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as jf:
        json.dump(data, jf)
        json_path = jf.name

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as df:
        db_path = df.name

    try:
        # Step 1: Migrate
        db = GhostDB(db_path)
        count = db.migrate_from_json(json_path)
        assert count == 50

        # Step 2: Clean
        sightings = db.get_unprocessed_sightings()
        assert len(sightings) > 0
        events = consolidate_events(sightings)
        assert len(events) > 0
        assert len(events) <= 50  # consolidation should reduce count

        # Step 3: District assign
        events = assign_district_to_events(events)
        assert all(ev["district"] != "" for ev in events)

        # Step 4: Insert events
        db.insert_events(events)
        stored = db.get_all_events()
        assert len(stored) == len(events)

        # Step 5: Build features
        target_dt = datetime(2026, 6, 13, 15, 0, 0)
        features_df = build_features(stored, target_dt)
        assert len(features_df) > 0
        assert "district" in features_df.columns
        assert "hour" in features_df.columns

        db.close()
    finally:
        os.unlink(json_path)
        os.unlink(db_path)
```

- [ ] **Step 2: Run integration test**

Run: `py -m pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `py -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test for full prediction pipeline"
```

---

### Task 11: Run Migration on Live Data

- [ ] **Step 1: Migrate existing ghost_alerts.json**

Run: `py ghost_predict.py migrate ghost_alerts.json`
Expected: "Migrated X alerts from ghost_alerts.json into ghost_alerts.db"

- [ ] **Step 2: Run event consolidation**

Run: `py ghost_predict.py clean`
Expected: Shows raw sightings count and events after consolidation

- [ ] **Step 3: Check stats**

Run: `py ghost_predict.py stats`
Expected: Shows days collected, sighting count, event count, training gate status

- [ ] **Step 4: Commit database**

```bash
git add ghost_alerts.db
git commit -m "data: initial SQLite migration from ghost_alerts.json"
```
