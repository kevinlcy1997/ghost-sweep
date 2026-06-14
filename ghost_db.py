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
                float(rec.get("lat", 0)),
                float(rec.get("lng", 0)),
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
        """Get sightings that pass cleaning filters."""
        rows = self._conn.execute("""
            SELECT * FROM sightings
            WHERE lat != 0 AND lng != 0
              AND lat BETWEEN 22.15 AND 22.56
              AND lng BETWEEN 113.83 AND 114.41
              AND NOT (upvote = 0 AND downvote >= 3)
            ORDER BY create_dt
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

    def migrate_from_json(self, json_path: str) -> int:
        """Import alerts from ghost_alerts.json into sightings table."""
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

    @staticmethod
    def _compute_grid_cell(lat: float, lng: float) -> str:
        from ghost_utils import compute_grid_cell
        return compute_grid_cell(lat, lng)

    def close(self):
        self._conn.close()
