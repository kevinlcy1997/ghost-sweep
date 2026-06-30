"""Create fixed-grid analysis marts for HK zone discovery and model review."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_hk_coverage_grid import DB_PATH, OUTPUT_DIR as GEO_DIR
from ghost_time import to_hk_feature_time
from ghost_zones import DEFAULT_H3_RESOLUTION, assign_zone


OUTPUT_DIR = ROOT / "analysis" / "data_discovery"
TIMESTAMP_COLUMNS = (
    "create_dt",
    "created_at",
    "timestamp",
    "event_time",
    "time",
    "datetime",
    "sighted_at",
    "first_seen",
    "last_seen",
    "end_dt",
)
LAT_COLUMNS = ("lat", "latitude")
LNG_COLUMNS = ("lng", "lon", "long", "longitude")


def parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return to_hk_feature_time(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return to_hk_feature_time(datetime.strptime(text[: len(fmt)], fmt))
        except ValueError:
            pass
    return None


def _find_named_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {column.lower(): column for column in columns}
    return next((normalized[name] for name in candidates if name in normalized), None)


def load_coverage_grid(
    resolution: int = DEFAULT_H3_RESOLUTION,
    path: Path | None = None,
) -> list[dict[str, str]]:
    grid_path = path or GEO_DIR / f"hk_h3_coverage_res{resolution}.csv"
    with grid_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_observed_events(
    db_path: Path = DB_PATH,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    if not db_path.exists():
        return events

    with sqlite3.connect(db_path) as conn:
        tables = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        ]
        if "events" in tables:
            tables = ["events"]
        for table in tables:
            columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
            lat_col = _find_named_column(columns, LAT_COLUMNS)
            lng_col = _find_named_column(columns, LNG_COLUMNS)
            ts_col = _find_named_column(columns, TIMESTAMP_COLUMNS)
            if not lat_col or not lng_col or not ts_col:
                continue

            query = (
                f'SELECT "{lat_col}", "{lng_col}", "{ts_col}" FROM "{table}" '
                f'WHERE "{lat_col}" IS NOT NULL AND "{lng_col}" IS NOT NULL'
            )
            for lat, lng, timestamp in conn.execute(query):
                try:
                    lat_f = float(lat)
                    lng_f = float(lng)
                except (TypeError, ValueError):
                    continue
                parsed_time = parse_timestamp(timestamp)
                if parsed_time is None:
                    continue
                zone = assign_zone(lat_f, lng_f, resolution=resolution)
                events.append(
                    {
                        "h3_zone": zone["h3_zone"],
                        "district": zone["district"],
                        "region": zone["region"],
                        "event_time": parsed_time,
                        "hour": parsed_time.hour,
                        "day_of_week": parsed_time.weekday(),
                    }
                )
    return events


def build_zone_sparsity_profile(
    grid: list[dict[str, object]],
    events: list[dict[str, object]],
) -> list[dict[str, object]]:
    event_counts = Counter(str(event["h3_zone"]) for event in events)
    district_counts = Counter(str(event["district"]) for event in events)
    region_counts = Counter(str(event["region"]) for event in events)
    rows = []
    for cell in grid:
        zone_id = str(cell["h3_zone"])
        district = str(cell.get("district", "Unknown"))
        region = str(cell.get("region", "Unknown"))
        event_count = event_counts[zone_id]
        rows.append(
            {
                "h3_zone": zone_id,
                "district": district,
                "region": region,
                "event_count": event_count,
                "log_event_count": round(__import__("math").log1p(event_count), 6),
                "has_observed_history": int(event_count > 0),
                "is_zero_history": int(event_count == 0),
                "district_event_count": district_counts[district],
                "region_event_count": region_counts[region],
            }
        )
    return rows


def build_zone_hour_profile(
    grid: list[dict[str, object]],
    events: list[dict[str, object]],
) -> list[dict[str, object]]:
    counts = Counter((str(event["h3_zone"]), int(event["hour"])) for event in events)
    total_by_zone = Counter(str(event["h3_zone"]) for event in events)
    rows = []
    for cell in grid:
        zone_id = str(cell["h3_zone"])
        for hour in range(24):
            count = counts[(zone_id, hour)]
            total = total_by_zone[zone_id]
            rows.append(
                {
                    "h3_zone": zone_id,
                    "hour": hour,
                    "event_count": count,
                    "zone_hour_share": round(count / total, 6) if total else 0.0,
                }
            )
    return rows


def build_zone_daily_profile(
    grid: list[dict[str, object]],
    events: list[dict[str, object]],
) -> list[dict[str, object]]:
    counts = Counter(
        (str(event["h3_zone"]), int(event["day_of_week"])) for event in events
    )
    total_by_zone = Counter(str(event["h3_zone"]) for event in events)
    rows = []
    for cell in grid:
        zone_id = str(cell["h3_zone"])
        for day in range(7):
            count = counts[(zone_id, day)]
            total = total_by_zone[zone_id]
            rows.append(
                {
                    "h3_zone": zone_id,
                    "day_of_week": day,
                    "event_count": count,
                    "zone_day_share": round(count / total, 6) if total else 0.0,
                }
            )
    return rows


def build_zone_recency_profile(
    grid: list[dict[str, object]],
    events: list[dict[str, object]],
) -> list[dict[str, object]]:
    latest_by_zone: dict[str, datetime] = {}
    if events:
        as_of = max(event["event_time"] for event in events)
    else:
        as_of = datetime.now(timezone.utc).replace(tzinfo=None)
    for event in events:
        zone_id = str(event["h3_zone"])
        event_time = event["event_time"]
        if zone_id not in latest_by_zone or event_time > latest_by_zone[zone_id]:
            latest_by_zone[zone_id] = event_time

    rows = []
    for cell in grid:
        zone_id = str(cell["h3_zone"])
        latest = latest_by_zone.get(zone_id)
        days_since = (as_of - latest).total_seconds() / 86400 if latest else None
        rows.append(
            {
                "h3_zone": zone_id,
                "last_event_time": latest.isoformat(sep=" ") if latest else "",
                "days_since_last_event": round(days_since, 3) if days_since is not None else "",
                "has_recent_7d": int(days_since is not None and days_since <= 7),
                "has_recent_30d": int(days_since is not None and days_since <= 30),
            }
        )
    return rows


def build_zone_neighbor_context(
    grid: list[dict[str, object]],
    events: list[dict[str, object]],
) -> list[dict[str, object]]:
    fixed_zones = {str(row["h3_zone"]) for row in grid}
    event_counts = Counter(str(event["h3_zone"]) for event in events)
    rows = []
    for cell in grid:
        zone_id = str(cell["h3_zone"])
        neighbors = set(h3.grid_disk(zone_id, 1)) - {zone_id}
        fixed_neighbors = neighbors & fixed_zones
        observed_neighbor_count = sum(1 for zone in fixed_neighbors if event_counts[zone] > 0)
        neighbor_event_count = sum(event_counts[zone] for zone in fixed_neighbors)
        rows.append(
            {
                "h3_zone": zone_id,
                "fixed_neighbor_count": len(fixed_neighbors),
                "observed_neighbor_count": observed_neighbor_count,
                "neighbor_event_count": neighbor_event_count,
                "neighbor_observed_share": round(
                    observed_neighbor_count / len(fixed_neighbors), 6
                )
                if fixed_neighbors
                else 0.0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_feature_marts(resolution: int = DEFAULT_H3_RESOLUTION) -> dict[str, Path]:
    return build_feature_marts_for_grid(resolution=resolution)


def build_feature_marts_for_grid(
    resolution: int = DEFAULT_H3_RESOLUTION,
    grid_path: Path | None = None,
    output_suffix: str = "",
) -> dict[str, Path]:
    grid = load_coverage_grid(resolution=resolution, path=grid_path)
    events = load_observed_events(resolution=resolution)
    suffix = f"_{output_suffix}" if output_suffix else ""
    outputs = {
        "zone_sparsity_profile": OUTPUT_DIR / f"zone_sparsity_profile{suffix}_res{resolution}.csv",
        "zone_hour_profile": OUTPUT_DIR / f"zone_hour_profile{suffix}_res{resolution}.csv",
        "zone_daily_profile": OUTPUT_DIR / f"zone_daily_profile{suffix}_res{resolution}.csv",
        "zone_recency_profile": OUTPUT_DIR / f"zone_recency_profile{suffix}_res{resolution}.csv",
        "zone_neighbor_context": OUTPUT_DIR / f"zone_neighbor_context{suffix}_res{resolution}.csv",
    }
    write_csv(outputs["zone_sparsity_profile"], build_zone_sparsity_profile(grid, events))
    write_csv(outputs["zone_hour_profile"], build_zone_hour_profile(grid, events))
    write_csv(outputs["zone_daily_profile"], build_zone_daily_profile(grid, events))
    write_csv(outputs["zone_recency_profile"], build_zone_recency_profile(grid, events))
    write_csv(outputs["zone_neighbor_context"], build_zone_neighbor_context(grid, events))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=DEFAULT_H3_RESOLUTION)
    parser.add_argument("--grid-path", type=Path)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    outputs = build_feature_marts_for_grid(
        resolution=args.resolution,
        grid_path=args.grid_path,
        output_suffix=args.output_suffix,
    )
    print("Wrote fixed-grid feature marts:")
    for name, path in outputs.items():
        print(f"- {name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
