"""Build a fixed H3 coverage grid for the Hong Kong modeling surface."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import h3
from shapely.geometry import MultiPolygon, Polygon, shape

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghost_zones import DEFAULT_H3_RESOLUTION, assign_zone, h3_zone_polygon


DB_PATH = ROOT / "ghost_alerts.db"
OUTPUT_DIR = ROOT / "analysis" / "geo"
HK_BOUNDARY_PATH = OUTPUT_DIR / "hksar_18_district_boundary.json"

HK_BOUNDS = {
    "lat_min": 22.15,
    "lat_max": 22.58,
    "lng_min": 113.82,
    "lng_max": 114.45,
}

LAT_COLUMNS = ("lat", "latitude")
LNG_COLUMNS = ("lng", "lon", "long", "longitude")


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    value = start
    while value <= stop:
        yield value
        value += step


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def _find_coordinate_columns(columns: Iterable[str]) -> tuple[str | None, str | None]:
    normalized = {column.lower(): column for column in columns}
    lat_col = next((normalized[name] for name in LAT_COLUMNS if name in normalized), None)
    lng_col = next((normalized[name] for name in LNG_COLUMNS if name in normalized), None)
    return lat_col, lng_col


def load_hk_boundary_geometries(
    boundary_path: Path = HK_BOUNDARY_PATH,
) -> list[Polygon | MultiPolygon]:
    """Load official HK district boundary geometries."""
    if not boundary_path.exists():
        return []
    data = json.loads(boundary_path.read_text(encoding="utf-8-sig"))
    return [shape(feature["geometry"]) for feature in data.get("features", [])]


def _ring_to_latlng(ring: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(lat, lng) for lng, lat in ring]


def _polygon_to_h3_cells(polygon: Polygon, resolution: int) -> set[str]:
    outer = _ring_to_latlng(polygon.exterior.coords)
    holes = [_ring_to_latlng(interior.coords) for interior in polygon.interiors]
    return set(h3.polygon_to_cells(h3.LatLngPoly(outer, *holes), resolution))


def official_hk_h3_zones(resolution: int = DEFAULT_H3_RESOLUTION) -> set[str]:
    """Return H3 cells whose centers fall inside official HK district polygons."""
    zones: set[str] = set()
    for geometry in load_hk_boundary_geometries():
        polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
        for polygon in polygons:
            zones.update(_polygon_to_h3_cells(polygon, resolution))
    return zones


def observed_h3_zones(
    db_path: Path = DB_PATH,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> set[str]:
    """Return H3 zones represented by observed latitude/longitude rows."""
    if not db_path.exists():
        return set()

    zones: set[str] = set()
    with sqlite3.connect(db_path) as conn:
        tables = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        ]
        if "events" in tables:
            tables = ["events"]
        for table in tables:
            lat_col, lng_col = _find_coordinate_columns(_table_columns(conn, table))
            if not lat_col or not lng_col:
                continue

            query = (
                f'SELECT "{lat_col}", "{lng_col}" FROM "{table}" '
                f'WHERE "{lat_col}" IS NOT NULL AND "{lng_col}" IS NOT NULL'
            )
            for lat, lng in conn.execute(query):
                try:
                    lat_f = float(lat)
                    lng_f = float(lng)
                except (TypeError, ValueError):
                    continue
                if (
                    HK_BOUNDS["lat_min"] <= lat_f <= HK_BOUNDS["lat_max"]
                    and HK_BOUNDS["lng_min"] <= lng_f <= HK_BOUNDS["lng_max"]
                ):
                    zones.add(assign_zone(lat_f, lng_f, resolution=resolution)["h3_zone"])
    return zones


def sampled_hk_h3_zones(
    resolution: int = DEFAULT_H3_RESOLUTION,
    sample_step_degrees: float = 0.0025,
) -> set[str]:
    """Fill the HK bounding box with H3 cells at the requested resolution."""
    official_zones = official_hk_h3_zones(resolution)
    if official_zones:
        return official_zones

    polygon = h3.LatLngPoly(
        [
            (HK_BOUNDS["lat_min"], HK_BOUNDS["lng_min"]),
            (HK_BOUNDS["lat_min"], HK_BOUNDS["lng_max"]),
            (HK_BOUNDS["lat_max"], HK_BOUNDS["lng_max"]),
            (HK_BOUNDS["lat_max"], HK_BOUNDS["lng_min"]),
        ]
    )
    return set(h3.polygon_to_cells(polygon, resolution))


def build_hk_coverage_grid(
    resolution: int = DEFAULT_H3_RESOLUTION,
    db_path: Path = DB_PATH,
    sample_step_degrees: float = 0.0025,
) -> list[dict[str, object]]:
    """Build fixed HK coverage rows with observed-history flags."""
    observed = observed_h3_zones(db_path=db_path, resolution=resolution)
    zones = sampled_hk_h3_zones(
        resolution=resolution,
        sample_step_degrees=sample_step_degrees,
    ) | observed

    rows: list[dict[str, object]] = []
    for zone_id in sorted(zones):
        zone_lat, zone_lng = h3.cell_to_latlng(zone_id)
        context = assign_zone(zone_lat, zone_lng, resolution=resolution)
        rows.append(
            {
                "h3_zone": zone_id,
                "h3_resolution": resolution,
                "zone_lat": zone_lat,
                "zone_lng": zone_lng,
                "district": context["district"],
                "region": context["region"],
                "has_observed_history": int(zone_id in observed),
                "is_zero_history": int(zone_id not in observed),
                "coverage_source": "observed_and_grid"
                if zone_id in observed
                else "fixed_grid",
            }
        )
    return rows


def write_coverage_artifacts(
    rows: list[dict[str, object]],
    output_dir: Path = OUTPUT_DIR,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"hk_h3_coverage_res{resolution}.csv"
    geojson_path = output_dir / f"hk_h3_coverage_res{resolution}.geojson"

    fieldnames = [
        "h3_zone",
        "h3_resolution",
        "zone_lat",
        "zone_lng",
        "district",
        "region",
        "has_observed_history",
        "is_zero_history",
        "coverage_source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    features = []
    for row in rows:
        zone_id = str(row["h3_zone"])
        features.append(
            {
                "type": "Feature",
                "properties": row,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [h3_zone_polygon(zone_id)],
                },
            }
        )
    with geojson_path.open("w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features}, handle)

    return csv_path, geojson_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=DEFAULT_H3_RESOLUTION)
    parser.add_argument("--sample-step-degrees", type=float, default=0.0025)
    args = parser.parse_args()

    rows = build_hk_coverage_grid(
        resolution=args.resolution,
        sample_step_degrees=args.sample_step_degrees,
    )
    csv_path, geojson_path = write_coverage_artifacts(rows, resolution=args.resolution)
    observed = sum(int(row["has_observed_history"]) for row in rows)
    print(
        f"Wrote {len(rows):,} HK H3 cells ({observed:,} observed-history) "
        f"to {csv_path.relative_to(ROOT)} and {geojson_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
