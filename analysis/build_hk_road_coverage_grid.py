"""Build a road-accessible H3 coverage grid for Hong Kong."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

from h3 import H3CellInvalidError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_hk_coverage_grid import HK_BOUNDS, OUTPUT_DIR as GEO_DIR
from ghost_zones import DEFAULT_H3_RESOLUTION, h3_zone_polygon


OUTPUT_DIR = ROOT / "analysis" / "geo"
ROAD_GEOJSON_PATH = OUTPUT_DIR / "hk_drivable_roads.geojson"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

DRIVABLE_HIGHWAY_CLASSES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}

NON_DRIVABLE_HIGHWAY_CLASSES = {
    "footway",
    "path",
    "cycleway",
    "steps",
    "pedestrian",
    "bridleway",
    "corridor",
    "elevator",
    "escalator",
    "track",
    "bus_stop",
    "platform",
}


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def is_drivable_highway(value: object) -> bool:
    classes = set(_as_list(value))
    return bool(classes & DRIVABLE_HIGHWAY_CLASSES) and not bool(
        classes & NON_DRIVABLE_HIGHWAY_CLASSES
    )


def _line_coordinates(geometry: dict) -> Iterable[list[list[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "LineString":
        yield coordinates
    elif geometry_type == "MultiLineString":
        yield from coordinates


def drivable_road_segments_from_geojson(geojson: dict) -> list[dict[str, object]]:
    """Extract drivable road segments from GeoJSON LineString features."""
    segments: list[dict[str, object]] = []
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {}) or {}
        highway = properties.get("highway")
        if not is_drivable_highway(highway):
            continue
        road_class = _as_list(highway)[0]
        name = str(properties.get("name") or properties.get("ref") or "")
        for line in _line_coordinates(feature.get("geometry", {}) or {}):
            points = [
                (float(lat), float(lng))
                for lng, lat in line
                if isinstance(lng, (int, float)) and isinstance(lat, (int, float))
            ]
            for start, end in zip(points, points[1:]):
                segments.append(
                    {
                        "road_class": road_class,
                        "name": name,
                        "coordinates": [start, end],
                    }
                )
    return segments


def load_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_geojson(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _meters_per_lng_degree(lat: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat))


def _project_around(lat: float, lng: float, ref_lat: float) -> tuple[float, float]:
    return lng * _meters_per_lng_degree(ref_lat), lat * 110_574.0


def point_to_segment_distance_m(
    lat: float,
    lng: float,
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Approximate distance from a lat/lng point to a segment in meters."""
    ref_lat = lat
    px, py = _project_around(lat, lng, ref_lat)
    ax, ay = _project_around(start[0], start[1], ref_lat)
    bx, by = _project_around(end[0], end[1], ref_lat)
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def _road_index_key(lat: float, lng: float, bin_degrees: float) -> tuple[int, int]:
    return math.floor(lat / bin_degrees), math.floor(lng / bin_degrees)


def build_road_spatial_index(
    roads: list[dict[str, object]],
    bin_degrees: float = 0.01,
) -> dict[tuple[int, int], list[dict[str, object]]]:
    index: dict[tuple[int, int], list[dict[str, object]]] = {}
    for segment in roads:
        coordinates = segment["coordinates"]
        (lat_a, lng_a), (lat_b, lng_b) = coordinates
        lat_min, lat_max = sorted([lat_a, lat_b])
        lng_min, lng_max = sorted([lng_a, lng_b])
        lat_start, lng_start = _road_index_key(lat_min, lng_min, bin_degrees)
        lat_end, lng_end = _road_index_key(lat_max, lng_max, bin_degrees)
        for lat_key in range(lat_start, lat_end + 1):
            for lng_key in range(lng_start, lng_end + 1):
                index.setdefault((lat_key, lng_key), []).append(segment)
    return index


def nearby_segments(
    index: dict[tuple[int, int], list[dict[str, object]]],
    lat: float,
    lng: float,
    bin_degrees: float = 0.01,
    radius_bins: int = 1,
) -> list[dict[str, object]]:
    lat_key, lng_key = _road_index_key(lat, lng, bin_degrees)
    candidates: list[dict[str, object]] = []
    seen: set[int] = set()
    for d_lat in range(-radius_bins, radius_bins + 1):
        for d_lng in range(-radius_bins, radius_bins + 1):
            for segment in index.get((lat_key + d_lat, lng_key + d_lng), []):
                marker = id(segment)
                if marker not in seen:
                    seen.add(marker)
                    candidates.append(segment)
    return candidates


def road_match_for_cell(
    lat: float,
    lng: float,
    road_index: dict[tuple[int, int], list[dict[str, object]]],
    buffer_m: float,
) -> dict[str, object]:
    candidates = nearby_segments(road_index, lat, lng, radius_bins=1)
    if not candidates:
        candidates = nearby_segments(road_index, lat, lng, radius_bins=3)

    nearest_distance = math.inf
    nearest_name = ""
    nearby_classes: set[str] = set()
    nearby_count = 0
    for segment in candidates:
        start, end = segment["coordinates"]
        distance = point_to_segment_distance_m(lat, lng, start, end)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_name = str(segment.get("name", ""))
        if distance <= buffer_m:
            nearby_count += 1
            nearby_classes.add(str(segment.get("road_class", "")))

    return {
        "has_drivable_road": int(nearby_count > 0),
        "nearest_road_m": round(nearest_distance, 1) if math.isfinite(nearest_distance) else "",
        "road_segment_count": nearby_count,
        "road_classes": "|".join(sorted(item for item in nearby_classes if item)),
        "nearest_road_name": nearest_name,
    }


def build_road_coverage_rows(
    grid_rows: list[dict[str, object]],
    roads: list[dict[str, object]],
    buffer_m: float = 60.0,
) -> list[dict[str, object]]:
    road_index = build_road_spatial_index(roads)
    rows: list[dict[str, object]] = []
    for cell in grid_rows:
        lat = float(cell["zone_lat"])
        lng = float(cell["zone_lng"])
        road_match = road_match_for_cell(lat, lng, road_index, buffer_m)
        has_observed_history = int(float(cell.get("has_observed_history", 0)))
        has_drivable_road = int(road_match["has_drivable_road"])
        if not has_drivable_road and not has_observed_history:
            continue

        row = dict(cell)
        row.update(road_match)
        row["has_observed_history"] = has_observed_history
        row["is_zero_history"] = int(not has_observed_history)
        row["road_source_mismatch"] = int(has_observed_history and not has_drivable_road)
        row["coverage_source"] = (
            "road_access" if has_drivable_road else "observed_without_road_match"
        )
        rows.append(row)
    return rows


def write_road_coverage_artifacts(
    rows: list[dict[str, object]],
    output_dir: Path = OUTPUT_DIR,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"hk_h3_road_coverage_res{resolution}.csv"
    geojson_path = output_dir / f"hk_h3_road_coverage_res{resolution}.geojson"

    fieldnames = [
        "h3_zone",
        "h3_resolution",
        "zone_lat",
        "zone_lng",
        "district",
        "region",
        "has_observed_history",
        "is_zero_history",
        "has_drivable_road",
        "road_source_mismatch",
        "nearest_road_m",
        "road_segment_count",
        "road_classes",
        "nearest_road_name",
        "coverage_source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    features = []
    for row in rows:
        try:
            geometry = {"type": "Polygon", "coordinates": [h3_zone_polygon(str(row["h3_zone"]))]}
        except H3CellInvalidError:
            geometry = {
                "type": "Point",
                "coordinates": [float(row["zone_lng"]), float(row["zone_lat"])],
            }
        features.append(
            {
                "type": "Feature",
                "properties": row,
                "geometry": geometry,
            }
        )
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    return csv_path, geojson_path


def overpass_query() -> str:
    lat_min = HK_BOUNDS["lat_min"]
    lat_max = HK_BOUNDS["lat_max"]
    lng_min = HK_BOUNDS["lng_min"]
    lng_max = HK_BOUNDS["lng_max"]
    highway_pattern = "|".join(sorted(DRIVABLE_HIGHWAY_CLASSES))
    return f"""
[out:json][timeout:180];
(
  way["highway"~"^({highway_pattern})$"]({lat_min},{lng_min},{lat_max},{lng_max});
);
out geom;
"""


def fetch_osm_roads_geojson(path: Path = ROAD_GEOJSON_PATH) -> Path:
    payload = urllib.parse.urlencode({"data": overpass_query()}).encode("utf-8")
    last_error: Exception | None = None
    data = None
    for url in OVERPASS_URLS:
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "ghost-sweep-road-coverage/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                data = json.loads(response.read().decode("utf-8"))
                break
        except Exception as exc:  # pragma: no cover - exercised only against live services
            last_error = exc
    if data is None:
        raise RuntimeError(f"Could not fetch OSM roads from Overpass: {last_error}")

    features = []
    for element in data.get("elements", []):
        geometry = element.get("geometry", [])
        if element.get("type") != "way" or len(geometry) < 2:
            continue
        tags = element.get("tags", {})
        coordinates = [[point["lon"], point["lat"]] for point in geometry]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "osm_id": element.get("id"),
                    "highway": tags.get("highway"),
                    "name": tags.get("name", ""),
                    "ref": tags.get("ref", ""),
                },
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    return path


def build_road_coverage_artifacts(
    resolution: int = DEFAULT_H3_RESOLUTION,
    buffer_m: float = 60.0,
    roads_path: Path = ROAD_GEOJSON_PATH,
    fetch_roads: bool = False,
) -> tuple[Path, Path, int]:
    grid_path = GEO_DIR / f"hk_h3_coverage_res{resolution}.csv"
    if fetch_roads or not roads_path.exists():
        fetch_osm_roads_geojson(roads_path)
    roads = drivable_road_segments_from_geojson(load_geojson(roads_path))
    rows = build_road_coverage_rows(load_csv_rows(grid_path), roads, buffer_m=buffer_m)
    csv_path, geojson_path = write_road_coverage_artifacts(rows, resolution=resolution)
    return csv_path, geojson_path, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=DEFAULT_H3_RESOLUTION)
    parser.add_argument("--buffer-m", type=float, default=60.0)
    parser.add_argument("--roads-path", type=Path, default=ROAD_GEOJSON_PATH)
    parser.add_argument("--fetch-roads", action="store_true")
    args = parser.parse_args()

    csv_path, geojson_path, row_count = build_road_coverage_artifacts(
        resolution=args.resolution,
        buffer_m=args.buffer_m,
        roads_path=args.roads_path,
        fetch_roads=args.fetch_roads,
    )
    print(
        f"Wrote {row_count:,} road-access H3 cells to "
        f"{csv_path.relative_to(ROOT)} and {geojson_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
