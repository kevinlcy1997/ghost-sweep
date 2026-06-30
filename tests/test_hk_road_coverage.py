from pathlib import Path

from analysis.build_hk_road_coverage_grid import (
    build_road_coverage_rows,
    drivable_road_segments_from_geojson,
    write_road_coverage_artifacts,
)


def test_drivable_road_segments_exclude_non_car_paths():
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"highway": "residential", "name": "Car Street"},
                "geometry": {"type": "LineString", "coordinates": [[114.17, 22.31], [114.18, 22.31]]},
            },
            {
                "type": "Feature",
                "properties": {"highway": "footway", "name": "Walking Path"},
                "geometry": {"type": "LineString", "coordinates": [[114.17, 22.32], [114.18, 22.32]]},
            },
        ],
    }

    segments = drivable_road_segments_from_geojson(geojson)

    assert len(segments) == 1
    assert segments[0]["road_class"] == "residential"


def test_road_coverage_keeps_near_road_and_observed_mismatch_cells():
    grid = [
        {
            "h3_zone": "near",
            "h3_resolution": 9,
            "zone_lat": 22.31,
            "zone_lng": 114.171,
            "district": "Mong Kok",
            "region": "Kowloon West",
            "has_observed_history": 0,
            "is_zero_history": 1,
        },
        {
            "h3_zone": "far",
            "h3_resolution": 9,
            "zone_lat": 22.40,
            "zone_lng": 114.30,
            "district": "Sha Tin",
            "region": "New Territories South",
            "has_observed_history": 0,
            "is_zero_history": 1,
        },
        {
            "h3_zone": "observed-miss",
            "h3_resolution": 9,
            "zone_lat": 22.42,
            "zone_lng": 114.32,
            "district": "Sha Tin",
            "region": "New Territories South",
            "has_observed_history": 1,
            "is_zero_history": 0,
        },
    ]
    roads = [
        {
            "road_class": "residential",
            "name": "Car Street",
            "coordinates": [(22.31, 114.17), (22.31, 114.18)],
        }
    ]

    rows = build_road_coverage_rows(grid, roads, buffer_m=60)

    assert {row["h3_zone"] for row in rows} == {"near", "observed-miss"}
    near = next(row for row in rows if row["h3_zone"] == "near")
    observed_miss = next(row for row in rows if row["h3_zone"] == "observed-miss")
    assert near["has_drivable_road"] == 1
    assert near["road_source_mismatch"] == 0
    assert observed_miss["has_drivable_road"] == 0
    assert observed_miss["road_source_mismatch"] == 1
    assert observed_miss["coverage_source"] == "observed_without_road_match"


def test_write_road_coverage_artifacts(tmp_path: Path):
    rows = [
        {
            "h3_zone": "89411cb3693ffff",
            "h3_resolution": 9,
            "zone_lat": 22.31,
            "zone_lng": 114.17,
            "district": "Mong Kok",
            "region": "Kowloon West",
            "has_observed_history": 1,
            "is_zero_history": 0,
            "has_drivable_road": 1,
            "road_source_mismatch": 0,
            "nearest_road_m": 0.0,
            "road_segment_count": 1,
            "road_classes": "residential",
            "nearest_road_name": "Car Street",
            "coverage_source": "road_access",
        }
    ]

    csv_path, geojson_path = write_road_coverage_artifacts(rows, tmp_path, resolution=9)

    assert csv_path.exists()
    assert geojson_path.exists()
    assert "has_drivable_road" in csv_path.read_text(encoding="utf-8")
    assert '"FeatureCollection"' in geojson_path.read_text(encoding="utf-8")
