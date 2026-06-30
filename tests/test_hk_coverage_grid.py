from pathlib import Path

from shapely.geometry import Point
from shapely.ops import unary_union

from analysis.build_hk_coverage_grid import (
    build_hk_coverage_grid,
    load_hk_boundary_geometries,
    observed_h3_zones,
    write_coverage_artifacts,
)
from ghost_zones import assign_zone


def test_fixed_grid_includes_known_hk_cell_and_zero_history_cells():
    rows = build_hk_coverage_grid(sample_step_degrees=0.05)
    zone_ids = {row["h3_zone"] for row in rows}

    assert assign_zone(22.3154, 114.1698)["h3_zone"] in zone_ids
    assert any(row["has_observed_history"] == 1 for row in rows)
    assert any(row["is_zero_history"] == 1 for row in rows)


def test_observed_zones_are_subset_of_fixed_grid():
    observed = observed_h3_zones()
    rows = build_hk_coverage_grid(sample_step_degrees=0.05)
    fixed = {row["h3_zone"] for row in rows}

    assert observed
    assert observed <= fixed


def test_fixed_grid_cells_are_inside_official_hk_boundary():
    rows = build_hk_coverage_grid()
    boundary = unary_union(load_hk_boundary_geometries())

    outside = [
        row
        for row in rows
        if not boundary.covers(Point(float(row["zone_lng"]), float(row["zone_lat"])))
    ]

    assert outside == []


def test_write_coverage_artifacts(tmp_path: Path):
    rows = build_hk_coverage_grid(sample_step_degrees=0.1)
    csv_path, geojson_path = write_coverage_artifacts(rows, tmp_path)

    assert csv_path.exists()
    assert geojson_path.exists()
    assert csv_path.read_text(encoding="utf-8").startswith("h3_zone,h3_resolution")
    assert '"FeatureCollection"' in geojson_path.read_text(encoding="utf-8")
