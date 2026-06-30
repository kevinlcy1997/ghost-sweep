import h3

from ghost_zones import DEFAULT_H3_RESOLUTION, assign_zone, compute_h3_zone, h3_zone_polygon


def test_default_resolution_is_200m_class_for_hong_kong():
    assert DEFAULT_H3_RESOLUTION == 9
    assert 175 <= h3.average_hexagon_edge_length(DEFAULT_H3_RESOLUTION, "m") <= 225


def test_assign_zone_includes_grid_h3_and_district_context():
    zone = assign_zone(22.3154, 114.1698)

    assert zone["grid_cell"] == "22.315_114.170"
    assert zone["h3_zone"] == compute_h3_zone(22.3154, 114.1698)
    assert zone["h3_resolution"] == 9
    assert zone["district"] == "Yau Tsim Mong"
    assert zone["region"] == "Kowloon West"
    assert isinstance(zone["zone_lat"], float)
    assert isinstance(zone["zone_lng"], float)


def test_h3_zone_polygon_is_closed_geojson_ring():
    ring = h3_zone_polygon(compute_h3_zone(22.3154, 114.1698))

    assert len(ring) >= 7
    assert ring[0] == ring[-1]
    assert all(len(point) == 2 for point in ring)
def test_compute_h3_zone_accepts_resolution():
    res8 = compute_h3_zone(22.3154, 114.1698, resolution=8)
    res9 = compute_h3_zone(22.3154, 114.1698, resolution=9)

    assert res8 != res9
    assert len(res9) >= len(res8)


def test_assign_zone_accepts_resolution():
    row = assign_zone(22.3154, 114.1698, resolution=9)

    assert row["h3_resolution"] == 9
    assert row["h3_zone"] == compute_h3_zone(22.3154, 114.1698, resolution=9)
