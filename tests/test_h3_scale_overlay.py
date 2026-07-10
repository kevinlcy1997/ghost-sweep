from analysis.make_h3_scale_overlay import build_overlay_html


def test_build_overlay_html_includes_toggle_and_layers():
    datasets = {
        7: {"cell_count": 2, "edge_m": 1406.4, "polygons": [[[114.0, 22.2], [114.1, 22.2], [114.1, 22.3], [114.0, 22.2]]]},
        8: {"cell_count": 3, "edge_m": 531.4, "polygons": [[[114.0, 22.2], [114.05, 22.2], [114.05, 22.25], [114.0, 22.2]]]},
        9: {"cell_count": 4, "edge_m": 200.8, "polygons": [[[114.0, 22.2], [114.02, 22.2], [114.02, 22.22], [114.0, 22.2]]]},
    }
    html = build_overlay_html(datasets, (113.8, 114.5, 22.1, 22.6))
    assert "HK H3 Grid Scale" in html
    assert "H3 res 7" in html
    assert 'id="res-8"' in html
    assert 'setResolution' in html
