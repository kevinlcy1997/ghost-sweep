import json
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis import dashboard_service as service


def test_summary_endpoint_reports_fixed_grid_counts():
    status, headers, body = service.dispatch("GET", "/api/summary")

    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    payload = json.loads(body)
    assert payload["h3_resolution"] == 9
    assert payload["coverage_mode"] == "road_access"
    assert 175 <= payload["average_hex_edge_m"] <= 225
    assert payload["coverage_cells"] > 1000
    assert payload["observed_cells"] > 0
    assert payload["zero_history_cells"] > payload["observed_cells"]
    assert payload["event_count"] > 0
    assert payload["artifact_groups"] >= 1


def test_coverage_endpoint_filters_and_paginates():
    status, _, body = service.dispatch(
        "GET",
        "/api/coverage?region=Kowloon%20West&min_events=1&limit=5&offset=0",
    )

    assert status == 200
    payload = json.loads(body)
    assert payload["limit"] == 5
    assert payload["offset"] == 0
    assert len(payload["rows"]) <= 5
    assert payload["total"] >= len(payload["rows"])
    assert all(row["region"] == "Kowloon West" for row in payload["rows"])
    assert all(row["event_count"] >= 1 for row in payload["rows"])
    assert all("has_drivable_road" in row for row in payload["rows"])


def test_timeseries_endpoint_returns_hour_and_day_profiles():
    hour_status, _, hour_body = service.dispatch("GET", "/api/timeseries?grain=hour")
    day_status, _, day_body = service.dispatch("GET", "/api/timeseries?grain=day")

    assert hour_status == 200
    assert day_status == 200
    assert len(json.loads(hour_body)["rows"]) == 24
    assert len(json.loads(day_body)["rows"]) == 7
    assert sum(row["event_count"] for row in json.loads(hour_body)["rows"]) > 0


def test_predictions_endpoint_supports_horizons():
    status, _, body = service.dispatch("GET", "/api/predictions?horizon=30m&limit=10")

    assert status == 200
    payload = json.loads(body)
    assert payload["horizon"] == "30m"
    assert payload["limit"] == 10
    assert "rows" in payload


def test_artifacts_endpoint_flattens_manifest_groups():
    status, _, body = service.dispatch("GET", "/api/artifacts")

    assert status == 200
    payload = json.loads(body)
    assert any(row["group"] == "coverage_grid" for row in payload["rows"])
    assert all("path" in row for row in payload["rows"])


def test_model_metrics_endpoint_returns_horizon_rows_and_metadata():
    status, headers, body = service.dispatch("GET", "/api/model-metrics")

    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    payload = json.loads(body)
    assert payload["total"] >= 3
    horizons = {row["horizon"] for row in payload["rows"]}
    assert {"30m", "1h", "2h"}.issubset(horizons)
    row = next(item for item in payload["rows"] if item["horizon"] == "30m")
    assert row["chosen_model"] == "lightgbm_conservative"
    assert row["metadata"]["horizon_slug"] == "30m"
    assert row["metadata"]["target_col"] == "alert_next_30m"
    assert row["metadata_path"].endswith("best_iterated_model_metadata_30m.json")


def test_grid_geojson_endpoint_returns_real_overlay_features():
    status, headers, body = service.dispatch("GET", "/api/grid.geojson?min_events=1")

    assert status == 200
    assert headers["Content-Type"] == "application/geo+json; charset=utf-8"
    payload = json.loads(body)
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) > 0
    first = payload["features"][0]
    assert first["geometry"]["type"] == "Polygon"
    assert "event_count" in first["properties"]
    assert first["properties"]["h3_resolution"] == 9
    assert first["properties"]["event_count"] >= 1
    assert "has_drivable_road" in first["properties"]
    assert "road_source_mismatch" in first["properties"]


def test_grid_geojson_merges_horizon_probability_properties():
    zone_id = service.read_csv_rows(service.PATHS["coverage"])[0]["h3_zone"]
    with TemporaryDirectory() as tmpdir:
        pred_path = Path(tmpdir) / "predictions.csv"
        pred_path.write_text(
            "\n".join(
                [
                    "target_time,zone_id,district,region,zone_lat,zone_lng,score,probability,risk_band,rank,actual",
                    f"2026-06-30 01:00:00,{zone_id},Southern,Hong Kong Island,22.0,114.0,0.72,0.83,high,7,1",
                ]
            ),
            encoding="utf-8",
        )
        original = service.PATHS["predictions_30m"]
        service.PATHS["predictions_30m"] = pred_path
        try:
            status, _, body = service.dispatch("GET", "/api/grid.geojson?horizon=30m&min_events=0")
        finally:
            service.PATHS["predictions_30m"] = original

    assert status == 200
    payload = json.loads(body)
    feature = next(item for item in payload["features"] if item["properties"]["h3_zone"] == zone_id)
    assert feature["properties"]["probability"] == 0.83
    assert feature["properties"]["risk_band"] == "high"
    assert feature["properties"]["rank"] == 7
    assert feature["properties"]["score"] == 0.72
    assert feature["properties"]["zone_id"] == zone_id


def test_grid_geojson_zero_history_overlay_is_not_silently_truncated():
    _, _, summary_body = service.dispatch("GET", "/api/summary")
    summary = json.loads(summary_body)
    status, _, body = service.dispatch("GET", "/api/grid.geojson?min_events=0")

    assert status == 200
    payload = json.loads(body)
    assert payload["total"] == summary["coverage_cells"]
    assert len(payload["features"]) == payload["total"]
    assert any(feature["properties"]["is_zero_history"] == 1 for feature in payload["features"])


def test_dashboard_html_fetches_api_instead_of_embedding_payload():
    status, headers, body = service.dispatch("GET", "/")

    assert status == 200
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert "Ghost Sweep Service Dashboard" in body
    assert "edgeScale" in body
    assert "leaflet" in body.lower()
    assert "L.tileLayer" in body
    assert "L.geoJSON" in body
    assert "fetchJson('/api/summary')" in body
    assert "fetchJson('/api/model-metrics')" in body
    assert "fetchJson('/api/grid.geojson?horizon='" in body
    assert "modelMetrics" in body
    assert "const DATA =" not in body
