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
    original = service.PATHS["two_stage_summary"]
    service.PATHS["two_stage_summary"] = Path("__missing_two_stage_summary__.csv")
    service._read_csv_cached.cache_clear()
    service._read_json_cached.cache_clear()
    try:
        status, headers, body = service.dispatch("GET", "/api/model-metrics")
    finally:
        service.PATHS["two_stage_summary"] = original
        service._read_csv_cached.cache_clear()
        service._read_json_cached.cache_clear()

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


def test_model_metrics_endpoint_prefers_two_stage_summary(tmp_path):
    activity_metadata = tmp_path / "activity_model_metadata_30m.json"
    spatial_metadata = tmp_path / "spatial_model_metadata_30m.json"
    activity_metadata.write_text(
        json.dumps({"stage": "activity", "horizon_slug": "30m", "holdout_split": {"holdout_positives": 4}}),
        encoding="utf-8",
    )
    spatial_metadata.write_text(
        json.dumps({"stage": "spatial", "horizon_slug": "30m", "holdout_split": {"holdout_positives": 7}}),
        encoding="utf-8",
    )
    summary = tmp_path / "two_stage_summary_latest.csv"
    summary.write_text(
        "\n".join(
            [
                "horizon_minutes,horizon,model_family,activity_model,spatial_model,activity_average_precision,activity_roc_auc,activity_brier_score,activity_holdout_rows,activity_holdout_positives,activity_holdout_start,activity_holdout_end,spatial_precision_at_20,spatial_precision_at_50,spatial_average_precision,spatial_top_decile_lift,spatial_holdout_rows,spatial_holdout_positives,spatial_holdout_start,spatial_holdout_end,activity_metadata_path,spatial_metadata_path,activity_predictions_path,predictions_path",
                f"30,30m,two_stage,logistic_balanced,lightgbm_conservative,0.44,0.71,0.12,12,4,2026-06-01,2026-06-02,0.2,0.18,0.11,3.5,30,7,2026-06-01,2026-06-02,{activity_metadata},{spatial_metadata},analysis/activity_predictions_30m_latest.csv,analysis/spatial_zone_predictions_30m_latest.csv",
            ]
        ),
        encoding="utf-8",
    )
    original = dict(service.PATHS)
    service.PATHS["two_stage_summary"] = summary
    service._read_csv_cached.cache_clear()
    service._read_json_cached.cache_clear()
    try:
        status, _, body = service.dispatch("GET", "/api/model-metrics")
    finally:
        service.PATHS.clear()
        service.PATHS.update(original)
        service._read_csv_cached.cache_clear()
        service._read_json_cached.cache_clear()

    assert status == 200
    payload = json.loads(body)
    assert payload["total"] == 1
    row = payload["rows"][0]
    assert row["model_family"] == "two_stage"
    assert row["activity_model"] == "logistic_balanced"
    assert row["spatial_model"] == "lightgbm_conservative"
    assert row["activity_metadata"]["stage"] == "activity"
    assert row["spatial_metadata"]["stage"] == "spatial"
    assert row["activity_holdout_positives"] == 4
    assert row["spatial_holdout_positives"] == 7


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


def test_predictions_endpoint_prefers_two_stage_spatial_predictions(tmp_path):
    predictions = tmp_path / "spatial_zone_predictions_30m_latest.csv"
    predictions.write_text(
        "\n".join(
            [
                "target_time,zone_id,district,region,zone_lat,zone_lng,spatial_probability,activity_probability,probability,score,risk_band,rank,actual",
                "2026-06-30 01:00:00,abc,Central,Hong Kong Island,22.28,114.16,0.80,0.50,0.40,0.40,elevated,1,1",
            ]
        ),
        encoding="utf-8",
    )
    original = dict(service.PATHS)
    service.PATHS["spatial_predictions_30m"] = predictions
    service._read_csv_cached.cache_clear()
    try:
        status, _, body = service.dispatch("GET", "/api/predictions?horizon=30m&limit=5")
    finally:
        service.PATHS.clear()
        service.PATHS.update(original)
        service._read_csv_cached.cache_clear()

    assert status == 200
    row = json.loads(body)["rows"][0]
    assert row["zone_id"] == "abc"
    assert row["activity_probability"] == 0.5
    assert row["spatial_probability"] == 0.8
    assert row["probability"] == 0.4
    assert row["risk_band"] == "elevated"


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
        original_spatial = service.PATHS["spatial_predictions_30m"]
        service.PATHS["predictions_30m"] = pred_path
        service.PATHS["spatial_predictions_30m"] = Path(tmpdir) / "missing_spatial.csv"
        service._read_csv_cached.cache_clear()
        try:
            status, _, body = service.dispatch("GET", "/api/grid.geojson?horizon=30m&min_events=0")
        finally:
            service.PATHS["predictions_30m"] = original
            service.PATHS["spatial_predictions_30m"] = original_spatial
            service._read_csv_cached.cache_clear()

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


def test_worklog_endpoint_returns_latest_progress_sections(tmp_path):
    worklog = tmp_path / "WORKLOG.md"
    worklog.write_text(
        "\n".join(
            [
                "# Worklog",
                "",
                "## 2026-07-07 Earlier update",
                "",
                "Current objective:",
                "- Old objective",
                "",
                "Blockers:",
                "- Old blocker",
                "",
                "Next steps:",
                "- Old next step",
                "",
                "## 2026-07-08 Live worklog page",
                "",
                "Current objective:",
                "- Ship live worklog progress page",
                "",
                "Test results:",
                "- 2 focused tests passing",
                "",
                "Blockers:",
                "- None.",
                "",
                "Next steps:",
                "- Launch the dashboard and confirm /worklog refreshes.",
            ]
        ),
        encoding="utf-8",
    )
    original = dict(service.PATHS)
    service.PATHS["worklog"] = worklog
    try:
        status, headers, body = service.dispatch("GET", "/api/worklog")
    finally:
        service.PATHS.clear()
        service.PATHS.update(original)

    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    payload = json.loads(body)
    assert payload["entry_count"] == 2
    assert payload["latest_title"] == "2026-07-08 Live worklog page"
    assert payload["current_objective"] == ["Ship live worklog progress page"]
    assert payload["test_results"] == ["2 focused tests passing"]
    assert payload["blockers"] == ["None."]
    assert payload["next_steps"] == ["Launch the dashboard and confirm /worklog refreshes."]
    assert "## 2026-07-08 Live worklog page" in payload["raw_markdown"]
    assert "Old objective" not in payload["raw_markdown"]


def test_worklog_endpoint_returns_descending_detail_html(tmp_path):
    worklog = tmp_path / "WORKLOG.md"
    worklog.write_text(
        "\n".join(
            [
                "# Worklog",
                "",
                "## 2026-07-07 Older entry",
                "",
                "Current objective:",
                "- Finish the first pass.",
                "",
                "## 2026-07-08 Newer entry",
                "",
                "Current objective:",
                "- Ship the newest change.",
            ]
        ),
        encoding="utf-8",
    )
    original = dict(service.PATHS)
    service.PATHS["worklog"] = worklog
    try:
        status, headers, body = service.dispatch("GET", "/api/worklog")
    finally:
        service.PATHS.clear()
        service.PATHS.update(original)

    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    payload = json.loads(body)
    assert payload["latest_title"] == "2026-07-08 Newer entry"
    assert payload["current_objective"] == ["Ship the newest change."]
    assert payload["html"].index("2026-07-07 Older entry") < payload["html"].index("2026-07-08 Newer entry")
    assert payload["detail_html"].index("2026-07-08 Newer entry") < payload["detail_html"].index(
        "2026-07-07 Older entry"
    )


def test_worklog_endpoint_returns_rendered_markdown_html(tmp_path):
    worklog = tmp_path / "WORKLOG.md"
    worklog.write_text(
        "\n".join(
            [
                "# Worklog",
                "",
                "A paragraph with *focus* and [docs](https://example.com).",
                "",
                "```python",
                "print('hello')",
                "```",
                "",
                "1. first",
                "2. second",
                "",
                "| Metric | Value |",
                "| --- | --- |",
                "| Tests | 6 |",
                "",
                "<script>alert('x')</script>",
            ]
        ),
        encoding="utf-8",
    )
    original = dict(service.PATHS)
    service.PATHS["worklog"] = worklog
    try:
        status, headers, body = service.dispatch("GET", "/api/worklog")
    finally:
        service.PATHS.clear()
        service.PATHS.update(original)

    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    payload = json.loads(body)
    assert "<h1>Worklog</h1>" in payload["html"]
    assert "<em>focus</em>" in payload["html"]
    assert '<a href="https://example.com">' in payload["html"]
    assert "<pre><code" in payload["html"]
    assert "print(&#39;hello&#39;)" in payload["html"] or "print('hello')" in payload["html"]
    assert "<ol>" in payload["html"]
    assert "<li>first</li>" in payload["html"]
    assert "<table>" in payload["html"]
    assert "&lt;script&gt;alert" in payload["html"]
    assert "<script>" not in payload["html"]


def test_worklog_endpoint_sanitizes_unsafe_links(tmp_path):
    worklog = tmp_path / "WORKLOG.md"
    worklog.write_text("[bad](javascript:alert(1))", encoding="utf-8")
    original = dict(service.PATHS)
    service.PATHS["worklog"] = worklog
    try:
        status, headers, body = service.dispatch("GET", "/api/worklog")
    finally:
        service.PATHS.clear()
        service.PATHS.update(original)

    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    payload = json.loads(body)
    assert "javascript:alert(1)" not in payload["html"]
    assert "<a" in payload["html"]


def test_worklog_page_polls_live_endpoint():
    status, headers, body = service.dispatch("GET", "/worklog")

    assert status == 200
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert "Ghost Sweep Worklog" in body
    assert "fetchJson('/api/worklog')" in body
    assert "setInterval(loadWorklog, 5000)" in body
    assert "currentObjective" in body
    assert "worklogHtml" in body


def test_worklog_page_uses_rendered_html_field():
    status, headers, body = service.dispatch("GET", "/worklog")

    assert status == 200
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert "<h2>Full worklog</h2>" in body
    assert 'class="logWrap"' in body
    assert 'id="worklogHtml"' in body
    assert "payload.html" in body
    assert "innerHTML = payload.html" in body
    assert ".markdown-body table" in body
    assert ".markdown-body ul" in body
    assert ".markdown-body ol" in body


def test_worklog_page_matches_rich_shell_contract():
    status, headers, body = service.dispatch("GET", "/worklog")

    assert status == 200
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert "Self-service progress board" in body
    assert "Refresh now" in body
    assert "Latest entry" in body
    assert "Objectives" in body
    assert "Blockers" in body
    assert "Next steps" in body
    assert "Live status" in body
    assert 'id="worklogHtml"' in body
    assert "payload.html" in body
    assert 'rel="icon"' in body
    assert 'href="data:,"' in body
    assert 'id="entryChip"' in body
    assert 'id="liveChip"' in body
    assert 'id="objectiveCount"' in body
    assert 'id="blockerCount"' in body
    assert 'id="nextStepCount"' in body
    assert 'id="logTimestamp"' in body


def test_worklog_page_inlines_a_favicon():
    status, headers, body = service.dispatch("GET", "/worklog")

    assert status == 200
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert 'rel="icon"' in body
    assert 'href="data:,"' in body
