import json

from analysis.build_dashboard_manifest import build_manifest, write_manifest


def test_manifest_groups_traceable_artifacts(tmp_path):
    manifest = build_manifest()

    assert manifest["coverage_mode"] == "road_access"
    assert "coverage_grid" in manifest["artifact_groups"]
    assert "road_coverage_grid" in manifest["artifact_groups"]
    assert "feature_mart" in manifest["artifact_groups"]
    assert "multi_horizon_models" in manifest["artifact_groups"]
    assert "two_stage_models" in manifest["artifact_groups"]
    assert "two_stage_metadata" in manifest["artifact_groups"]
    assert "two_stage_predictions" in manifest["artifact_groups"]
    assert all(
        "path" in artifact
        for artifacts in manifest["artifact_groups"].values()
        for artifact in artifacts
    )

    output = write_manifest(tmp_path / "manifest.json")
    assert json.loads(output.read_text(encoding="utf-8"))["project"] == "ghost-sweep"
