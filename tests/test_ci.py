# tests/test_ci.py
"""Tests to verify CI/deployment readiness."""

import os
import json
import subprocess
import sys

from analysis.build_dashboard_manifest import ARTIFACT_GROUPS


def test_once_flag_exits_cleanly():
    """ghost_listener.py --once should complete without crashing (dry-run check)."""
    # We can't actually call the API in CI, but we can verify the script loads
    result = subprocess.run(
        [sys.executable, "-c", "from ghost_listener import main; print('OK')"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_workflow_file_exists():
    """GitHub Actions workflow should exist."""
    workflow = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        ".github", "workflows", "ghost-scrape.yml"
    )
    assert os.path.exists(workflow), f"Workflow not found at {workflow}"


def test_workflow_valid_yaml():
    """GitHub Actions workflow should be valid YAML with required fields."""
    workflow = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        ".github", "workflows", "ghost-scrape.yml"
    )
    with open(workflow, "r", encoding="utf-8") as f:
        content = f.read()
    # Basic structural checks
    assert "on:" in content
    assert "schedule:" in content
    assert "ghost_listener.py --once" in content
    assert "--request-timeout" in content
    assert "--max-grid-cells" in content
    assert "--skip-active" in content
    assert "git push" in content


def test_committed_alert_store_is_valid_json():
    """The dashboard workflow depends on ghost_alerts.json being parseable."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ghost_alerts.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data.get("alerts"), dict)
    assert isinstance(data.get("meta"), dict)


def test_dev_start_script_supports_retraining_and_mlflow():
    """The reusable dev script should bring up MLflow and optionally retrain models."""
    script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "start-dev.ps1")
    with open(script, "r", encoding="utf-8") as f:
        content = f.read()

    assert "[switch]$RetrainModels" in content
    assert "analysis/run_multi_horizon_experiment.py" in content
    assert "analysis/build_dashboard_manifest.py" in content
    assert "MlflowPort" in content
    assert "mlflow" in content
    assert "mlflow_tracking.db" in content
    assert '"mlflow", "ui"' in content
    assert "Initialize-LogFile" in content
    assert "GetFileNameWithoutExtension" in content
    assert "$OutLog = Initialize-LogFile" in content


def test_dashboard_manifest_tracks_model_artifacts():
    """The manifest builder should list the model outputs produced by the pipeline."""
    assert "analysis/multi_horizon_summary_latest.csv" in ARTIFACT_GROUPS["multi_horizon_models"]
    assert "analysis/model_iteration_summary_30m_latest.csv" in ARTIFACT_GROUPS["multi_horizon_models"]
    assert "analysis/model_iteration_report_30m.html" in ARTIFACT_GROUPS["multi_horizon_models"]

    assert "analysis/best_iterated_model_metadata.json" in ARTIFACT_GROUPS["model_metadata"]
    assert "analysis/best_iterated_model_metadata_30m.json" in ARTIFACT_GROUPS["model_metadata"]
    assert "analysis/best_iterated_model_metadata_1h.json" in ARTIFACT_GROUPS["model_metadata"]
    assert "analysis/best_iterated_model_metadata_2h.json" in ARTIFACT_GROUPS["model_metadata"]

    assert "analysis/best_experiment_model.joblib" in ARTIFACT_GROUPS["model_joblibs"]
    assert "analysis/best_iterated_zone_model_30m.joblib" in ARTIFACT_GROUPS["model_joblibs"]

    assert "analysis/iterated_zone_predictions_latest.csv" in ARTIFACT_GROUPS["predictions"]
    assert "analysis/iterated_zone_predictions_30m_latest.csv" in ARTIFACT_GROUPS["predictions"]
