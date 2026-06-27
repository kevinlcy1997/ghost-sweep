from pathlib import Path


def test_experiment_runner_logs_to_mlflow():
    source = Path("analysis/run_ml_experiment.py").read_text(encoding="utf-8")

    assert "import mlflow" in source
    assert "MLFLOW_TRACKING_URI" in source
    assert "mlflow.set_tracking_uri" in source
    assert "mlflow.set_experiment" in source
    assert "mlflow.start_run" in source
    assert "mlflow.log_metric" in source
    assert "mlflow.log_artifact" in source
    assert "BEST_MODEL_PATH" in source
