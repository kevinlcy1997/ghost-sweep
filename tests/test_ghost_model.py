# tests/test_ghost_model.py
import pytest
import pandas as pd
from ghost_model import GhostModel


def test_data_gate_blocks_training():
    model = GhostModel()
    result = model.check_data_gate(days_collected=5)
    assert result["ready"] is False
    assert result["days_needed"] == 14


def test_data_gate_allows_training():
    model = GhostModel()
    result = model.check_data_gate(days_collected=14)
    assert result["ready"] is True


def test_train_and_predict():
    import numpy as np
    rng = np.random.default_rng(42)
    model = GhostModel()

    n = 200
    cell_24h = rng.integers(0, 10, n)
    has_warden = (cell_24h >= 5).astype(int)

    df = pd.DataFrame({
        "grid_cell": [f"22.{i%10:03d}_114.{i%10:03d}" for i in range(n)],
        "district": ["Mong Kok"] * n,
        "region": ["Kowloon West"] * n,
        "hour": [i % 24 for i in range(n)],
        "day_of_week": [i % 7 for i in range(n)],
        "is_weekend": [1 if i % 7 >= 5 else 0 for i in range(n)],
        "month": [6] * n,
        "cell_historical_freq": rng.integers(1, 50, n),
        "cell_7d_count": rng.integers(0, 10, n),
        "cell_24h_count": cell_24h,
        "cell_last_seen_hours_ago": rng.uniform(0, 100, n),
        "neighbor_24h_count": rng.integers(0, 10, n),
        "streak_active": rng.integers(0, 2, n),
        "upvote_ratio": rng.uniform(0.5, 1.0, n),
        "avg_report_count": rng.uniform(1, 5, n),
        "district_24h_count": rng.integers(5, 30, n),
        "district_historical_rate": rng.uniform(1, 10, n),
        "district_active_cells": rng.integers(1, 10, n),
        "district_hour_rate": rng.uniform(0, 3, n),
        "hour_historical_rate": rng.uniform(0, 5, n),
        "dow_hour_rate": rng.uniform(0, 2, n),
        "has_warden": has_warden,
    })

    metrics = model.train(df)
    assert "auc_roc" in metrics
    assert metrics["auc_roc"] >= 0.6
    assert metrics["n_estimators"] > 0

    pred_df = df.drop(columns=["has_warden"]).head(10)
    predictions = model.predict(pred_df)
    assert len(predictions) == 10
    assert all(0 <= p <= 1 for p in predictions)
