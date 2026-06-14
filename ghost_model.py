# ghost_model.py
"""LightGBM model for Ghost Sweep warden activity prediction."""

import os
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib

log = logging.getLogger("ghost_model")

FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend", "month",
    "cell_historical_freq", "cell_7d_count", "cell_24h_count",
    "cell_last_seen_hours_ago", "neighbor_24h_count", "streak_active",
    "upvote_ratio", "avg_report_count",
    "district_24h_count", "district_historical_rate",
    "district_active_cells", "district_hour_rate",
    "hour_historical_rate", "dow_hour_rate",
]
CATEGORICAL_COLS = ["district", "region"]
LABEL_COL = "has_warden"
MODEL_DIR = Path("models")


class GhostModel:
    def __init__(self, model_path: str | None = None):
        self._model: lgb.Booster | None = None
        if model_path and os.path.exists(model_path):
            self._model = joblib.load(model_path)

    def check_data_gate(self, days_collected: int) -> dict:
        min_days = 14
        return {
            "ready": days_collected >= min_days,
            "days_collected": days_collected,
            "days_needed": min_days,
            "days_remaining": max(0, min_days - days_collected),
        }

    def train(self, df: pd.DataFrame) -> dict:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

        X = df[feature_cols + cat_cols].copy()
        y = df[LABEL_COL]
        for col in cat_cols:
            X[col] = X[col].astype("category")

        n = len(X)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain)

        params = {
            "objective": "binary", "metric": "auc",
            "num_leaves": 31, "learning_rate": 0.05,
            "is_unbalance": True, "verbose": -1, "seed": 42,
        }
        self._model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        y_pred_prob = self._model.predict(X_test)
        y_pred_bin = (y_pred_prob >= 0.5).astype(int)

        metrics = {
            "auc_roc": float(roc_auc_score(y_test, y_pred_prob)) if y_test.nunique() > 1 else 0.0,
            "precision": float(precision_score(y_test, y_pred_bin, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_bin, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_bin, zero_division=0)),
            "train_size": train_end, "val_size": val_end - train_end,
            "test_size": n - val_end, "n_estimators": self._model.num_trees(),
        }

        importance = self._model.feature_importance(importance_type="gain")
        feat_names = self._model.feature_name()
        top_features = sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)[:10]
        metrics["top_features"] = [{"name": n, "importance": float(v)} for n, v in top_features]

        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / "model_latest.joblib"
        joblib.dump(self._model, model_path)
        metrics["model_path"] = str(model_path)
        return metrics

    def predict(self, df: pd.DataFrame) -> list[float]:
        if self._model is None:
            raise RuntimeError("No trained model loaded.")
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
        X = df[feature_cols + cat_cols].copy()
        for col in cat_cols:
            X[col] = X[col].astype("category")
        return self._model.predict(X).tolist()

    def load(self, path: str):
        self._model = joblib.load(path)
