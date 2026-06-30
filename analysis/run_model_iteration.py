from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from ghost_ranking_features import build_zone_ranking_training_data
from ghost_ranking_metrics import (
    brier_score,
    expected_calibration_error,
    group_hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    risk_band,
    top_decile_lift,
)

try:
    from run_zone_ranking_experiment import (
        CATEGORICAL_FEATURES,
        DB_PATH,
        NUMERIC_FEATURES,
        TARGET,
        _preprocessor,
    )
except ModuleNotFoundError:
    from analysis.run_zone_ranking_experiment import (
        CATEGORICAL_FEATURES,
        DB_PATH,
        NUMERIC_FEATURES,
        TARGET,
        _preprocessor,
    )


OUTPUT_DIR = ROOT / "analysis"
FOLD_RESULTS_PATH = OUTPUT_DIR / "model_iteration_folds_latest.csv"
SUMMARY_PATH = OUTPUT_DIR / "model_iteration_summary_latest.csv"
REPORT_PATH = OUTPUT_DIR / "model_iteration_report.html"
MODEL_PATH = OUTPUT_DIR / "best_iterated_zone_model.joblib"
METADATA_PATH = OUTPUT_DIR / "best_iterated_model_metadata.json"
PREDICTIONS_PATH = OUTPUT_DIR / "iterated_zone_predictions_latest.csv"


def horizon_slug(horizon_minutes: int) -> str:
    if horizon_minutes % 60 == 0:
        return f"{horizon_minutes // 60}h"
    return f"{horizon_minutes}m"


def target_for_horizon(horizon_minutes: int) -> str:
    return f"alert_next_{horizon_slug(horizon_minutes)}"


def _artifact_paths(slug: str) -> dict[str, Path]:
    return {
        "fold_results": OUTPUT_DIR / f"model_iteration_folds_{slug}_latest.csv",
        "summary": OUTPUT_DIR / f"model_iteration_summary_{slug}_latest.csv",
        "report": OUTPUT_DIR / f"model_iteration_report_{slug}.html",
        "model": OUTPUT_DIR / f"best_iterated_zone_model_{slug}.joblib",
        "metadata": OUTPUT_DIR / f"best_iterated_model_metadata_{slug}.json",
        "predictions": OUTPUT_DIR / f"iterated_zone_predictions_{slug}_latest.csv",
    }


@dataclass(frozen=True)
class Candidate:
    name: str
    factory: Callable[[], Any]


def make_walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 4,
) -> list[tuple[pd.Series, pd.Series]]:
    frame = df.copy()
    frame["target_time"] = pd.to_datetime(frame["target_time"])
    times = pd.Series(sorted(frame["target_time"].unique()))
    if len(times) < 4:
        raise ValueError("Need at least four target_time values for walk-forward evaluation.")

    test_size = max(1, len(times) // (n_splits + 2))
    splits: list[tuple[pd.Series, pd.Series]] = []
    for split_index in range(n_splits):
        train_end = len(times) - test_size * (n_splits - split_index)
        test_end = min(len(times), train_end + test_size)
        if train_end <= 0 or test_end <= train_end:
            continue
        train_times = set(times.iloc[:train_end])
        test_times = set(times.iloc[train_end:test_end])
        train_mask = frame["target_time"].isin(train_times)
        test_mask = frame["target_time"].isin(test_times)
        if train_mask.any() and test_mask.any():
            splits.append((train_mask, test_mask))
    if not splits:
        raise ValueError("Could not construct non-empty walk-forward splits.")
    return splits


def _score_predictions(
    y_true: pd.Series,
    y_score: np.ndarray,
    groups: dict[str, pd.Series] | None = None,
) -> dict[str, float]:
    positives = int(pd.Series(y_true).sum())
    rows = int(len(y_true))
    metrics = {
        "rows": rows,
        "positives": positives,
        "base_rate": float(positives / rows) if rows else 0.0,
        "precision_at_20": precision_at_k(y_true, y_score, 20),
        "precision_at_50": precision_at_k(y_true, y_score, 50),
        "precision_at_100": precision_at_k(y_true, y_score, 100),
        "recall_at_20": recall_at_k(y_true, y_score, 20),
        "recall_at_50": recall_at_k(y_true, y_score, 50),
        "recall_at_100": recall_at_k(y_true, y_score, 100),
        "top_decile_lift": top_decile_lift(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "brier_score": brier_score(y_true, y_score),
        "expected_calibration_error": expected_calibration_error(y_true, y_score),
    }
    if groups:
        for name, values in groups.items():
            metrics[f"{name}_hit_rate_at_20"] = group_hit_rate_at_k(y_true, y_score, values, 20)
            metrics[f"{name}_hit_rate_at_50"] = group_hit_rate_at_k(y_true, y_score, values, 50)
            metrics[f"{name}_hit_rate_at_100"] = group_hit_rate_at_k(y_true, y_score, values, 100)
    if y_true.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["roc_auc"] = 0.0
    return {key: float(value) for key, value in metrics.items()}


def _candidate_models() -> list[Candidate]:
    return [
        Candidate(
            "logistic_balanced",
            lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        Candidate(
            "random_forest_balanced",
            lambda: RandomForestClassifier(
                n_estimators=240,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        Candidate(
            "extra_trees_balanced",
            lambda: ExtraTreesClassifier(
                n_estimators=260,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        Candidate(
            "lightgbm_balanced",
            lambda: LGBMClassifier(
                n_estimators=260,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
            ),
        ),
        Candidate(
            "lightgbm_conservative",
            lambda: LGBMClassifier(
                n_estimators=360,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=45,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
            ),
        ),
    ]


def _make_pipeline(candidate: Candidate) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _preprocessor()),
            ("model", candidate.factory()),
        ]
    )


def evaluate_candidates(
    df: pd.DataFrame,
    n_splits: int = 4,
    target_col: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = df.copy()
    frame["target_time"] = pd.to_datetime(frame["target_time"])
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    splits = make_walk_forward_splits(frame, n_splits=n_splits)
    rows: list[dict[str, Any]] = []

    for candidate in _candidate_models():
        for fold_index, (train_mask, test_mask) in enumerate(splits, start=1):
            pipeline = _make_pipeline(candidate)
            pipeline.fit(frame.loc[train_mask, feature_cols], frame.loc[train_mask, target_col])
            scores = pipeline.predict_proba(frame.loc[test_mask, feature_cols])[:, 1]
            metrics = _score_predictions(
                frame.loc[test_mask, target_col],
                scores,
                groups={
                    "district": frame.loc[test_mask, "district"],
                    "region": frame.loc[test_mask, "region"],
                },
            )
            rows.append(
                {
                    "model": candidate.name,
                    "fold": fold_index,
                    "train_rows": int(train_mask.sum()),
                    "test_rows": int(test_mask.sum()),
                    **metrics,
                }
            )

    fold_results = pd.DataFrame(rows)
    summary_rows = []
    for model, group in fold_results.groupby("model", sort=False):
        summary_rows.append(
            {
                "model": model,
                "folds": int(len(group)),
                "median_precision_at_20": float(group["precision_at_20"].median()),
                "median_precision_at_50": float(group["precision_at_50"].median()),
                "median_precision_at_100": float(group["precision_at_100"].median()),
                "median_recall_at_50": float(group["recall_at_50"].median()),
                "median_recall_at_100": float(group["recall_at_100"].median()),
                "median_average_precision": float(group["average_precision"].median()),
                "median_top_decile_lift": float(group["top_decile_lift"].median()),
                "median_district_hit_rate_at_50": float(group["district_hit_rate_at_50"].median()),
                "median_region_hit_rate_at_50": float(group["region_hit_rate_at_50"].median()),
                "median_brier_score": float(group["brier_score"].median()),
                "median_expected_calibration_error": float(
                    group["expected_calibration_error"].median()
                ),
                "median_roc_auc": float(group["roc_auc"].median()),
                "precision_at_20_std": float(group["precision_at_20"].std(ddof=0)),
                "average_precision_std": float(group["average_precision"].std(ddof=0)),
            }
        )
    return fold_results, pd.DataFrame(summary_rows)


def select_satisfactory_model(rows: list[dict[str, Any]] | pd.DataFrame) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    ranked = frame.sort_values(
        by=[
            "median_top_decile_lift",
            "median_average_precision",
            "median_precision_at_100",
            "median_precision_at_50",
            "median_brier_score",
            "precision_at_20_std",
        ],
        ascending=[False, False, False, False, True, True],
    )
    return ranked.iloc[0].to_dict()


def format_prediction_artifact(
    base: pd.DataFrame,
    probabilities: np.ndarray | list[float],
    actual: pd.Series | np.ndarray | list[int],
) -> pd.DataFrame:
    """Return dashboard-ready predictions sorted by probability descending."""
    predictions = base.copy()
    clipped = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    predictions["score"] = clipped
    predictions["probability"] = clipped
    predictions["actual"] = np.asarray(actual, dtype=int)
    predictions = predictions.sort_values(
        by=["probability", "zone_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    predictions["rank"] = np.arange(1, len(predictions) + 1)
    predictions["risk_band"] = predictions["probability"].map(risk_band)
    return predictions


def _load_events() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("select * from events", conn)


def _write_report(
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    chosen: dict[str, Any],
    report_path: Path = REPORT_PATH,
) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Ghost Sweep Model Iteration</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    h1, h2 {{ margin-bottom: 10px; }}
    .summary {{ padding: 14px; border: 1px solid #d9e2ef; border-radius: 8px; background: #f7f9fc; margin-bottom: 22px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ef; padding: 8px; text-align: right; }}
    th {{ background: #eef3f9; }}
    td:first-child, th:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Ghost Sweep Model Iteration</h1>
  <div class="summary">
    <strong>Chosen model:</strong> {chosen["model"]}.
    Median AP {float(chosen["median_average_precision"]):.3f},
    median precision@20 {float(chosen["median_precision_at_20"]):.3f},
    median lift {float(chosen["median_top_decile_lift"]):.3f}.
  </div>
  <h2>Model Summary</h2>
  {summary.to_html(index=False)}
  <h2>Walk-Forward Fold Results</h2>
  {fold_results.to_html(index=False)}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")


def run_model_iteration(resolution: int = 8, horizon_minutes: int = 120) -> dict[str, Any]:
    slug = horizon_slug(horizon_minutes)
    target_col = target_for_horizon(horizon_minutes)
    paths = _artifact_paths(slug)
    events = _load_events()
    df = build_zone_ranking_training_data(
        events.to_dict("records"),
        forecast_hours=max(1, horizon_minutes // 60),
        horizon_minutes=horizon_minutes,
        target_col=target_col,
        lookback_days=7,
        resolution=resolution,
    )
    fold_results, summary = evaluate_candidates(df, target_col=target_col)
    chosen = select_satisfactory_model(summary)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    train_cutoff = pd.to_datetime(df["target_time"]).quantile(0.8)
    train_mask = pd.to_datetime(df["target_time"]) < train_cutoff
    test_mask = ~train_mask
    candidate = next(item for item in _candidate_models() if item.name == chosen["model"])
    pipeline = _make_pipeline(candidate)
    pipeline.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, target_col])
    scores = pipeline.predict_proba(df.loc[test_mask, feature_cols])[:, 1]
    holdout_metrics = _score_predictions(
        df.loc[test_mask, target_col],
        scores,
        groups={
            "district": df.loc[test_mask, "district"],
            "region": df.loc[test_mask, "region"],
        },
    )

    predictions = format_prediction_artifact(
        df.loc[test_mask, ["target_time", "zone_id", "district", "region", "zone_lat", "zone_lng"]],
        scores,
        df.loc[test_mask, target_col],
    )

    fold_results.to_csv(paths["fold_results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    joblib.dump(pipeline, paths["model"])
    metadata = {
        "resolution": resolution,
        "horizon_minutes": horizon_minutes,
        "horizon_slug": slug,
        "target_col": target_col,
        "chosen_model": chosen,
        "holdout_metrics": holdout_metrics,
        "features": feature_cols,
        "training_rows": int(len(df)),
    }
    metadata["metadata_path"] = str(paths["metadata"].relative_to(ROOT))
    metadata["predictions_path"] = str(paths["predictions"].relative_to(ROOT))
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_report(fold_results, summary, chosen, paths["report"])
    return metadata


def main() -> None:
    metadata = run_model_iteration()
    chosen = metadata["chosen_model"]
    holdout = metadata["holdout_metrics"]
    print(
        f"chosen={chosen['model']} "
        f"median_ap={chosen['median_average_precision']:.3f} "
        f"median_p20={chosen['median_precision_at_20']:.3f} "
        f"holdout_ap={holdout['average_precision']:.3f} "
        f"holdout_p20={holdout['precision_at_20']:.3f}"
    )
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
