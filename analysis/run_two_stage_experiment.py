from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.run_model_iteration import (
    _make_pipeline as _make_spatial_pipeline,
    _score_predictions as _score_spatial_predictions,
    horizon_slug,
    target_for_horizon,
)
from analysis.run_zone_ranking_experiment import CATEGORICAL_FEATURES, DB_PATH, NUMERIC_FEATURES
from analysis.two_stage_splits import make_positive_count_holdout, make_purged_rolling_splits
from ghost_activity_features import build_activity_training_data
from ghost_ranking_features import build_zone_ranking_training_data, sample_spatial_training_rows
from ghost_ranking_metrics import brier_score, expected_calibration_error, risk_band
from ghost_zones import DEFAULT_H3_RESOLUTION


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)

OUTPUT_DIR = ROOT / "analysis"
HORIZONS = [30, 60, 120]
SUMMARY_PATH = OUTPUT_DIR / "two_stage_summary_latest.csv"
REPORT_PATH = OUTPUT_DIR / "two_stage_report.html"

ACTIVITY_NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "city_event_count_1h",
    "city_event_count_3h",
    "city_event_count_24h",
    "city_event_count_7d",
    "active_districts_24h",
    "active_regions_24h",
    "city_3h_to_24h_ratio",
    "city_24h_to_7d_ratio",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]
ACTIVITY_CATEGORICAL_FEATURES = ["hour_bucket"]


@dataclass(frozen=True)
class Candidate:
    name: str
    factory: Callable[[], Any]


def activity_target_for_horizon(horizon_minutes: int) -> str:
    return f"activity_next_{horizon_slug(horizon_minutes)}"


def _artifact_paths(slug: str) -> dict[str, Path]:
    return {
        "activity_folds": OUTPUT_DIR / f"activity_model_folds_{slug}_latest.csv",
        "activity_metadata": OUTPUT_DIR / f"activity_model_metadata_{slug}.json",
        "activity_model": OUTPUT_DIR / f"best_activity_model_{slug}.joblib",
        "activity_predictions": OUTPUT_DIR / f"activity_predictions_{slug}_latest.csv",
        "spatial_folds": OUTPUT_DIR / f"spatial_model_folds_{slug}_latest.csv",
        "spatial_metadata": OUTPUT_DIR / f"spatial_model_metadata_{slug}.json",
        "spatial_model": OUTPUT_DIR / f"best_spatial_zone_model_{slug}.joblib",
        "spatial_predictions": OUTPUT_DIR / f"spatial_zone_predictions_{slug}_latest.csv",
    }


def _candidate_models() -> list[Candidate]:
    return [
        Candidate(
            "logistic_balanced",
            lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        Candidate(
            "extra_trees_balanced",
            lambda: ExtraTreesClassifier(
                n_estimators=220,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
        ),
        Candidate(
            "lightgbm_conservative",
            lambda: LGBMClassifier(
                n_estimators=260,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=35,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
            ),
        ),
    ]


def _activity_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ACTIVITY_NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ACTIVITY_CATEGORICAL_FEATURES,
            ),
        ]
    )


def _make_activity_pipeline(candidate: Candidate) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _activity_preprocessor()),
            ("model", candidate.factory()),
        ]
    )


def _binary_metrics(y_true: pd.Series, y_score: np.ndarray) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    y_score = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    y_pred = (y_score >= 0.5).astype(int)
    positives = int(y_true.sum())
    rows = int(len(y_true))
    return {
        "rows": float(rows),
        "positives": float(positives),
        "base_rate": float(positives / rows) if rows else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_true.nunique() > 1 else 0.0,
        "average_precision": float(average_precision_score(y_true, y_score)) if rows else 0.0,
        "brier_score": float(brier_score(y_true, y_score)),
        "expected_calibration_error": float(expected_calibration_error(y_true, y_score)),
        "precision_at_threshold_0_5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold_0_5": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _select_model(summary: pd.DataFrame, stage: str) -> dict[str, Any]:
    if stage == "activity":
        ranked = summary.sort_values(
            by=["median_average_precision", "median_roc_auc", "median_brier_score"],
            ascending=[False, False, True],
        )
    else:
        ranked = summary.sort_values(
            by=["median_top_decile_lift", "median_average_precision", "median_precision_at_50"],
            ascending=[False, False, False],
        )
    return ranked.iloc[0].to_dict()


def _evaluate_activity_candidates(
    df: pd.DataFrame,
    target_col: str,
    horizon_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = ACTIVITY_NUMERIC_FEATURES + ACTIVITY_CATEGORICAL_FEATURES
    splits = make_purged_rolling_splits(df, horizon_minutes=horizon_minutes)
    rows: list[dict[str, Any]] = []
    for candidate in _candidate_models():
        for split in splits:
            if df.loc[split.train_mask, target_col].nunique() < 2:
                continue
            pipeline = _make_activity_pipeline(candidate)
            pipeline.fit(df.loc[split.train_mask, feature_cols], df.loc[split.train_mask, target_col])
            scores = pipeline.predict_proba(df.loc[split.validation_mask, feature_cols])[:, 1]
            rows.append(
                {
                    "model": candidate.name,
                    "fold": split.metadata["fold"],
                    "train_rows": split.metadata["train_rows"],
                    "validation_rows": split.metadata["validation_rows"],
                    **_binary_metrics(df.loc[split.validation_mask, target_col], scores),
                    **{f"split_{key}": value for key, value in split.metadata.items()},
                }
            )
    fold_results = pd.DataFrame(rows)
    if fold_results.empty:
        raise ValueError("No valid activity folds had both classes in training.")
    summary = (
        fold_results.groupby("model", sort=False)
        .agg(
            folds=("fold", "count"),
            median_average_precision=("average_precision", "median"),
            median_roc_auc=("roc_auc", "median"),
            median_brier_score=("brier_score", "median"),
            median_expected_calibration_error=("expected_calibration_error", "median"),
            median_precision_at_threshold_0_5=("precision_at_threshold_0_5", "median"),
            median_recall_at_threshold_0_5=("recall_at_threshold_0_5", "median"),
        )
        .reset_index()
    )
    return fold_results, summary


def _evaluate_spatial_candidates(
    df: pd.DataFrame,
    target_col: str,
    horizon_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    splits = make_purged_rolling_splits(df, horizon_minutes=horizon_minutes)
    rows: list[dict[str, Any]] = []
    for candidate in _candidate_models():
        for split in splits:
            train_sample = sample_spatial_training_rows(
                df.loc[split.train_mask],
                target_col,
                negative_ratio=5,
                inactive_negative_fraction=0.02,
            )
            if train_sample.empty or train_sample[target_col].nunique() < 2:
                continue
            pipeline = _make_spatial_pipeline(candidate)
            pipeline.fit(train_sample[feature_cols], train_sample[target_col])
            scores = pipeline.predict_proba(df.loc[split.validation_mask, feature_cols])[:, 1]
            metrics = _score_spatial_predictions(
                df.loc[split.validation_mask, target_col],
                scores,
                groups={
                    "district": df.loc[split.validation_mask, "district"],
                    "region": df.loc[split.validation_mask, "region"],
                },
            )
            rows.append(
                {
                    "model": candidate.name,
                    "fold": split.metadata["fold"],
                    "train_rows": int(len(train_sample)),
                    "unsampled_train_rows": split.metadata["train_rows"],
                    "validation_rows": split.metadata["validation_rows"],
                    **metrics,
                    **{f"split_{key}": value for key, value in split.metadata.items()},
                }
            )
    fold_results = pd.DataFrame(rows)
    if fold_results.empty:
        raise ValueError("No valid spatial folds had both classes after sampling.")
    summary = (
        fold_results.groupby("model", sort=False)
        .agg(
            folds=("fold", "count"),
            median_precision_at_20=("precision_at_20", "median"),
            median_precision_at_50=("precision_at_50", "median"),
            median_precision_at_100=("precision_at_100", "median"),
            median_recall_at_50=("recall_at_50", "median"),
            median_recall_at_100=("recall_at_100", "median"),
            median_average_precision=("average_precision", "median"),
            median_top_decile_lift=("top_decile_lift", "median"),
            median_district_hit_rate_at_50=("district_hit_rate_at_50", "median"),
            median_region_hit_rate_at_50=("region_hit_rate_at_50", "median"),
            median_brier_score=("brier_score", "median"),
            median_expected_calibration_error=("expected_calibration_error", "median"),
            median_roc_auc=("roc_auc", "median"),
        )
        .reset_index()
    )
    return fold_results, summary


def _probabilities_for_time(
    target_times: pd.Series,
    activity_scores: pd.DataFrame,
    default_probability: float,
) -> np.ndarray:
    if activity_scores.empty:
        return np.full(len(target_times), default_probability, dtype=float)
    lookup = {
        pd.Timestamp(row.target_time): float(row.activity_probability)
        for row in activity_scores[["target_time", "activity_probability"]].itertuples(index=False)
    }
    return np.asarray(
        [lookup.get(pd.Timestamp(value), default_probability) for value in target_times],
        dtype=float,
    )


def combine_activity_and_spatial_scores(
    spatial: pd.DataFrame,
    activity_probability: float | pd.Series | np.ndarray,
) -> pd.DataFrame:
    combined = spatial.copy()
    if np.isscalar(activity_probability):
        activity_scores = np.full(len(combined), float(activity_probability), dtype=float)
    else:
        activity_scores = np.asarray(activity_probability, dtype=float)
    spatial_scores = np.clip(combined["spatial_probability"].astype(float).to_numpy(), 0.0, 1.0)
    activity_scores = np.clip(activity_scores, 0.0, 1.0)
    final_scores = np.clip(spatial_scores * activity_scores, 0.0, 1.0)
    combined["activity_probability"] = activity_scores
    combined["probability"] = final_scores
    combined["score"] = final_scores
    combined = combined.sort_values(
        by=["probability", "zone_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    combined["rank"] = np.arange(1, len(combined) + 1)
    combined["risk_band"] = combined["probability"].map(risk_band)
    return combined


def _load_events() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("select * from events", conn)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _fit_activity_holdout(
    df: pd.DataFrame,
    target_col: str,
    chosen: dict[str, Any],
    horizon_minutes: int,
    paths: dict[str, Path],
) -> dict[str, Any]:
    feature_cols = ACTIVITY_NUMERIC_FEATURES + ACTIVITY_CATEGORICAL_FEATURES
    split = make_positive_count_holdout(df, target_col, min_positives=50)
    candidate = next(item for item in _candidate_models() if item.name == chosen["model"])
    pipeline = _make_activity_pipeline(candidate)
    pipeline.fit(df.loc[split.train_mask, feature_cols], df.loc[split.train_mask, target_col])
    scores = pipeline.predict_proba(df.loc[split.holdout_mask, feature_cols])[:, 1]
    holdout = df.loc[split.holdout_mask, ["target_time"]].copy()
    holdout["activity_probability"] = np.clip(scores, 0.0, 1.0)
    holdout["actual"] = df.loc[split.holdout_mask, target_col].astype(int).to_numpy()
    holdout.to_csv(paths["activity_predictions"], index=False)
    joblib.dump(pipeline, paths["activity_model"])
    return {
        "model_path": _relative(paths["activity_model"]),
        "predictions_path": _relative(paths["activity_predictions"]),
        "holdout_metrics": _binary_metrics(df.loc[split.holdout_mask, target_col], scores),
        "holdout_split": split.metadata,
        "holdout_predictions": holdout,
    }


def _fit_spatial_holdout(
    df: pd.DataFrame,
    target_col: str,
    chosen: dict[str, Any],
    activity_holdout_predictions: pd.DataFrame,
    paths: dict[str, Path],
) -> dict[str, Any]:
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    split = make_positive_count_holdout(df, target_col, min_positives=50)
    train_sample = sample_spatial_training_rows(
        df.loc[split.train_mask],
        target_col,
        negative_ratio=5,
        inactive_negative_fraction=0.02,
    )
    candidate = next(item for item in _candidate_models() if item.name == chosen["model"])
    pipeline = _make_spatial_pipeline(candidate)
    pipeline.fit(train_sample[feature_cols], train_sample[target_col])
    spatial_scores = pipeline.predict_proba(df.loc[split.holdout_mask, feature_cols])[:, 1]
    default_activity = (
        float(activity_holdout_predictions["activity_probability"].mean())
        if not activity_holdout_predictions.empty
        else 1.0
    )
    activity_scores = _probabilities_for_time(
        df.loc[split.holdout_mask, "target_time"],
        activity_holdout_predictions,
        default_activity,
    )
    base = df.loc[
        split.holdout_mask,
        ["target_time", "zone_id", "district", "region", "zone_lat", "zone_lng"],
    ].copy()
    base["spatial_probability"] = np.clip(spatial_scores, 0.0, 1.0)
    base["actual"] = df.loc[split.holdout_mask, target_col].astype(int).to_numpy()
    predictions = combine_activity_and_spatial_scores(base, activity_scores)
    predictions.to_csv(paths["spatial_predictions"], index=False)
    joblib.dump(pipeline, paths["spatial_model"])
    return {
        "model_path": _relative(paths["spatial_model"]),
        "predictions_path": _relative(paths["spatial_predictions"]),
        "holdout_metrics": _score_spatial_predictions(
            df.loc[split.holdout_mask, target_col],
            predictions["spatial_probability"].to_numpy(),
            groups={
                "district": df.loc[split.holdout_mask, "district"],
                "region": df.loc[split.holdout_mask, "region"],
            },
        ),
        "holdout_split": split.metadata,
        "training_rows_sampled": int(len(train_sample)),
    }


def run_two_stage_horizon(
    events: list[dict],
    horizon_minutes: int,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> dict[str, Any]:
    slug = horizon_slug(horizon_minutes)
    paths = _artifact_paths(slug)
    target_col = target_for_horizon(horizon_minutes)
    activity_target = activity_target_for_horizon(horizon_minutes)

    activity_df = build_activity_training_data(
        events,
        horizon_minutes=horizon_minutes,
        lookback_hours=24 * 7,
        resolution=resolution,
    )
    spatial_df = build_zone_ranking_training_data(
        events,
        forecast_hours=max(1, horizon_minutes // 60),
        horizon_minutes=horizon_minutes,
        target_col=target_col,
        lookback_days=7,
        resolution=resolution,
    )
    if activity_df.empty or spatial_df.empty:
        raise RuntimeError("Two-stage feature tables are empty.")

    activity_folds, activity_summary = _evaluate_activity_candidates(
        activity_df,
        activity_target,
        horizon_minutes,
    )
    spatial_folds, spatial_summary = _evaluate_spatial_candidates(
        spatial_df,
        target_col,
        horizon_minutes,
    )
    activity_chosen = _select_model(activity_summary, "activity")
    spatial_chosen = _select_model(spatial_summary, "spatial")

    activity_holdout = _fit_activity_holdout(
        activity_df,
        activity_target,
        activity_chosen,
        horizon_minutes,
        paths,
    )
    spatial_holdout = _fit_spatial_holdout(
        spatial_df,
        target_col,
        spatial_chosen,
        activity_holdout["holdout_predictions"],
        paths,
    )

    activity_folds.to_csv(paths["activity_folds"], index=False)
    spatial_folds.to_csv(paths["spatial_folds"], index=False)
    activity_metadata = {
        "stage": "activity",
        "resolution": resolution,
        "horizon_minutes": horizon_minutes,
        "horizon_slug": slug,
        "target_col": activity_target,
        "chosen_model": activity_chosen,
        "fold_summary": activity_summary.to_dict("records"),
        "folds_path": _relative(paths["activity_folds"]),
        "features": ACTIVITY_NUMERIC_FEATURES + ACTIVITY_CATEGORICAL_FEATURES,
        "training_rows": int(len(activity_df)),
        **{key: value for key, value in activity_holdout.items() if key != "holdout_predictions"},
    }
    spatial_metadata = {
        "stage": "spatial",
        "resolution": resolution,
        "horizon_minutes": horizon_minutes,
        "horizon_slug": slug,
        "target_col": target_col,
        "chosen_model": spatial_chosen,
        "fold_summary": spatial_summary.to_dict("records"),
        "folds_path": _relative(paths["spatial_folds"]),
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "training_rows": int(len(spatial_df)),
        **spatial_holdout,
    }
    paths["activity_metadata"].write_text(json.dumps(activity_metadata, indent=2), encoding="utf-8")
    paths["spatial_metadata"].write_text(json.dumps(spatial_metadata, indent=2), encoding="utf-8")

    return {
        "horizon_minutes": horizon_minutes,
        "horizon": slug,
        "activity_model": activity_metadata,
        "spatial_model": spatial_metadata,
        "activity_metadata_path": _relative(paths["activity_metadata"]),
        "spatial_metadata_path": _relative(paths["spatial_metadata"]),
        "activity_predictions_path": _relative(paths["activity_predictions"]),
        "predictions_path": _relative(paths["spatial_predictions"]),
    }


def write_two_stage_summary(metadata: list[dict[str, Any]], path: Path = SUMMARY_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in metadata:
        horizon_minutes = int(item["horizon_minutes"])
        activity = item["activity_model"]
        spatial = item["spatial_model"]
        activity_metrics = activity.get("holdout_metrics", {})
        spatial_metrics = spatial.get("holdout_metrics", {})
        activity_split = activity.get("holdout_split", {})
        spatial_split = spatial.get("holdout_split", {})
        rows.append(
            {
                "horizon_minutes": horizon_minutes,
                "horizon": horizon_slug(horizon_minutes),
                "model_family": "two_stage",
                "activity_model": activity["chosen_model"]["model"],
                "spatial_model": spatial["chosen_model"]["model"],
                "activity_average_precision": activity_metrics.get("average_precision", 0.0),
                "activity_roc_auc": activity_metrics.get("roc_auc", 0.0),
                "activity_brier_score": activity_metrics.get("brier_score", 0.0),
                "activity_holdout_rows": activity_split.get("holdout_rows", 0),
                "activity_holdout_positives": activity_split.get("holdout_positives", 0),
                "activity_holdout_start": activity_split.get("holdout_start", ""),
                "activity_holdout_end": activity_split.get("holdout_end", ""),
                "spatial_precision_at_20": spatial_metrics.get("precision_at_20", 0.0),
                "spatial_precision_at_50": spatial_metrics.get("precision_at_50", 0.0),
                "spatial_average_precision": spatial_metrics.get("average_precision", 0.0),
                "spatial_top_decile_lift": spatial_metrics.get("top_decile_lift", 0.0),
                "spatial_holdout_rows": spatial_split.get("holdout_rows", 0),
                "spatial_holdout_positives": spatial_split.get("holdout_positives", 0),
                "spatial_holdout_start": spatial_split.get("holdout_start", ""),
                "spatial_holdout_end": spatial_split.get("holdout_end", ""),
                "activity_metadata_path": item["activity_metadata_path"],
                "spatial_metadata_path": item["spatial_metadata_path"],
                "activity_predictions_path": item.get("activity_predictions_path", ""),
                "predictions_path": item["predictions_path"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("horizon_minutes")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame.to_dict("records")


def _write_report(rows: list[dict[str, Any]], report_path: Path = REPORT_PATH) -> None:
    frame = pd.DataFrame(rows)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Ghost Sweep Two-Stage Risk</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ef; padding: 8px; text-align: right; }}
    th {{ background: #eef3f9; }}
    td:first-child, th:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Ghost Sweep Two-Stage Risk</h1>
  <p>Stage 1 estimates whether Hong Kong has near-term activity. Stage 2 ranks road-access zones conditional on activity.</p>
  {frame.to_html(index=False)}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")


def run_two_stage_experiment(
    horizons: list[int] | None = None,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> list[dict[str, Any]]:
    events = _load_events().to_dict("records")
    results = [
        run_two_stage_horizon(events, horizon_minutes=horizon, resolution=resolution)
        for horizon in (horizons or HORIZONS)
    ]
    rows = write_two_stage_summary(results, SUMMARY_PATH)
    _write_report(rows, REPORT_PATH)
    return results


def main() -> None:
    results = run_two_stage_experiment()
    print(json.dumps(write_two_stage_summary(results, SUMMARY_PATH), indent=2))
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
