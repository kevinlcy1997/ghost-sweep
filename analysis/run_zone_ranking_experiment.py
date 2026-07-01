"""Run H3 zone-ranking experiments and emit MLflow/report artifacts."""

from __future__ import annotations

import html
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghost_db import GhostDB
from ghost_ranking_features import build_zone_ranking_training_data
from ghost_ranking_metrics import precision_at_k, recall_at_k, top_decile_lift
from ghost_zones import DEFAULT_H3_RESOLUTION, h3_zone_polygon


DB_PATH = ROOT / "ghost_alerts.db"
OUTPUT_DIR = ROOT / "analysis"
REPORT_PATH = OUTPUT_DIR / f"zone_ranking_report_{datetime.now():%Y%m%d_%H%M%S}.html"
PREDICTIONS_PATH = OUTPUT_DIR / "zone_predictions_latest.csv"
FEATURE_RANK_PATH = OUTPUT_DIR / "zone_feature_ranking_latest.csv"
MODEL_PATH = OUTPUT_DIR / "best_zone_model.joblib"
GEOJSON_PATH = ROOT / "ghost_zone_forecast.geojson"
MLFLOW_TRACKING_DB = OUTPUT_DIR / "mlflow_tracking.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_TRACKING_DB.as_posix()}"
MLFLOW_ARTIFACT_ROOT = OUTPUT_DIR / "mlruns"
MLFLOW_EXPERIMENT_NAME = "ghost-sweep-zone-ranking"

NUMERIC_FEATURES = [
    "zone_lat",
    "zone_lng",
    "hour",
    "day_of_week",
    "is_weekend",
    "zone_event_count_1h",
    "zone_event_count_3h",
    "zone_event_count_24h",
    "zone_event_count_7d",
    "zone_hours_since_last_event",
    "district_event_count_3h",
    "district_event_count_24h",
    "district_active_zones_24h",
    "zone_same_hour_rate",
    "district_same_hour_rate",
    "is_urban_core",
    "zone_3h_to_24h_ratio",
    "zone_24h_to_7d_ratio",
    "district_3h_to_24h_ratio",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "neighbor_event_count_3h",
    "neighbor_event_count_24h",
    "neighbor_event_count_7d",
    "neighbor_active_zones_24h",
    "ring2_event_count_24h",
    "ring2_event_count_7d",
    "ring2_active_zones_24h",
    "ring2_to_ring1_24h_ratio",
    "distance_to_nearest_event_3h_m",
    "distance_to_nearest_event_24h_m",
    "distance_to_district_recent_centroid_24h_m",
    "zone_24h_share_of_district",
    "zone_7d_rank_in_district",
    "zone_same_hour_percentile_in_district",
    "nearest_road_m",
    "road_segment_count",
    "road_source_mismatch",
    "has_drivable_road",
]
CATEGORICAL_FEATURES = ["district", "region", "hour_bucket"]
TARGET = "alert_next_2h"


def _db_counts() -> dict[str, object]:
    conn = sqlite3.connect(DB_PATH)
    return {
        "sightings": conn.execute("SELECT COUNT(*) FROM sightings").fetchone()[0],
        "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        "date_range": conn.execute("SELECT MIN(create_dt), MAX(create_dt) FROM sightings").fetchone(),
    }


def _preprocessor() -> ColumnTransformer:
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
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _score(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    pipe = Pipeline([("pre", _preprocessor()), ("model", model)])
    pipe.fit(X_train, y_train)
    if hasattr(pipe, "predict_proba"):
        score = pipe.predict_proba(X_test)[:, 1]
    else:
        score = pipe.decision_function(X_test)
    return {
        "model": name,
        "pipeline": pipe,
        "roc_auc": roc_auc_score(y_test, score) if y_test.nunique() > 1 else 0.0,
        "average_precision": average_precision_score(y_test, score),
        "precision_at_10": precision_at_k(y_test, score, 10),
        "precision_at_20": precision_at_k(y_test, score, 20),
        "recall_at_20": recall_at_k(y_test, score, 20),
        "top_decile_lift": top_decile_lift(y_test, score),
    }


def _historical_baseline(df: pd.DataFrame, train_mask: pd.Series, test_mask: pd.Series) -> dict:
    train = df.loc[train_mask]
    test = df.loc[test_mask].copy()
    district_rates = train.groupby("district")[TARGET].mean()
    global_rate = float(train[TARGET].mean())
    score = test["district"].map(district_rates).fillna(global_rate).to_numpy()
    y_test = test[TARGET].astype(int)
    return {
        "model": "district historical baseline",
        "pipeline": None,
        "roc_auc": roc_auc_score(y_test, score) if y_test.nunique() > 1 else 0.0,
        "average_precision": average_precision_score(y_test, score),
        "precision_at_10": precision_at_k(y_test, score, 10),
        "precision_at_20": precision_at_k(y_test, score, 20),
        "recall_at_20": recall_at_k(y_test, score, 20),
        "top_decile_lift": top_decile_lift(y_test, score),
    }


def _feature_ranking(best_pipeline: Pipeline) -> pd.DataFrame:
    model = best_pipeline.named_steps["model"]
    pre = best_pipeline.named_steps["pre"]
    names = pre.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        values = np.zeros(len(names))
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(30)
    )


def _write_geojson(predictions: pd.DataFrame, geojson_path: Path = GEOJSON_PATH) -> None:
    latest_time = predictions["target_time"].max()
    latest = predictions[predictions["target_time"] == latest_time].copy()
    latest = latest.sort_values("score", ascending=False).reset_index(drop=True)
    latest["risk_rank"] = latest.index + 1
    features = []
    for row in latest.head(200).to_dict("records"):
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [h3_zone_polygon(row["zone_id"])],
                },
                "properties": {
                    "zone_id": row["zone_id"],
                    "score": float(row["score"]),
                    "risk_rank": int(row["risk_rank"]),
                    "district": row["district"],
                    "region": row["region"],
                    "top_feature_reason": "recent district and zone activity",
                    "recent_events_24h": int(row["zone_event_count_24h"]),
                    "target_time": str(row["target_time"]),
                },
            }
        )
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def _write_report(
    metrics: list[dict],
    counts: dict,
    df: pd.DataFrame,
    feature_rank: pd.DataFrame,
    report_path: Path = REPORT_PATH,
) -> None:
    public_metrics = [{k: v for k, v in row.items() if k != "pipeline"} for row in metrics]
    metrics_df = pd.DataFrame(public_metrics).sort_values("precision_at_20", ascending=False)
    top_zones = (
        df.groupby(["zone_id", "district", "region"], dropna=False)
        .agg(rows=("zone_id", "size"), positives=(TARGET, "sum"))
        .reset_index()
        .sort_values("positives", ascending=False)
        .head(20)
    )
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Ghost Sweep Zone Ranking</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; background: #f7f8fb; }}
main {{ max-width: 1180px; margin: 0 auto; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th,td {{ border-bottom: 1px solid #dfe5ef; padding: 8px; text-align: left; font-size: 13px; }}
th {{ background: #eef3f9; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
.card {{ background:white; border:1px solid #dfe5ef; border-radius:8px; padding:14px; }}
.metric {{ font-size:24px; font-weight:700; margin-top:4px; }}
</style></head><body><main>
<h1>Ghost Sweep Zone Ranking Experiment</h1>
<p>Target: rank H3 road-zone candidates by at least one alert in the next 2 hours.</p>
<div class="grid">
<div class="card">Sightings<div class="metric">{counts["sightings"]}</div></div>
<div class="card">Cleaned events<div class="metric">{counts["events"]}</div></div>
<div class="card">Training rows<div class="metric">{len(df)}</div></div>
<div class="card">Positive rate<div class="metric">{df[TARGET].mean():.2%}</div></div>
</div>
<p>Data range: {html.escape(str(counts["date_range"][0]))} to {html.escape(str(counts["date_range"][1]))}.</p>
<h2>Model Selection</h2>
{metrics_df.to_html(index=False)}
<h2>Top Historical Positive Zones</h2>
{top_zones.to_html(index=False)}
<h2>Feature Importance</h2>
{feature_rank.to_html(index=False)}
<p>Artifacts: <code>{PREDICTIONS_PATH.name}</code>, <code>{FEATURE_RANK_PATH.name}</code>, <code>{MODEL_PATH.name}</code>, <code>{GEOJSON_PATH.name}</code>.</p>
</main></body></html>"""
    report_path.write_text(html_doc, encoding="utf-8")


def _ensure_mlflow_experiment() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACT_ROOT.as_uri(),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def run_experiment(
    resolution: int = DEFAULT_H3_RESOLUTION,
    write_latest_alias: bool = True,
) -> dict[str, object]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_path = OUTPUT_DIR / f"zone_predictions_res{resolution}_latest.csv"
    feature_rank_path = OUTPUT_DIR / f"zone_feature_ranking_res{resolution}_latest.csv"
    model_path = OUTPUT_DIR / f"best_zone_model_res{resolution}.joblib"
    geojson_path = ROOT / f"ghost_zone_forecast_res{resolution}.geojson"
    report_path = OUTPUT_DIR / f"zone_ranking_report_res{resolution}_{timestamp}.html"

    _ensure_mlflow_experiment()
    events = GhostDB(str(DB_PATH)).get_all_events()
    df = build_zone_ranking_training_data(events, forecast_hours=2, lookback_days=7, resolution=resolution)
    if df.empty:
        raise RuntimeError("Zone ranking feature table is empty")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    unique_times = sorted(pd.to_datetime(df["target_time"]).unique())
    split_time = unique_times[int(len(unique_times) * 0.8)]
    train_mask = pd.to_datetime(df["target_time"]) <= split_time
    test_mask = ~train_mask
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, TARGET].astype(int)
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, TARGET].astype(int)

    models = {
        "logistic regression balanced": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "random forest balanced": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "lightgbm balanced": LGBMClassifier(
            n_estimators=160,
            learning_rate=0.05,
            num_leaves=15,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        ),
        "dummy prior": DummyClassifier(strategy="prior"),
    }

    metrics = [_historical_baseline(df, train_mask, test_mask)]
    with mlflow.start_run(run_name=f"zone-ranking-{datetime.now():%Y%m%d-%H%M%S}") as parent:
        mlflow.log_param("target", TARGET)
        mlflow.log_param("forecast_hours", 2)
        mlflow.log_param("lookback_days", 7)
        mlflow.log_param("zone_type", f"h3_resolution_{resolution}")
        mlflow.log_param("h3_resolution", resolution)
        mlflow.log_param("rows", len(df))
        mlflow.log_param("train_rows", int(train_mask.sum()))
        mlflow.log_param("test_rows", int(test_mask.sum()))
        mlflow.log_param("positive_rate", float(df[TARGET].mean()))

        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                result = _score(name, model, X_train, y_train, X_test, y_test)
                for key, value in result.items():
                    if key not in {"model", "pipeline"}:
                        mlflow.log_metric(key, float(value))
                mlflow.log_param("model", name)
                metrics.append(result)

        best = max(metrics, key=lambda row: (row["precision_at_20"], row["average_precision"]))
        if best["pipeline"] is None:
            best_model = _score(
                "best fallback logistic regression balanced",
                models["logistic regression balanced"],
                X_train,
                y_train,
                X_test,
                y_test,
            )["pipeline"]
        else:
            best_model = best["pipeline"]
        joblib.dump(best_model, model_path)

        predictions = df.loc[test_mask, ["target_time", "zone_id", "district", "region"] + NUMERIC_FEATURES].copy()
        predictions["score"] = best_model.predict_proba(df.loc[test_mask, feature_cols])[:, 1]
        predictions["actual"] = y_test.to_numpy()
        predictions.to_csv(predictions_path, index=False)
        _write_geojson(predictions, geojson_path)

        feature_rank = _feature_ranking(best_model)
        feature_rank.to_csv(feature_rank_path, index=False)
        counts = _db_counts()
        _write_report(metrics, counts, df, feature_rank, report_path)

        mlflow.log_param("best_model", best["model"])
        for key in [
            "precision_at_10",
            "precision_at_20",
            "recall_at_20",
            "top_decile_lift",
            "average_precision",
            "roc_auc",
            ]:
            mlflow.log_metric(key, float(best[key]))
        for artifact in [report_path, predictions_path, feature_rank_path, model_path, geojson_path]:
            mlflow.log_artifact(str(artifact))
        survey = ROOT / "analysis" / "geo" / "hk_zone_summary.csv"
        if survey.exists():
            mlflow.log_artifact(str(survey))
        print(report_path)
    if write_latest_alias:
        predictions.to_csv(PREDICTIONS_PATH, index=False)
        feature_rank.to_csv(FEATURE_RANK_PATH, index=False)
        joblib.dump(best_model, MODEL_PATH)
        GEOJSON_PATH.write_text(geojson_path.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "resolution": resolution,
        "best_model": best["model"],
        "metrics": best,
        "predictions_path": str(predictions_path.relative_to(ROOT)),
        "feature_rank_path": str(feature_rank_path.relative_to(ROOT)),
        "model_path": str(model_path.relative_to(ROOT)),
        "geojson_path": str(geojson_path.relative_to(ROOT)),
        "report_path": str(report_path.relative_to(ROOT)),
        "training_rows": int(len(df)),
        "active_zones": int(df["zone_id"].nunique()),
        "median_events_per_zone": float(df.groupby("zone_id").size().median()),
        "one_off_zone_rate": float((df.groupby("zone_id").size() == 1).mean()),
        "positive_rate": float(df[TARGET].mean()),
    }


if __name__ == "__main__":
    run_experiment()
