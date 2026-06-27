"""Run Ghost Sweep feature/model experiments and emit an HTML report."""

from __future__ import annotations

import html
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghost_db import GhostDB
from ghost_features import build_training_data
from ghost_model import CATEGORICAL_COLS, FEATURE_COLS


DB_PATH = ROOT / "ghost_alerts.db"
REPORT_PATH = ROOT / "analysis" / f"ml_experiment_report_{datetime.now():%Y%m%d_%H%M%S}.html"
BEST_MODEL_PATH = ROOT / "analysis" / "best_experiment_model.joblib"
MLFLOW_TRACKING_DB = ROOT / "analysis" / "mlflow_tracking.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_TRACKING_DB.as_posix()}"
MLFLOW_ARTIFACT_ROOT = ROOT / "analysis" / "mlruns"
MLFLOW_EXPERIMENT_NAME = "ghost-sweep"
RECENT_EVENT_LIMIT = 500


def _db_counts() -> dict[str, object]:
    conn = sqlite3.connect(DB_PATH)
    counts: dict[str, object] = {}
    for table in ["sightings", "events", "poll_cycles"]:
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    counts["date_range"] = conn.execute(
        "SELECT MIN(create_dt), MAX(create_dt) FROM sightings"
    ).fetchone()
    return counts


def _preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )


def _score_model(name: str, pipe: Pipeline, X_train, y_train, X_test, y_test) -> dict[str, object]:
    pipe.fit(X_train, y_train)
    if hasattr(pipe, "predict_proba"):
        score = pipe.predict_proba(X_test)[:, 1]
    else:
        score = pipe.decision_function(X_test)
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, score) if len(set(y_test)) > 1 else np.nan,
        "average_precision": average_precision_score(y_test, score),
        "f1": f1_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "accuracy": accuracy_score(y_test, pred),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "pipeline": pipe,
    }


def _feature_scores(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    y = df["has_warden"].astype(int)
    for col in numeric_cols:
        x = df[[col]].fillna(df[col].median())
        rows.append(
            {
                "feature": col,
                "type": "numeric",
                "mutual_info": float(mutual_info_classif(x, y, random_state=42)[0]),
                "positive_rate_high": float(y[x[col] >= x[col].median()].mean()),
            }
        )
    for col in CATEGORICAL_COLS:
        rates = df.groupby(col, dropna=False)["has_warden"].agg(["count", "mean"])
        top = rates.sort_values(["mean", "count"], ascending=False).head(1)
        rows.append(
            {
                "feature": col,
                "type": "categorical",
                "mutual_info": np.nan,
                "positive_rate_high": float(top["mean"].iloc[0]) if not top.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mutual_info", "positive_rate_high"], ascending=False, na_position="last"
    )


def _table(rows: list[dict[str, object]] | pd.DataFrame, cols: list[str]) -> str:
    if isinstance(rows, pd.DataFrame):
        records = rows[cols].to_dict("records")
    else:
        records = rows
    head = "".join(f"<th>{html.escape(col)}</th>" for col in cols)
    body = []
    for row in records:
        cells = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACT_ROOT.as_uri(),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_events = GhostDB(str(DB_PATH)).get_all_events()
    events = sorted(all_events, key=lambda ev: ev.get("create_dt", ""))[-RECENT_EVENT_LIMIT:]
    df = build_training_data(events).reset_index(drop=True)
    numeric_cols = [c for c in FEATURE_COLS if c in df.columns]
    feature_cols = numeric_cols + CATEGORICAL_COLS
    df = df.dropna(subset=["has_warden"])

    split_at = int(len(df) * 0.8)
    X_train, X_test = df.loc[: split_at - 1, feature_cols], df.loc[split_at:, feature_cols]
    y_train = df.loc[: split_at - 1, "has_warden"].astype(int)
    y_test = df.loc[split_at:, "has_warden"].astype(int)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError("Train/test split lacks both classes; cannot evaluate classifiers.")

    pre = _preprocessor(numeric_cols)
    models = {
        "Dummy prior": DummyClassifier(strategy="prior"),
        "Logistic regression balanced": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "Random forest balanced": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
        "LightGBM balanced": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=20,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        ),
    }

    results = []
    with mlflow.start_run(run_name=f"experiment-{datetime.now():%Y%m%d-%H%M%S}") as parent_run:
        mlflow.log_param("db_path", str(DB_PATH.relative_to(ROOT)))
        mlflow.log_param("recent_event_limit", RECENT_EVENT_LIMIT)
        mlflow.log_param("cleaned_events_total", len(all_events))
        mlflow.log_param("training_rows", len(df))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("positive_rate", float(df["has_warden"].mean()))
        mlflow.log_text(
            json.dumps(
                {
                    "numeric_features": numeric_cols,
                    "categorical_features": CATEGORICAL_COLS,
                },
                indent=2,
            ),
            "feature_config.json",
        )
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                result = _score_model(
                    name,
                    Pipeline([("pre", pre), ("model", model)]),
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )
                for metric in [
                    "roc_auc",
                    "average_precision",
                    "f1",
                    "precision",
                    "recall",
                    "accuracy",
                ]:
                    mlflow.log_metric(metric, float(result[metric]))
                for count in ["tp", "fp", "tn", "fn"]:
                    mlflow.log_metric(count, int(result[count]))
                mlflow.log_param("model", name)
                results.append(result)

    best = max(results, key=lambda r: (r["roc_auc"], r["average_precision"]))
    joblib.dump(best["pipeline"], BEST_MODEL_PATH)
    public_results = [{k: v for k, v in row.items() if k != "pipeline"} for row in results]
    feature_rank = _feature_scores(df, numeric_cols).head(12)
    counts = _db_counts()

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pos_rate = float(df["has_warden"].mean())
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ghost Sweep ML Experiment Report</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; background: #f7f8fb; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 32px 24px 48px; }}
h1 {{ margin: 0 0 6px; font-size: 30px; }}
h2 {{ margin-top: 30px; font-size: 20px; }}
.muted {{ color: #637083; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 22px 0; }}
.card {{ background: white; border: 1px solid #dfe5ef; border-radius: 8px; padding: 16px; }}
.metric {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #dfe5ef; }}
th, td {{ padding: 9px 10px; border-bottom: 1px solid #edf1f6; text-align: left; font-size: 14px; }}
th {{ background: #eef3f9; color: #26364d; }}
.note {{ background: #fff7df; border: 1px solid #f0d98c; border-radius: 8px; padding: 14px 16px; }}
code {{ background: #eef3f9; padding: 2px 4px; border-radius: 4px; }}
</style>
</head>
<body><main>
<h1>Ghost Sweep ML Experiment Report</h1>
<div class="muted">Generated {generated} from <code>{html.escape(str(DB_PATH.name))}</code></div>
<div class="grid">
<div class="card"><div class="muted">Raw sightings</div><div class="metric">{counts["sightings"]}</div></div>
<div class="card"><div class="muted">Cleaned events</div><div class="metric">{counts["events"]}</div></div>
<div class="card"><div class="muted">Training rows</div><div class="metric">{len(df)}</div></div>
<div class="card"><div class="muted">Positive rate</div><div class="metric">{pos_rate:.1%}</div></div>
</div>
<div class="note">Source data covers {html.escape(str(counts["date_range"][0]))} to {html.escape(str(counts["date_range"][1]))}.
The JSON export is corrupt after the first complete alert block; sync used brace-balanced salvage of complete alert records.
Model experiments use the most recent {len(events)} cleaned events because full feature generation over {counts["events"]} events is too slow with the current historical feature builder.</div>
<h2>Model Selection</h2>
{_table(public_results, ["model", "roc_auc", "average_precision", "f1", "precision", "recall", "accuracy", "tp", "fp", "tn", "fn"])}
<p>Best holdout model by ROC-AUC then average precision: <strong>{html.escape(str(best["model"]))}</strong>.</p>
<p>Selected trained pipeline saved to <code>{html.escape(str(BEST_MODEL_PATH.relative_to(ROOT)))}</code>.</p>
<h2>Feature Selection Signals</h2>
{_table(feature_rank, ["feature", "type", "mutual_info", "positive_rate_high"])}
<h2>Experiment Configuration</h2>
<p>Features came from <code>ghost_features.build_training_data</code> and production <code>ghost_model.FEATURE_COLS</code>.
Evaluation used the first 80% of generated rows for training and the final 20% for holdout testing to preserve row-order chronology as much as the current feature table allows.</p>
<pre>{html.escape(json.dumps({"numeric_features": numeric_cols, "categorical_features": CATEGORICAL_COLS}, indent=2))}</pre>
</main></body></html>"""
    REPORT_PATH.write_text(html_doc, encoding="utf-8")
    with mlflow.start_run(run_id=parent_run.info.run_id):
        mlflow.log_param("best_model", str(best["model"]))
        mlflow.log_metric("best_roc_auc", float(best["roc_auc"]))
        mlflow.log_metric("best_average_precision", float(best["average_precision"]))
        mlflow.log_metric("best_f1", float(best["f1"]))
        feature_rank.to_csv(ROOT / "analysis" / "feature_ranking_latest.csv", index=False)
        mlflow.log_artifact(str(REPORT_PATH))
        mlflow.log_artifact(str(ROOT / "analysis" / "feature_ranking_latest.csv"))
        sync_verify = ROOT / "analysis" / "sync_verify.txt"
        if sync_verify.exists():
            mlflow.log_artifact(str(sync_verify))
        mlflow.log_artifact(str(BEST_MODEL_PATH), artifact_path="best_model")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
