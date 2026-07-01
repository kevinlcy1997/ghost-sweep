from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.run_two_stage_experiment import (  # noqa: E402
    ACTIVITY_CATEGORICAL_FEATURES,
    ACTIVITY_NUMERIC_FEATURES,
    activity_target_for_horizon,
    horizon_slug,
)
from analysis.run_zone_ranking_experiment import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DB_PATH,
    NUMERIC_FEATURES,
)
from ghost_activity_features import build_activity_training_data  # noqa: E402
from ghost_ranking_features import build_zone_ranking_training_data  # noqa: E402


OUTPUT_DIR = ROOT / "analysis" / "feature_correlation"
HORIZONS = [30, 60, 120]


def _load_events() -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("select * from events", conn).to_dict("records")


def _safe_abs_corr(frame: pd.DataFrame, features: list[str], target_col: str, method: str) -> pd.Series:
    values: dict[str, float] = {}
    target = frame[target_col].astype(float)
    for feature in features:
        if feature not in frame:
            continue
        series = pd.to_numeric(frame[feature], errors="coerce")
        if series.notna().sum() < 3 or series.nunique(dropna=True) <= 1:
            values[feature] = 0.0
            continue
        corr = series.corr(target, method=method)
        values[feature] = 0.0 if pd.isna(corr) else float(corr)
    return pd.Series(values, dtype=float)


def _mutual_info(frame: pd.DataFrame, features: list[str], target_col: str) -> pd.Series:
    usable = [feature for feature in features if feature in frame]
    if not usable or frame[target_col].nunique() < 2:
        return pd.Series({feature: 0.0 for feature in usable}, dtype=float)
    sample = frame[usable + [target_col]].copy()
    for feature in usable:
        sample[feature] = pd.to_numeric(sample[feature], errors="coerce")
        sample[feature] = sample[feature].fillna(sample[feature].median())
    y = sample[target_col].astype(int)
    values = mutual_info_classif(sample[usable], y, discrete_features=False, random_state=42)
    return pd.Series(dict(zip(usable, values)), dtype=float)


def _categorical_signal(frame: pd.DataFrame, features: list[str], target_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    global_rate = float(frame[target_col].mean()) if len(frame) else 0.0
    for feature in features:
        if feature not in frame:
            continue
        grouped = (
            frame.groupby(feature, dropna=False)[target_col]
            .agg(["count", "mean", "sum"])
            .reset_index()
            .sort_values(["mean", "count"], ascending=[False, False])
        )
        if grouped.empty:
            continue
        top = grouped.iloc[0]
        rows.append(
            {
                "feature": feature,
                "levels": int(grouped[feature].nunique(dropna=False)),
                "top_level": str(top[feature]),
                "top_level_rows": int(top["count"]),
                "top_level_positive_rate": float(top["mean"]),
                "lift_vs_global": float(top["mean"] / global_rate) if global_rate else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _target_correlation_table(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str,
    stage: str,
    horizon: str,
    subset: str,
) -> pd.DataFrame:
    pearson = _safe_abs_corr(frame, numeric_features, target_col, "pearson")
    spearman = _safe_abs_corr(frame, numeric_features, target_col, "spearman")
    mi = _mutual_info(frame, numeric_features, target_col)
    rows = []
    for feature in numeric_features:
        if feature not in frame:
            continue
        rows.append(
            {
                "stage": stage,
                "horizon": horizon,
                "subset": subset,
                "feature_type": "numeric",
                "feature": feature,
                "pearson": float(pearson.get(feature, 0.0)),
                "abs_pearson": abs(float(pearson.get(feature, 0.0))),
                "spearman": float(spearman.get(feature, 0.0)),
                "abs_spearman": abs(float(spearman.get(feature, 0.0))),
                "mutual_info": float(mi.get(feature, 0.0)),
                "rows": int(len(frame)),
                "positives": int(frame[target_col].sum()),
                "base_rate": float(frame[target_col].mean()) if len(frame) else 0.0,
            }
        )
    cat = _categorical_signal(frame, categorical_features, target_col)
    for row in cat.to_dict("records"):
        rows.append(
            {
                "stage": stage,
                "horizon": horizon,
                "subset": subset,
                "feature_type": "categorical",
                "feature": row["feature"],
                "pearson": np.nan,
                "abs_pearson": np.nan,
                "spearman": np.nan,
                "abs_spearman": np.nan,
                "mutual_info": np.nan,
                "rows": int(len(frame)),
                "positives": int(frame[target_col].sum()),
                "base_rate": float(frame[target_col].mean()) if len(frame) else 0.0,
                **{key: value for key, value in row.items() if key != "feature"},
            }
        )
    return pd.DataFrame(rows)


def _feature_pair_correlations(
    frame: pd.DataFrame,
    numeric_features: list[str],
    stage: str,
    horizon: str,
    subset: str,
) -> pd.DataFrame:
    usable = [feature for feature in numeric_features if feature in frame]
    if len(usable) < 2:
        return pd.DataFrame()
    corr = frame[usable].corr(method="spearman").abs()
    rows = []
    for i, left in enumerate(usable):
        for right in usable[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= 0.85:
                rows.append(
                    {
                        "stage": stage,
                        "horizon": horizon,
                        "subset": subset,
                        "feature_a": left,
                        "feature_b": right,
                        "abs_spearman": float(value),
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)


def _active_windows(frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    active = frame.groupby("target_time")[target_col].transform("sum") > 0
    return frame.loc[active].copy()


def _write_report(target_corr: pd.DataFrame, pair_corr: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Feature Correlation Analysis",
        "",
        "Correlation is exploratory, not causal. For sparse binary targets, Spearman, mutual information, and active-window subsets are more informative than raw Pearson alone.",
        "",
        "## Strongest Target Associations",
    ]
    for (stage, horizon, subset), group in target_corr.groupby(["stage", "horizon", "subset"], sort=False):
        lines.append(f"### {stage} {horizon} {subset}")
        cols = ["feature", "feature_type", "abs_spearman", "mutual_info", "base_rate", "positives", "rows"]
        top = group[group["feature_type"] == "numeric"].sort_values(
            ["mutual_info", "abs_spearman"],
            ascending=[False, False],
        ).head(12)
        lines.append(top[cols].to_markdown(index=False))
        cat = group[group["feature_type"] == "categorical"]
        if not cat.empty:
            lines.append("")
            lines.append("Categorical signal:")
            lines.append(
                cat[
                    [
                        "feature",
                        "levels",
                        "top_level",
                        "top_level_rows",
                        "top_level_positive_rate",
                        "lift_vs_global",
                    ]
                ].to_markdown(index=False)
            )
        lines.append("")

    lines.extend(["## Highly Correlated Feature Pairs", ""])
    if pair_corr.empty:
        lines.append("No numeric feature pairs reached absolute Spearman correlation >= 0.85.")
    else:
        lines.append(pair_corr.head(50).to_markdown(index=False))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_feature_correlation_analysis() -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = _load_events()
    target_frames = []
    pair_frames = []

    for horizon_minutes in HORIZONS:
        slug = horizon_slug(horizon_minutes)
        activity_target = activity_target_for_horizon(horizon_minutes)
        activity = build_activity_training_data(
            events,
            horizon_minutes=horizon_minutes,
            lookback_hours=24 * 7,
        )
        target_frames.append(
            _target_correlation_table(
                activity,
                ACTIVITY_NUMERIC_FEATURES,
                ACTIVITY_CATEGORICAL_FEATURES,
                activity_target,
                "activity",
                slug,
                "all_hours",
            )
        )
        pair_frames.append(
            _feature_pair_correlations(
                activity,
                ACTIVITY_NUMERIC_FEATURES,
                "activity",
                slug,
                "all_hours",
            )
        )

        spatial_target = f"alert_next_{slug}"
        spatial = build_zone_ranking_training_data(
            events,
            forecast_hours=max(1, horizon_minutes // 60),
            horizon_minutes=horizon_minutes,
            target_col=spatial_target,
            lookback_days=7,
        )
        for subset_name, subset in [
            ("all_zone_hours", spatial),
            ("active_windows", _active_windows(spatial, spatial_target)),
        ]:
            target_frames.append(
                _target_correlation_table(
                    subset,
                    NUMERIC_FEATURES,
                    CATEGORICAL_FEATURES,
                    spatial_target,
                    "spatial",
                    slug,
                    subset_name,
                )
            )
            pair_frames.append(
                _feature_pair_correlations(
                    subset,
                    NUMERIC_FEATURES,
                    "spatial",
                    slug,
                    subset_name,
                )
            )

    target_corr = pd.concat(target_frames, ignore_index=True)
    pair_corr = pd.concat([frame for frame in pair_frames if not frame.empty], ignore_index=True)
    target_path = OUTPUT_DIR / "target_feature_correlation_latest.csv"
    pair_path = OUTPUT_DIR / "feature_pair_correlation_latest.csv"
    report_path = OUTPUT_DIR / "feature_correlation_report.md"
    target_corr.to_csv(target_path, index=False)
    pair_corr.to_csv(pair_path, index=False)
    _write_report(target_corr, pair_corr, report_path)
    return {
        "target_correlation": str(target_path.relative_to(ROOT)),
        "pair_correlation": str(pair_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }


def main() -> None:
    print(json.dumps(run_feature_correlation_analysis(), indent=2))


if __name__ == "__main__":
    main()
