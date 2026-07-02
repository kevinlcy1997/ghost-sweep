from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = ("30m", "1h", "2h")


def _score_column(frame: pd.DataFrame) -> str:
    for column in ("score", "probability", "spatial_probability"):
        if column in frame:
            return column
    raise ValueError("Prediction frame must include score, probability, or spatial_probability.")


def _rank_column(frame: pd.DataFrame, score_col: str, rank_scope: str) -> pd.Series:
    if rank_scope == "artifact" and "rank" in frame:
        return frame["rank"].astype(int)
    if rank_scope == "per_target_time":
        return (
            frame.groupby("target_time", sort=False)[score_col]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    order = frame.sort_values([score_col, "zone_id"], ascending=[False, True], kind="mergesort").index
    ranks = pd.Series(index=order, data=np.arange(1, len(order) + 1), dtype=int)
    return ranks.sort_index()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace(0, np.nan)).fillna(0.0)


def summarize_horizon(
    predictions: pd.DataFrame,
    horizon: str,
    k: int,
    rank_scope: str,
) -> dict[str, float | int | str]:
    score_col = _score_column(predictions)
    frame = predictions.copy()
    frame["rank"] = _rank_column(frame, score_col, rank_scope)
    frame["target_time"] = pd.to_datetime(frame["target_time"])
    frame["actual"] = frame["actual"].astype(int)

    positives = frame[frame["actual"] == 1]
    topk = frame[frame["rank"] <= k]
    topk_false_positives = topk[topk["actual"] == 0]
    topk_true_positives = topk[topk["actual"] == 1]
    missed_positives = positives[positives["rank"] > k]

    return {
        "horizon": horizon,
        "rank_scope": rank_scope,
        "rows": int(len(frame)),
        "target_times": int(frame["target_time"].nunique()),
        "positives": int(len(positives)),
        "base_rate": float(frame["actual"].mean()),
        "score_min": float(frame[score_col].min()),
        "score_p50": float(frame[score_col].median()),
        "score_p95": float(frame[score_col].quantile(0.95)),
        "score_max": float(frame[score_col].max()),
        "positive_rank_p50": float(positives["rank"].median()) if len(positives) else 0.0,
        "positive_rank_p90": float(positives["rank"].quantile(0.90)) if len(positives) else 0.0,
        "positive_rank_p95": float(positives["rank"].quantile(0.95)) if len(positives) else 0.0,
        f"top{k}_true_positives": int(len(topk_true_positives)),
        f"top{k}_false_positives": int(len(topk_false_positives)),
        f"top{k}_missed_positives": int(len(missed_positives)),
        f"top{k}_precision": float(len(topk_true_positives) / len(topk)) if len(topk) else 0.0,
        f"top{k}_recall": float(len(topk_true_positives) / len(positives)) if len(positives) else 0.0,
    }


def summarize_by_group(
    predictions: pd.DataFrame,
    horizon: str,
    group_col: str,
    k: int,
    rank_scope: str,
) -> pd.DataFrame:
    score_col = _score_column(predictions)
    frame = predictions.copy()
    frame["rank"] = _rank_column(frame, score_col, rank_scope)
    frame["actual"] = frame["actual"].astype(int)
    frame["is_topk"] = frame["rank"] <= k
    frame["topk_tp"] = ((frame["actual"] == 1) & frame["is_topk"]).astype(int)
    frame["topk_fp"] = ((frame["actual"] == 0) & frame["is_topk"]).astype(int)
    frame["missed_positive"] = ((frame["actual"] == 1) & ~frame["is_topk"]).astype(int)

    grouped = (
        frame.groupby(group_col, dropna=False)
        .agg(
            rows=("actual", "size"),
            positives=("actual", "sum"),
            topk_rows=("is_topk", "sum"),
            topk_tp=("topk_tp", "sum"),
            topk_fp=("topk_fp", "sum"),
            missed_positives=("missed_positive", "sum"),
            mean_score=(score_col, "mean"),
            max_score=(score_col, "max"),
            best_positive_rank=("rank", lambda rank: float(rank[frame.loc[rank.index, "actual"] == 1].min()) if (frame.loc[rank.index, "actual"] == 1).any() else np.nan),
        )
        .reset_index()
    )
    grouped["horizon"] = horizon
    grouped["rank_scope"] = rank_scope
    grouped["base_rate"] = _safe_divide(grouped["positives"], grouped["rows"])
    grouped[f"top{k}_precision"] = _safe_divide(grouped["topk_tp"], grouped["topk_rows"])
    grouped[f"top{k}_recall"] = _safe_divide(grouped["topk_tp"], grouped["positives"])
    grouped[f"top{k}_fp_share"] = _safe_divide(grouped["topk_fp"], pd.Series([grouped["topk_fp"].sum()] * len(grouped)))
    return grouped.sort_values(["missed_positives", "topk_fp", "positives"], ascending=False)


def summarize_by_target_time(
    predictions: pd.DataFrame,
    horizon: str,
    k: int,
    rank_scope: str,
) -> pd.DataFrame:
    score_col = _score_column(predictions)
    frame = predictions.copy()
    frame["rank"] = _rank_column(frame, score_col, rank_scope)
    frame["target_time"] = pd.to_datetime(frame["target_time"])
    frame["hour"] = frame["target_time"].dt.hour
    frame["dayofweek"] = frame["target_time"].dt.dayofweek
    frame["actual"] = frame["actual"].astype(int)
    frame["is_topk"] = frame["rank"] <= k
    frame["topk_tp"] = ((frame["actual"] == 1) & frame["is_topk"]).astype(int)
    frame["topk_fp"] = ((frame["actual"] == 0) & frame["is_topk"]).astype(int)
    frame["missed_positive"] = ((frame["actual"] == 1) & ~frame["is_topk"]).astype(int)

    grouped = (
        frame.groupby(["target_time", "hour", "dayofweek"], sort=True)
        .agg(
            rows=("actual", "size"),
            positives=("actual", "sum"),
            topk_tp=("topk_tp", "sum"),
            topk_fp=("topk_fp", "sum"),
            missed_positives=("missed_positive", "sum"),
            top_score=(score_col, "max"),
            median_score=(score_col, "median"),
        )
        .reset_index()
    )
    grouped["horizon"] = horizon
    grouped["rank_scope"] = rank_scope
    grouped[f"top{k}_precision"] = grouped["topk_tp"] / k
    grouped[f"top{k}_recall"] = _safe_divide(grouped["topk_tp"], grouped["positives"])
    return grouped.sort_values(["missed_positives", "topk_fp"], ascending=False)


def write_error_analysis(input_dir: Path, output_dir: Path, k: int) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon_rows: list[dict[str, float | int | str]] = []
    district_rows: list[pd.DataFrame] = []
    region_rows: list[pd.DataFrame] = []
    time_rows: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        path = input_dir / f"spatial_zone_predictions_{horizon}_latest.csv"
        predictions = pd.read_csv(path)
        for rank_scope in ("artifact", "per_target_time"):
            horizon_rows.append(summarize_horizon(predictions, horizon, k, rank_scope))
            district_rows.append(summarize_by_group(predictions, horizon, "district", k, rank_scope))
            region_rows.append(summarize_by_group(predictions, horizon, "region", k, rank_scope))
            time_rows.append(summarize_by_target_time(predictions, horizon, k, rank_scope))

    outputs = {
        "summary": output_dir / "spatial_model_error_summary_latest.csv",
        "district": output_dir / "spatial_model_error_by_district_latest.csv",
        "region": output_dir / "spatial_model_error_by_region_latest.csv",
        "target_time": output_dir / "spatial_model_error_by_target_time_latest.csv",
    }
    pd.DataFrame(horizon_rows).to_csv(outputs["summary"], index=False)
    pd.concat(district_rows, ignore_index=True).to_csv(outputs["district"], index=False)
    pd.concat(region_rows, ignore_index=True).to_csv(outputs["region"], index=False)
    pd.concat(time_rows, ignore_index=True).to_csv(outputs["target_time"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze spatial model ranking errors.")
    parser.add_argument("--input-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()

    outputs = write_error_analysis(args.input_dir, args.output_dir, args.k)
    summary = pd.read_csv(outputs["summary"])
    print(summary.to_string(index=False))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
