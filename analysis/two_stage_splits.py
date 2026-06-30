from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PurgedSplit:
    train_mask: pd.Series
    validation_mask: pd.Series
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HoldoutSplit:
    train_mask: pd.Series
    holdout_mask: pd.Series
    metadata: dict[str, Any]


def _time_series(df: pd.DataFrame, time_col: str) -> pd.Series:
    if time_col not in df:
        raise ValueError(f"Missing required time column: {time_col}")
    return pd.to_datetime(df[time_col])


def make_purged_rolling_splits(
    df: pd.DataFrame,
    horizon_minutes: int,
    n_splits: int = 4,
    time_col: str = "target_time",
) -> list[PurgedSplit]:
    frame = df.copy()
    frame[time_col] = _time_series(frame, time_col)
    times = pd.Series(sorted(frame[time_col].dropna().unique()))
    if len(times) < n_splits + 3:
        raise ValueError("Need more target_time values for purged rolling splits.")

    validation_size = max(1, len(times) // (n_splits + 2))
    purge = pd.Timedelta(minutes=horizon_minutes)
    splits: list[PurgedSplit] = []
    for split_index in range(n_splits):
        validation_start_index = len(times) - validation_size * (n_splits - split_index)
        validation_end_index = min(len(times), validation_start_index + validation_size)
        if validation_start_index <= 0 or validation_end_index <= validation_start_index:
            continue

        validation_times = times.iloc[validation_start_index:validation_end_index]
        validation_start = pd.Timestamp(validation_times.min())
        train_cutoff = validation_start - purge
        train_mask = frame[time_col] <= train_cutoff
        validation_mask = frame[time_col].isin(set(validation_times))
        if not train_mask.any() or not validation_mask.any():
            continue

        metadata = {
            "fold": len(splits) + 1,
            "purge_minutes": int(horizon_minutes),
            "train_start": frame.loc[train_mask, time_col].min().isoformat(),
            "train_end": frame.loc[train_mask, time_col].max().isoformat(),
            "validation_start": frame.loc[validation_mask, time_col].min().isoformat(),
            "validation_end": frame.loc[validation_mask, time_col].max().isoformat(),
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
        }
        splits.append(
            PurgedSplit(
                train_mask=train_mask,
                validation_mask=validation_mask,
                metadata=metadata,
            )
        )
    if not splits:
        raise ValueError("Could not construct non-empty purged rolling splits.")
    return splits


def make_positive_count_holdout(
    df: pd.DataFrame,
    target_col: str,
    min_positives: int = 50,
    time_col: str = "target_time",
) -> HoldoutSplit:
    if target_col not in df:
        raise ValueError(f"Missing required target column: {target_col}")

    frame = df.copy()
    frame[time_col] = _time_series(frame, time_col)
    by_time = (
        frame.groupby(time_col, sort=True)[target_col]
        .sum()
        .reset_index()
        .sort_values(time_col, ascending=False)
    )

    cumulative = 0
    selected_times: list[pd.Timestamp] = []
    for row in by_time.itertuples(index=False):
        selected_times.append(pd.Timestamp(getattr(row, time_col)))
        cumulative += int(getattr(row, target_col))
        if cumulative >= min_positives:
            break

    selected_set = set(selected_times)
    holdout_mask = frame[time_col].isin(selected_set)
    train_mask = ~holdout_mask
    positives = int(frame.loc[holdout_mask, target_col].sum())
    rows = int(holdout_mask.sum())
    metadata = {
        "min_positives": int(min_positives),
        "met_min_positives": bool(positives >= min_positives),
        "holdout_start": frame.loc[holdout_mask, time_col].min().isoformat() if rows else None,
        "holdout_end": frame.loc[holdout_mask, time_col].max().isoformat() if rows else None,
        "holdout_rows": rows,
        "holdout_positives": positives,
        "holdout_base_rate": float(positives / rows) if rows else 0.0,
        "train_rows": int(train_mask.sum()),
    }
    return HoldoutSplit(train_mask=train_mask, holdout_mask=holdout_mask, metadata=metadata)
