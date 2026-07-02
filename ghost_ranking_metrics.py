"""Ranking metrics for zone risk forecasts."""

from __future__ import annotations

import numpy as np


def _ordered_labels(y_true, y_score) -> np.ndarray:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    order = np.argsort(-scores, kind="mergesort")
    return labels[order]


def _validated_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    return labels, scores


def precision_at_k(y_true, y_score, k: int) -> float:
    """Return precision among the top-k scored rows."""
    if k <= 0:
        raise ValueError("k must be positive")
    ordered = _ordered_labels(y_true, y_score)
    if ordered.size == 0:
        return 0.0
    top = ordered[: min(k, ordered.size)]
    return float(top.mean()) if top.size else 0.0


def recall_at_k(y_true, y_score, k: int) -> float:
    """Return recall captured by the top-k scored rows."""
    if k <= 0:
        raise ValueError("k must be positive")
    ordered = _ordered_labels(y_true, y_score)
    positives = int(ordered.sum())
    if positives == 0:
        return 0.0
    return float(ordered[: min(k, ordered.size)].sum() / positives)


def top_decile_lift(y_true, y_score) -> float:
    """Return top-decile positive rate divided by overall positive rate."""
    ordered = _ordered_labels(y_true, y_score)
    if ordered.size == 0:
        return 0.0
    base_rate = float(ordered.mean())
    if base_rate == 0.0:
        return 0.0
    top_n = max(1, int(np.ceil(ordered.size * 0.1)))
    return float(ordered[:top_n].mean() / base_rate)


def brier_score(y_true, y_score) -> float:
    """Return mean squared error between labels and probability-like scores."""
    labels, scores = _validated_arrays(y_true, y_score)
    if labels.size == 0:
        return 0.0
    clipped = np.clip(scores, 0.0, 1.0)
    return float(np.mean((clipped - labels) ** 2))


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """Return equal-width expected calibration error for probability scores."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    labels, scores = _validated_arrays(y_true, y_score)
    if labels.size == 0:
        return 0.0

    clipped = np.clip(scores, 0.0, 1.0)
    bin_ids = np.minimum((clipped * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        confidence = float(clipped[mask].mean())
        accuracy = float(labels[mask].mean())
        error += float(mask.mean()) * abs(confidence - accuracy)
    return float(error)


def group_hit_rate_at_k(y_true, y_score, groups, k: int) -> float:
    """Return share of positive groups represented in the top-k scored rows."""
    if k <= 0:
        raise ValueError("k must be positive")
    labels, scores = _validated_arrays(y_true, y_score)
    group_values = np.asarray(groups, dtype=object)
    if group_values.shape[0] != labels.shape[0]:
        raise ValueError("groups must have the same length as y_true")
    if labels.size == 0 or labels.sum() == 0:
        return 0.0

    positive_groups = {
        str(group)
        for label, group in zip(labels, group_values)
        if int(label) == 1 and str(group)
    }
    if not positive_groups:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    top_groups = {str(group_values[index]) for index in order[: min(k, labels.size)]}
    return float(len(positive_groups & top_groups) / len(positive_groups))


def group_precision_at_k(
    predictions,
    k: int,
    score_col: str = "score",
    label_col: str = "actual",
    group_col: str = "target_time",
) -> float:
    """Return mean per-group precision@k across all groups."""
    if k <= 0:
        raise ValueError("k must be positive")
    if predictions.empty:
        return 0.0

    values: list[float] = []
    for _, group in predictions.groupby(group_col, sort=False):
        top = group.sort_values(score_col, ascending=False).head(k)
        if top.empty:
            values.append(0.0)
            continue
        values.append(float(top[label_col].astype(int).mean()))
    return float(sum(values) / len(values)) if values else 0.0


def group_recall_at_k(
    predictions,
    k: int,
    score_col: str = "score",
    label_col: str = "actual",
    group_col: str = "target_time",
) -> float:
    """Return mean per-positive-group recall@k; groups without positives are excluded."""
    if k <= 0:
        raise ValueError("k must be positive")
    if predictions.empty:
        return 0.0

    values: list[float] = []
    for _, group in predictions.groupby(group_col, sort=False):
        positives = int(group[label_col].astype(int).sum())
        if positives == 0:
            continue
        top_hits = int(
            group.sort_values(score_col, ascending=False).head(k)[label_col].astype(int).sum()
        )
        values.append(float(top_hits / positives))
    return float(sum(values) / len(values)) if values else 0.0


def near_miss_hit_rate_at_k(
    predictions,
    k: int,
    neighbor_lookup: dict[str, set[str]],
    score_col: str = "score",
    label_col: str = "actual",
    group_col: str = "target_time",
    zone_col: str = "zone_id",
) -> float:
    """Return share of positive groups with an exact or neighbor top-k hit."""
    if k <= 0:
        raise ValueError("k must be positive")
    if predictions.empty:
        return 0.0

    hits: list[int] = []
    for _, group in predictions.groupby(group_col, sort=False):
        positives = set(group.loc[group[label_col].astype(int) == 1, zone_col].astype(str))
        if not positives:
            hits.append(0)
            continue
        top = group.sort_values(score_col, ascending=False).head(k)
        group_hit = any(
            str(row[zone_col]) in positives
            or bool(neighbor_lookup.get(str(row[zone_col]), set()) & positives)
            for _, row in top.iterrows()
        )
        hits.append(int(group_hit))
    return float(sum(hits) / len(hits)) if hits else 0.0


def risk_band(probability: float) -> str:
    """Map a probability-like score to a stable dashboard risk band."""
    value = float(np.clip(probability, 0.0, 1.0))
    if value >= 0.10:
        return "critical"
    if value >= 0.05:
        return "high"
    if value >= 0.02:
        return "elevated"
    return "low"
