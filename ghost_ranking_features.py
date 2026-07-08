"""Leakage-safe zone ranking features for near-term alert risk."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from ghost_time import to_hk_feature_time
from ghost_zones import DEFAULT_H3_RESOLUTION, assign_zone, h3_zone_centroid


ROOT = Path(__file__).resolve().parent
DEFAULT_ROAD_CONTEXT_PATH = ROOT / "analysis" / "geo" / f"hk_h3_road_coverage_res{DEFAULT_H3_RESOLUTION}.csv"
MISSING_DISTANCE_M = 999_999.0

URBAN_CORE_DISTRICTS = {
    "Central",
    "Wan Chai",
    "Eastern",
    "Yau Tsim",
    "Mong Kok",
    "Sham Shui Po",
    "Kowloon City",
    "Kwun Tong",
}


def parse_event_dt(value: str) -> datetime:
    """Parse project event timestamps."""
    return to_hk_feature_time(value)


def enrich_events_with_zones(
    events: list[dict],
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> list[dict]:
    """Return events with H3 zone, district, region, and parsed timestamp fields."""
    enriched = []
    for event in events:
        lat = float(event.get("lat", 0.0))
        lng = float(event.get("lng", 0.0))
        if not (22.0 <= lat <= 22.7 and 113.7 <= lng <= 114.6):
            continue
        zone = assign_zone(lat, lng, resolution)
        row = dict(event)
        row.update(zone)
        row["dt"] = parse_event_dt(row.get("create_dt", "2000-01-01 00:00:00"))
        enriched.append(row)
    enriched.sort(key=lambda item: item["dt"])
    return enriched


def _count_between(events: list[dict], start: datetime, end: datetime, zone_col: str | None = None) -> Counter:
    counter: Counter = Counter()
    for event in events:
        if start <= event["dt"] < end:
            counter[event[zone_col] if zone_col else "_all"] += 1
    return counter


def _same_hour_rate(history: list[dict], key: str, value: str, hour: int) -> float:
    matching = [event for event in history if event.get(key) == value]
    if not matching:
        return 0.0
    return sum(1 for event in matching if event["dt"].hour == hour) / len(matching)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _distance_m(lat_a: float, lng_a: float, lat_b: float, lng_b: float) -> float:
    radius_m = 6_371_000.0
    phi_a = math.radians(float(lat_a))
    phi_b = math.radians(float(lat_b))
    delta_phi = math.radians(float(lat_b) - float(lat_a))
    delta_lam = math.radians(float(lng_b) - float(lng_a))
    hav = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lam / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(hav), math.sqrt(1 - hav))


def _ensure_zone_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    if "zone_lat" not in frame:
        frame["zone_lat"] = frame["zone_id"].map(lambda zone: h3_zone_centroid(str(zone))[0])
    if "zone_lng" not in frame:
        frame["zone_lng"] = frame["zone_id"].map(lambda zone: h3_zone_centroid(str(zone))[1])
    return frame


def _add_ring2_features_legacy(enhanced: pd.DataFrame) -> pd.DataFrame:
    ring2_24h_values: list[float] = []
    ring2_7d_values: list[float] = []
    ring2_active_values: list[int] = []
    ring2_ratio_values: list[float] = []
    if "zone_id" not in enhanced or "target_time" not in enhanced:
        enhanced["ring2_event_count_24h"] = 0.0
        enhanced["ring2_event_count_7d"] = 0.0
        enhanced["ring2_active_zones_24h"] = 0
        enhanced["ring2_to_ring1_24h_ratio"] = 0.0
        return enhanced

    row_lookup = {
        (row.target_time, row.zone_id): row
        for row in enhanced[
            [
                "target_time",
                "zone_id",
                "zone_event_count_24h",
                "zone_event_count_7d",
            ]
        ].itertuples(index=False)
    }
    ring2_cache: dict[str, set[str]] = {}
    for idx, row in enhanced[["target_time", "zone_id", "neighbor_event_count_24h"]].iterrows():
        zone_id = row["zone_id"]
        if zone_id not in ring2_cache:
            ring2_cache[zone_id] = set(h3.grid_disk(zone_id, 2)) - set(h3.grid_disk(zone_id, 1))
        ring2_rows = [
            row_lookup[(row["target_time"], ring2_zone)]
            for ring2_zone in ring2_cache[zone_id]
            if (row["target_time"], ring2_zone) in row_lookup
        ]
        ring2_24h = sum(ring2.zone_event_count_24h for ring2 in ring2_rows)
        enhanced.at[idx, "ring2_event_count_24h"] = ring2_24h
        enhanced.at[idx, "ring2_event_count_7d"] = sum(
            ring2.zone_event_count_7d for ring2 in ring2_rows
        )
        enhanced.at[idx, "ring2_active_zones_24h"] = sum(
            1 for ring2 in ring2_rows if ring2.zone_event_count_24h > 0
        )
        enhanced.at[idx, "ring2_to_ring1_24h_ratio"] = _safe_divide(
            ring2_24h,
            row["neighbor_event_count_24h"],
        )
    return enhanced


def _add_ring2_features(enhanced: pd.DataFrame) -> pd.DataFrame:
    if "zone_id" not in enhanced or "target_time" not in enhanced:
        enhanced["ring2_event_count_24h"] = 0.0
        enhanced["ring2_event_count_7d"] = 0.0
        enhanced["ring2_active_zones_24h"] = 0
        enhanced["ring2_to_ring1_24h_ratio"] = 0.0
        return enhanced

    count_24h_by_time: dict[object, dict[str, float]] = {}
    count_7d_by_time: dict[object, dict[str, float]] = {}
    for row in enhanced[
        ["target_time", "zone_id", "zone_event_count_24h", "zone_event_count_7d"]
    ].itertuples(index=False):
        count_24h_by_time.setdefault(row.target_time, {})[row.zone_id] = float(
            row.zone_event_count_24h
        )
        count_7d_by_time.setdefault(row.target_time, {})[row.zone_id] = float(
            row.zone_event_count_7d
        )

    ring2_cache: dict[str, set[str]] = {}
    ring2_24h_values: list[float] = []
    ring2_7d_values: list[float] = []
    ring2_active_values: list[int] = []
    ring2_ratio_values: list[float] = []

    for row in enhanced[["target_time", "zone_id", "neighbor_event_count_24h"]].itertuples(
        index=False
    ):
        zone_id = row.zone_id
        if zone_id not in ring2_cache:
            ring2_cache[zone_id] = set(h3.grid_disk(zone_id, 2)) - set(h3.grid_disk(zone_id, 1))
        counts_24h = count_24h_by_time.get(row.target_time, {})
        counts_7d = count_7d_by_time.get(row.target_time, {})
        ring2_24h = sum(counts_24h.get(ring2_zone, 0.0) for ring2_zone in ring2_cache[zone_id])
        ring2_24h_values.append(ring2_24h)
        ring2_7d_values.append(
            sum(counts_7d.get(ring2_zone, 0.0) for ring2_zone in ring2_cache[zone_id])
        )
        ring2_active_values.append(
            sum(1 for ring2_zone in ring2_cache[zone_id] if counts_24h.get(ring2_zone, 0.0) > 0)
        )
        ring2_ratio_values.append(_safe_divide(ring2_24h, row.neighbor_event_count_24h))

    enhanced["ring2_event_count_24h"] = ring2_24h_values
    enhanced["ring2_event_count_7d"] = ring2_7d_values
    enhanced["ring2_active_zones_24h"] = ring2_active_values
    enhanced["ring2_to_ring1_24h_ratio"] = ring2_ratio_values
    return enhanced


def _add_district_relative_features(enhanced: pd.DataFrame) -> pd.DataFrame:
    enhanced["zone_24h_share_of_district"] = 0.0
    enhanced["zone_7d_rank_in_district"] = 0.0
    enhanced["zone_same_hour_percentile_in_district"] = 0.0
    if not {"target_time", "district"}.issubset(enhanced.columns):
        return enhanced

    group_cols = ["target_time", "district"]
    district_total = enhanced.groupby(group_cols)["zone_event_count_24h"].transform("sum")
    fallback_total = enhanced["district_event_count_24h"] if "district_event_count_24h" in enhanced else 0
    denominator = district_total.where(district_total > 0, fallback_total)
    enhanced["zone_24h_share_of_district"] = [
        _safe_divide(zone_count, total)
        for zone_count, total in zip(enhanced["zone_event_count_24h"], denominator)
    ]
    enhanced["zone_7d_rank_in_district"] = enhanced.groupby(group_cols)[
        "zone_event_count_7d"
    ].rank(method="min", ascending=False)
    if "zone_same_hour_rate" in enhanced:
        enhanced["zone_same_hour_percentile_in_district"] = enhanced.groupby(group_cols)[
            "zone_same_hour_rate"
        ].rank(method="max", pct=True)
    return enhanced


def _nearest_distance_to_rows(row: pd.Series, candidates: pd.DataFrame) -> float:
    if candidates.empty:
        return MISSING_DISTANCE_M
    lat1 = math.radians(float(row["zone_lat"]))
    lng1 = math.radians(float(row["zone_lng"]))
    lat2 = np.radians(candidates["zone_lat"].astype(float).to_numpy())
    lng2 = np.radians(candidates["zone_lng"].astype(float).to_numpy())
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    hav = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    distances = 2 * 6_371_000.0 * np.arctan2(np.sqrt(hav), np.sqrt(1 - hav))
    return round(float(np.min(distances)), 3) if len(distances) else MISSING_DISTANCE_M


def _distance_to_weighted_centroid(row: pd.Series, candidates: pd.DataFrame) -> float:
    if candidates.empty:
        return MISSING_DISTANCE_M
    weights = candidates["zone_event_count_24h"].astype(float).clip(lower=0.0)
    if float(weights.sum()) <= 0:
        weights = pd.Series([1.0] * len(candidates), index=candidates.index)
    lat = float((candidates["zone_lat"].astype(float) * weights).sum() / weights.sum())
    lng = float((candidates["zone_lng"].astype(float) * weights).sum() / weights.sum())
    return round(_distance_m(row["zone_lat"], row["zone_lng"], lat, lng), 3)


def _add_hotspot_distance_features(enhanced: pd.DataFrame) -> pd.DataFrame:
    enhanced["distance_to_nearest_event_3h_m"] = MISSING_DISTANCE_M
    enhanced["distance_to_nearest_event_24h_m"] = MISSING_DISTANCE_M
    enhanced["distance_to_district_recent_centroid_24h_m"] = MISSING_DISTANCE_M
    if not {"target_time", "zone_lat", "zone_lng"}.issubset(enhanced.columns):
        return enhanced

    for _, group in enhanced.groupby("target_time", sort=False):
        active_3h = group[group["zone_event_count_3h"].astype(float) > 0]
        active_24h = group[group["zone_event_count_24h"].astype(float) > 0]
        active_24h_by_district = (
            {
                district: district_rows
                for district, district_rows in active_24h.groupby("district", sort=False)
            }
            if "district" in group.columns
            else {}
        )
        for idx, row in group.iterrows():
            enhanced.at[idx, "distance_to_nearest_event_3h_m"] = _nearest_distance_to_rows(
                row,
                active_3h,
            )
            enhanced.at[idx, "distance_to_nearest_event_24h_m"] = _nearest_distance_to_rows(
                row,
                active_24h,
            )
            district_active = active_24h_by_district.get(row.get("district"), active_24h)
            enhanced.at[idx, "distance_to_district_recent_centroid_24h_m"] = (
                _distance_to_weighted_centroid(row, district_active)
            )
    return enhanced


def _add_hotspot_distance_features(enhanced: pd.DataFrame) -> pd.DataFrame:
    enhanced["distance_to_nearest_event_3h_m"] = MISSING_DISTANCE_M
    enhanced["distance_to_nearest_event_24h_m"] = MISSING_DISTANCE_M
    enhanced["distance_to_district_recent_centroid_24h_m"] = MISSING_DISTANCE_M
    if not {"target_time", "zone_lat", "zone_lng"}.issubset(enhanced.columns):
        return enhanced

    def nearest_distances(
        source_lats: np.ndarray,
        source_lngs: np.ndarray,
        candidate_lats: np.ndarray,
        candidate_lngs: np.ndarray,
        chunk_size: int = 512,
    ) -> np.ndarray:
        if len(candidate_lats) == 0:
            return np.full(len(source_lats), MISSING_DISTANCE_M, dtype=float)
        result = np.empty(len(source_lats), dtype=float)
        cand_lats_rad = np.radians(candidate_lats.astype(float))
        cand_lngs_rad = np.radians(candidate_lngs.astype(float))
        for start in range(0, len(source_lats), chunk_size):
            end = min(start + chunk_size, len(source_lats))
            lat_rad = np.radians(source_lats[start:end].astype(float))[:, None]
            lng_rad = np.radians(source_lngs[start:end].astype(float))[:, None]
            dlat = cand_lats_rad[None, :] - lat_rad
            dlng = cand_lngs_rad[None, :] - lng_rad
            hav = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * np.cos(cand_lats_rad)[
                None, :
            ] * np.sin(dlng / 2) ** 2
            distances = 2 * 6_371_000.0 * np.arctan2(np.sqrt(hav), np.sqrt(1 - hav))
            result[start:end] = np.min(distances, axis=1)
        return np.round(result, 3)

    for _, group in enhanced.groupby("target_time", sort=False):
        indexes = group.index
        source_lats = group["zone_lat"].astype(float).to_numpy()
        source_lngs = group["zone_lng"].astype(float).to_numpy()
        active_3h = group[group["zone_event_count_3h"].astype(float) > 0]
        active_24h = group[group["zone_event_count_24h"].astype(float) > 0]

        enhanced.loc[indexes, "distance_to_nearest_event_3h_m"] = nearest_distances(
            source_lats,
            source_lngs,
            active_3h["zone_lat"].astype(float).to_numpy(),
            active_3h["zone_lng"].astype(float).to_numpy(),
        )
        enhanced.loc[indexes, "distance_to_nearest_event_24h_m"] = nearest_distances(
            source_lats,
            source_lngs,
            active_24h["zone_lat"].astype(float).to_numpy(),
            active_24h["zone_lng"].astype(float).to_numpy(),
        )

        if active_24h.empty:
            continue

        weights = active_24h["zone_event_count_24h"].astype(float).clip(lower=0.0)
        if float(weights.sum()) <= 0:
            weights = pd.Series(np.ones(len(active_24h)), index=active_24h.index)
        fallback_lat = float((active_24h["zone_lat"].astype(float) * weights).sum() / weights.sum())
        fallback_lng = float((active_24h["zone_lng"].astype(float) * weights).sum() / weights.sum())
        centroid_by_district: dict[str, tuple[float, float]] = {}
        if "district" in group.columns:
            for district, district_rows in active_24h.groupby("district", sort=False):
                district_weights = district_rows["zone_event_count_24h"].astype(float).clip(lower=0.0)
                if float(district_weights.sum()) <= 0:
                    district_weights = pd.Series(np.ones(len(district_rows)), index=district_rows.index)
                centroid_by_district[district] = (
                    float(
                        (district_rows["zone_lat"].astype(float) * district_weights).sum()
                        / district_weights.sum()
                    ),
                    float(
                        (district_rows["zone_lng"].astype(float) * district_weights).sum()
                        / district_weights.sum()
                    ),
                )

        centroid_lats = []
        centroid_lngs = []
        for district in group.get("district", pd.Series(index=indexes, dtype=object)):
            lat, lng = centroid_by_district.get(district, (fallback_lat, fallback_lng))
            centroid_lats.append(lat)
            centroid_lngs.append(lng)
        enhanced.loc[indexes, "distance_to_district_recent_centroid_24h_m"] = np.round(
            [
                _distance_m(lat, lng, centroid_lat, centroid_lng)
                for lat, lng, centroid_lat, centroid_lng in zip(
                    source_lats,
                    source_lngs,
                    centroid_lats,
                    centroid_lngs,
                    strict=True,
                )
            ],
            3,
        )
    return enhanced


def _add_road_context_features(
    enhanced: pd.DataFrame,
    road_context_path: Path | None,
) -> pd.DataFrame:
    road_columns = {
        "nearest_road_m": MISSING_DISTANCE_M,
        "road_segment_count": 0.0,
        "road_source_mismatch": 0.0,
        "has_drivable_road": 0.0,
    }
    for column, default in road_columns.items():
        enhanced[column] = default
    if road_context_path is None or not road_context_path.exists() or "zone_id" not in enhanced:
        return enhanced

    road_context = pd.read_csv(road_context_path)
    if "h3_zone" not in road_context:
        return enhanced
    keep_cols = ["h3_zone", *road_columns.keys()]
    road_context = road_context[[column for column in keep_cols if column in road_context]].copy()
    road_context = road_context.rename(columns={"h3_zone": "zone_id"})
    merged = enhanced.merge(road_context, on="zone_id", how="left", suffixes=("", "_road"))
    for column, default in road_columns.items():
        road_col = f"{column}_road"
        if road_col in merged:
            merged[column] = merged[road_col].fillna(merged[column]).fillna(default)
            merged = merged.drop(columns=[road_col])
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(default)
    return merged


def add_engineered_ranking_features(
    df: pd.DataFrame,
    road_context_path: Path | str | None = None,
) -> pd.DataFrame:
    """Add past-only ranking features to zone-time rows."""
    if df.empty:
        return df.copy()

    enhanced = df.copy()
    road_path = Path(road_context_path) if road_context_path is not None else DEFAULT_ROAD_CONTEXT_PATH
    for column in [
        "zone_event_count_3h",
        "zone_event_count_24h",
        "zone_event_count_7d",
        "district_event_count_3h",
        "district_event_count_24h",
    ]:
        if column not in enhanced:
            enhanced[column] = 0.0

    enhanced["zone_3h_to_24h_ratio"] = enhanced.apply(
        lambda row: _safe_divide(row["zone_event_count_3h"], row["zone_event_count_24h"]),
        axis=1,
    )
    enhanced["zone_24h_to_7d_ratio"] = enhanced.apply(
        lambda row: _safe_divide(row["zone_event_count_24h"], row["zone_event_count_7d"]),
        axis=1,
    )
    enhanced["district_3h_to_24h_ratio"] = enhanced.apply(
        lambda row: _safe_divide(row["district_event_count_3h"], row["district_event_count_24h"]),
        axis=1,
    )

    hour_angle = 2 * math.pi * enhanced["hour"].astype(float) / 24.0
    dow_angle = 2 * math.pi * enhanced["day_of_week"].astype(float) / 7.0
    enhanced["hour_sin"] = hour_angle.map(math.sin)
    enhanced["hour_cos"] = hour_angle.map(math.cos)
    enhanced["dow_sin"] = dow_angle.map(math.sin)
    enhanced["dow_cos"] = dow_angle.map(math.cos)

    enhanced["neighbor_event_count_3h"] = 0.0
    enhanced["neighbor_event_count_24h"] = 0.0
    enhanced["neighbor_event_count_7d"] = 0.0
    enhanced["neighbor_active_zones_24h"] = 0

    if "zone_id" not in enhanced or "target_time" not in enhanced:
        enhanced = _add_district_relative_features(enhanced)
        enhanced = _add_hotspot_distance_features(enhanced)
        return _add_road_context_features(enhanced, road_path)

    row_lookup = {
        (row.target_time, row.zone_id): row
        for row in enhanced[
            [
                "target_time",
                "zone_id",
                "zone_event_count_3h",
                "zone_event_count_24h",
                "zone_event_count_7d",
            ]
        ].itertuples(index=False)
    }
    neighbor_cache: dict[str, set[str]] = {}
    for idx, row in enhanced[["target_time", "zone_id"]].iterrows():
        zone_id = row["zone_id"]
        if zone_id not in neighbor_cache:
            neighbor_cache[zone_id] = set(h3.grid_disk(zone_id, 1)) - {zone_id}
        neighbors = [
            row_lookup[(row["target_time"], neighbor)]
            for neighbor in neighbor_cache[zone_id]
            if (row["target_time"], neighbor) in row_lookup
        ]
        enhanced.at[idx, "neighbor_event_count_3h"] = sum(
            neighbor.zone_event_count_3h for neighbor in neighbors
        )
        enhanced.at[idx, "neighbor_event_count_24h"] = sum(
            neighbor.zone_event_count_24h for neighbor in neighbors
        )
        enhanced.at[idx, "neighbor_event_count_7d"] = sum(
            neighbor.zone_event_count_7d for neighbor in neighbors
        )
        enhanced.at[idx, "neighbor_active_zones_24h"] = sum(
            1 for neighbor in neighbors if neighbor.zone_event_count_24h > 0
        )

    enhanced = _ensure_zone_coordinates(enhanced)
    enhanced = _add_ring2_features(enhanced)
    enhanced = _add_district_relative_features(enhanced)
    enhanced = _add_hotspot_distance_features(enhanced)
    enhanced = _add_road_context_features(enhanced, road_path)
    return enhanced


def sample_spatial_training_rows(
    df: pd.DataFrame,
    target_col: str,
    negative_ratio: int = 5,
    inactive_negative_fraction: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample Stage 2 rows around active windows while retaining positives."""
    if df.empty:
        return df.copy()
    if target_col not in df:
        raise ValueError(f"Missing required target column: {target_col}")

    frame = df.copy()
    active_by_time = frame.groupby("target_time")[target_col].transform("sum") > 0
    positives = frame[frame[target_col].astype(int) == 1]
    if positives.empty:
        return frame.iloc[0:0].copy()

    negative_ratio = max(0, int(negative_ratio))
    active_negative_limit = int(len(positives) * negative_ratio)
    active_negatives = frame[(frame[target_col].astype(int) == 0) & active_by_time].copy()
    sort_cols = [
        column
        for column in [
            "zone_event_count_24h",
            "district_event_count_24h",
            "zone_event_count_7d",
        ]
        if column in active_negatives
    ]
    if sort_cols:
        active_negatives = active_negatives.sort_values(sort_cols, ascending=False)
    if active_negative_limit > 0:
        active_negatives = active_negatives.head(active_negative_limit)
    else:
        active_negatives = active_negatives.iloc[0:0]

    inactive_limit = int(math.ceil(active_negative_limit * max(0.0, inactive_negative_fraction)))
    inactive_negatives = frame[(frame[target_col].astype(int) == 0) & ~active_by_time].copy()
    if inactive_limit > 0 and not inactive_negatives.empty:
        inactive_negatives = inactive_negatives.sample(
            n=min(inactive_limit, len(inactive_negatives)),
            random_state=random_state,
        )
    else:
        inactive_negatives = inactive_negatives.iloc[0:0]

    sampled = (
        pd.concat([positives, active_negatives, inactive_negatives], axis=0)
        .loc[lambda rows: ~rows.index.duplicated(keep="first")]
        .sort_values(["target_time", target_col, "zone_id"], ascending=[True, False, True])
    )
    return sampled.reset_index(drop=True)


def build_zone_ranking_training_data(
    events: list[dict],
    zone_col: str = "h3_zone",
    forecast_hours: int = 2,
    lookback_days: int = 14,
    resolution: int = DEFAULT_H3_RESOLUTION,
    horizon_minutes: int | None = None,
    target_col: str = "alert_next_2h",
) -> pd.DataFrame:
    """Build rows for ranking zones by alert risk in the next forecast window.

    Features use only events before ``target_time``. Labels use events in
    ``[target_time, target_time + forecast_hours)``.
    """
    enriched = enrich_events_with_zones(events, resolution)
    if not enriched:
        return pd.DataFrame()

    zones = sorted({event[zone_col] for event in enriched})
    start = min(event["dt"] for event in enriched).replace(minute=0, second=0, microsecond=0)
    end = max(event["dt"] for event in enriched).replace(minute=0, second=0, microsecond=0)
    target_times = pd.date_range(start + timedelta(hours=lookback_days * 24), end, freq="1h")

    rows = []
    for target_time in target_times:
        target_dt = target_time.to_pydatetime()
        history_start = target_dt - timedelta(days=lookback_days)
        history = [event for event in enriched if history_start <= event["dt"] < target_dt]
        if not history:
            continue

        if horizon_minutes is None:
            horizon_minutes = forecast_hours * 60
        future_start = target_dt
        future_end = target_dt + timedelta(minutes=horizon_minutes)
        future_by_zone = _count_between(enriched, future_start, future_end, zone_col)
        counts_1h = _count_between(history, target_dt - timedelta(hours=1), target_dt, zone_col)
        counts_3h = _count_between(history, target_dt - timedelta(hours=3), target_dt, zone_col)
        counts_24h = _count_between(history, target_dt - timedelta(hours=24), target_dt, zone_col)
        counts_7d = _count_between(history, target_dt - timedelta(days=7), target_dt, zone_col)
        district_3h = _count_between(history, target_dt - timedelta(hours=3), target_dt, "district")
        district_24h = _count_between(history, target_dt - timedelta(hours=24), target_dt, "district")

        latest_by_zone: dict[str, datetime] = {}
        exemplar_by_zone: dict[str, dict] = {}
        for event in history:
            latest_by_zone[event[zone_col]] = event["dt"]
            exemplar_by_zone[event[zone_col]] = event

        active_district_zones = {}
        for event in history:
            if event["dt"] >= target_dt - timedelta(hours=24):
                active_district_zones.setdefault(event["district"], set()).add(event[zone_col])

        history_by_zone: dict[str, list[dict]] = {}
        history_by_district: dict[str, list[dict]] = {}
        for event in history:
            history_by_zone.setdefault(event[zone_col], []).append(event)
            history_by_district.setdefault(event.get("district", "Unknown"), []).append(event)

        for zone_id in zones:
            exemplar = exemplar_by_zone.get(zone_id)
            if exemplar is None:
                zone_lat, zone_lng = h3_zone_centroid(zone_id)
                district = "Unknown"
                region = "Unknown"
            else:
                zone_lat = float(exemplar.get("zone_lat", 0.0))
                zone_lng = float(exemplar.get("zone_lng", 0.0))
                district = str(exemplar.get("district", "Unknown"))
                region = str(exemplar.get("region", "Unknown"))

            last_seen = latest_by_zone.get(zone_id)
            hours_since_last = (
                (target_dt - last_seen).total_seconds() / 3600.0 if last_seen else 9999.0
            )
            zone_history = history_by_zone.get(zone_id, [])
            district_history = history_by_district.get(district, [])
            row = {
                "target_time": target_dt,
                "zone_id": zone_id,
                "h3_zone": zone_id,
                "h3_resolution": resolution,
                "district": district,
                "region": region,
                "zone_lat": zone_lat,
                "zone_lng": zone_lng,
                "hour": target_dt.hour,
                "day_of_week": target_dt.weekday(),
                "is_weekend": int(target_dt.weekday() >= 5),
                "hour_bucket": _hour_bucket(target_dt.hour),
                "zone_event_count_1h": counts_1h[zone_id],
                "zone_event_count_3h": counts_3h[zone_id],
                "zone_event_count_24h": counts_24h[zone_id],
                "zone_event_count_7d": counts_7d[zone_id],
                "zone_hours_since_last_event": hours_since_last,
                "district_event_count_3h": district_3h[district],
                "district_event_count_24h": district_24h[district],
                "district_active_zones_24h": len(active_district_zones.get(district, set())),
                "zone_same_hour_rate": _same_hour_rate(zone_history, zone_col, zone_id, target_dt.hour),
                "district_same_hour_rate": _same_hour_rate(district_history, "district", district, target_dt.hour),
                "is_urban_core": int(district in URBAN_CORE_DISTRICTS),
                "event_count_next_2h": future_by_zone[zone_id],
                target_col: int(future_by_zone[zone_id] > 0),
            }
            rows.append(row)

    return add_engineered_ranking_features(pd.DataFrame(rows))


def _hour_bucket(hour: int) -> str:
    if 7 <= hour < 10:
        return "commute_am"
    if 10 <= hour < 15:
        return "midday"
    if 15 <= hour < 20:
        return "commute_pm"
    if 20 <= hour or hour < 2:
        return "late"
    return "overnight"
