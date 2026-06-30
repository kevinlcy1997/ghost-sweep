"""Leakage-safe zone ranking features for near-term alert risk."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta

import h3
import pandas as pd

from ghost_time import to_hk_feature_time
from ghost_zones import DEFAULT_H3_RESOLUTION, assign_zone, h3_zone_centroid


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


def add_engineered_ranking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add past-only ranking features to zone-time rows."""
    if df.empty:
        return df.copy()

    enhanced = df.copy()
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
        return enhanced

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
            zone_history = [event for event in history if event[zone_col] == zone_id]
            district_history = [event for event in history if event.get("district") == district]
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


