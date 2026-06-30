"""City-level activity features for the two-stage zone risk model."""

from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd

from ghost_ranking_features import enrich_events_with_zones
from ghost_zones import DEFAULT_H3_RESOLUTION


def horizon_slug(horizon_minutes: int) -> str:
    if horizon_minutes % 60 == 0:
        return f"{horizon_minutes // 60}h"
    return f"{horizon_minutes}m"


def activity_target_for_horizon(horizon_minutes: int) -> str:
    return f"activity_next_{horizon_slug(horizon_minutes)}"


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


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _count_between(events: list[dict], start, end) -> int:
    return sum(1 for event in events if start <= event["dt"] < end)


def build_activity_training_data(
    events: list[dict],
    horizon_minutes: int,
    lookback_hours: int = 168,
    resolution: int = DEFAULT_H3_RESOLUTION,
) -> pd.DataFrame:
    """Build hourly city-level rows for predicting any HK activity soon.

    Features use only events before ``target_time``. The target uses events in
    ``[target_time, target_time + horizon_minutes)``.
    """
    enriched = enrich_events_with_zones(events, resolution)
    if not enriched:
        return pd.DataFrame()

    start = min(event["dt"] for event in enriched).replace(minute=0, second=0, microsecond=0)
    end = max(event["dt"] for event in enriched).replace(minute=0, second=0, microsecond=0)
    target_times = pd.date_range(start + timedelta(hours=lookback_hours), end, freq="1h")
    target_col = activity_target_for_horizon(horizon_minutes)

    rows = []
    for target_time in target_times:
        target_dt = target_time.to_pydatetime()
        history = [
            event
            for event in enriched
            if target_dt - timedelta(hours=lookback_hours) <= event["dt"] < target_dt
        ]
        future_end = target_dt + timedelta(minutes=horizon_minutes)
        future_count = _count_between(enriched, target_dt, future_end)
        history_24h = [
            event for event in history if event["dt"] >= target_dt - timedelta(hours=24)
        ]
        city_3h = _count_between(history, target_dt - timedelta(hours=3), target_dt)
        city_24h = len(history_24h)
        city_7d = _count_between(history, target_dt - timedelta(days=7), target_dt)
        hour_angle = 2 * math.pi * target_dt.hour / 24.0
        dow_angle = 2 * math.pi * target_dt.weekday() / 7.0

        rows.append(
            {
                "target_time": target_dt,
                "hour": target_dt.hour,
                "day_of_week": target_dt.weekday(),
                "is_weekend": int(target_dt.weekday() >= 5),
                "hour_bucket": _hour_bucket(target_dt.hour),
                "city_event_count_1h": _count_between(
                    history,
                    target_dt - timedelta(hours=1),
                    target_dt,
                ),
                "city_event_count_3h": city_3h,
                "city_event_count_24h": city_24h,
                "city_event_count_7d": city_7d,
                "active_districts_24h": len(
                    {event.get("district", "Unknown") for event in history_24h}
                ),
                "active_regions_24h": len({event.get("region", "Unknown") for event in history_24h}),
                "city_3h_to_24h_ratio": _safe_divide(city_3h, city_24h),
                "city_24h_to_7d_ratio": _safe_divide(city_24h, city_7d),
                "hour_sin": math.sin(hour_angle),
                "hour_cos": math.cos(hour_angle),
                "dow_sin": math.sin(dow_angle),
                "dow_cos": math.cos(dow_angle),
                "event_count_next_horizon": future_count,
                target_col: int(future_count > 0),
            }
        )

    return pd.DataFrame(rows)
