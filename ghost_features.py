# ghost_features.py
"""Feature engineering for Ghost Sweep predictive model."""

from datetime import datetime, timedelta
import pandas as pd

from ghost_time import to_hk_feature_time


def build_features(events: list[dict], target_dt: datetime,
                   window_hours: int = 1) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()

    target_dt = to_hk_feature_time(target_dt)
    df_events = pd.DataFrame(events)
    df_events["create_dt_parsed"] = pd.to_datetime(
        [to_hk_feature_time(value) for value in df_events["create_dt"]]
    )
    active_cells = df_events[["grid_cell", "district", "region"]].drop_duplicates("grid_cell")
    past_events = df_events[df_events["create_dt_parsed"] < target_dt].copy()

    hour = target_dt.hour
    day_of_week = target_dt.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    month = target_dt.month
    dt_24h_ago = target_dt - timedelta(hours=24)
    dt_7d_ago = target_dt - timedelta(days=7)

    rows = []
    for _, cell_row in active_cells.iterrows():
        cell_id = cell_row["grid_cell"]
        district = cell_row["district"]
        region = cell_row["region"]

        cell_events = past_events[past_events["grid_cell"] == cell_id]
        cell_24h = cell_events[cell_events["create_dt_parsed"] >= dt_24h_ago]
        cell_7d = cell_events[cell_events["create_dt_parsed"] >= dt_7d_ago]

        cell_historical_freq = len(cell_events)
        cell_7d_count = len(cell_7d)
        cell_24h_count = len(cell_24h)

        if len(cell_events) > 0:
            last_event_dt = cell_events["create_dt_parsed"].max()
            cell_last_seen_hours_ago = (target_dt - last_event_dt).total_seconds() / 3600
        else:
            cell_last_seen_hours_ago = 9999

        total_up = cell_events["total_upvotes"].sum() if len(cell_events) > 0 else 0
        total_down = cell_events["total_downvotes"].sum() if len(cell_events) > 0 else 0
        upvote_ratio = total_up / (total_up + total_down) if (total_up + total_down) > 0 else 0.5
        avg_report_count = cell_events["report_count"].mean() if len(cell_events) > 0 else 0

        streak = 0
        if len(cell_events) > 0:
            event_days = cell_events["create_dt_parsed"].dt.date.unique()
            event_days_sorted = sorted(event_days)
            if len(event_days_sorted) >= 2:
                for i in range(len(event_days_sorted) - 1, 0, -1):
                    if (event_days_sorted[i] - event_days_sorted[i-1]).days == 1:
                        streak = 1
                        break

        district_events = past_events[past_events["district"] == district]
        district_24h = district_events[district_events["create_dt_parsed"] >= dt_24h_ago]
        district_24h_count = len(district_24h)

        if len(district_events) > 0:
            first_event = district_events["create_dt_parsed"].min()
            days_span = max((target_dt - first_event).days, 1)
            district_historical_rate = len(district_events) / days_span
        else:
            district_historical_rate = 0

        district_active_cells = district_24h["grid_cell"].nunique() if len(district_24h) > 0 else 0

        district_at_hour = district_events[district_events["create_dt_parsed"].dt.hour == hour]
        if len(district_events) > 0:
            days_span = max((target_dt - district_events["create_dt_parsed"].min()).days, 1)
            district_hour_rate = len(district_at_hour) / days_span
        else:
            district_hour_rate = 0

        all_at_hour = past_events[past_events["create_dt_parsed"].dt.hour == hour]
        if len(past_events) > 0:
            global_days = max((target_dt - past_events["create_dt_parsed"].min()).days, 1)
            hour_historical_rate = len(all_at_hour) / global_days
        else:
            hour_historical_rate = 0

        all_at_dow_hour = past_events[
            (past_events["create_dt_parsed"].dt.hour == hour) &
            (past_events["create_dt_parsed"].dt.dayofweek == day_of_week)
        ]
        dow_weeks = max((target_dt - past_events["create_dt_parsed"].min()).days // 7, 1) if len(past_events) > 0 else 1
        dow_hour_rate = len(all_at_dow_hour) / dow_weeks

        cell_parts = cell_id.split("_")
        neighbor_24h_count = 0
        if len(cell_parts) == 2:
            c_lat, c_lng = float(cell_parts[0]), float(cell_parts[1])
            neighbor_cells = [
                f"{c_lat + dlat:.3f}_{c_lng + dlng:.3f}"
                for dlat in (-0.005, 0, 0.005)
                for dlng in (-0.005, 0, 0.005)
                if not (dlat == 0 and dlng == 0)
            ]
            neighbor_events = past_events[
                (past_events["grid_cell"].isin(neighbor_cells)) &
                (past_events["create_dt_parsed"] >= dt_24h_ago)
            ]
            neighbor_24h_count = len(neighbor_events)

        rows.append({
            "grid_cell": cell_id, "district": district, "region": region,
            "hour": hour, "day_of_week": day_of_week, "is_weekend": is_weekend, "month": month,
            "cell_historical_freq": cell_historical_freq,
            "cell_7d_count": cell_7d_count, "cell_24h_count": cell_24h_count,
            "cell_last_seen_hours_ago": cell_last_seen_hours_ago,
            "neighbor_24h_count": neighbor_24h_count, "streak_active": streak,
            "upvote_ratio": upvote_ratio, "avg_report_count": avg_report_count,
            "district_24h_count": district_24h_count,
            "district_historical_rate": district_historical_rate,
            "district_active_cells": district_active_cells,
            "district_hour_rate": district_hour_rate,
            "hour_historical_rate": hour_historical_rate,
            "dow_hour_rate": dow_hour_rate,
        })

    return pd.DataFrame(rows)


def build_training_data(events: list[dict], window_hours: int = 1) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()

    df_events = pd.DataFrame(events)
    df_events["create_dt_parsed"] = pd.to_datetime(
        [to_hk_feature_time(value) for value in df_events["create_dt"]]
    )
    all_times = sorted(df_events["create_dt_parsed"].dt.floor("h").unique())

    all_rows = []
    for t in all_times:
        target_dt = t.to_pydatetime()
        window_end = target_dt + timedelta(hours=window_hours)
        features_df = build_features(events, target_dt, window_hours)
        if features_df.empty:
            continue
        future_events = df_events[
            (df_events["create_dt_parsed"] >= target_dt) &
            (df_events["create_dt_parsed"] < window_end)
        ]
        active_cells_in_window = set(future_events["grid_cell"].unique())
        features_df["has_warden"] = features_df["grid_cell"].isin(active_cells_in_window).astype(int)
        all_rows.append(features_df)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)
