# ghost_clean.py
"""Data cleaning: consolidate raw sightings into unique warden events."""

import math
from datetime import datetime

SPATIAL_THRESHOLD_M = 20
TEMPORAL_THRESHOLD_MIN = 15


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_dt(dt_str: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return datetime.strptime(dt_str[:19], "%Y-%m-%dT%H:%M:%S")


def consolidate_events(sightings: list[dict]) -> list[dict]:
    """Greedy spatiotemporal clustering: merge if within 20m AND 15min."""
    if not sightings:
        return []

    sorted_sightings = sorted(sightings, key=lambda s: s.get("create_dt", ""))
    clusters: list[dict] = []

    for sight in sorted_sightings:
        lat = float(sight.get("lat", 0))
        lng = float(sight.get("lng", 0))
        dt = _parse_dt(sight.get("create_dt", "2000-01-01 00:00:00"))

        merged = False
        for cluster in clusters:
            time_diff = (dt - cluster["latest_dt"]).total_seconds() / 60.0
            if time_diff > TEMPORAL_THRESHOLD_MIN:
                continue
            c_lat = sum(cluster["lats"]) / len(cluster["lats"])
            c_lng = sum(cluster["lngs"]) / len(cluster["lngs"])
            dist = _haversine_m(lat, lng, c_lat, c_lng)
            if dist <= SPATIAL_THRESHOLD_M:
                cluster["lats"].append(lat)
                cluster["lngs"].append(lng)
                cluster["alerts"].append(sight)
                cluster["latest_dt"] = dt
                merged = True
                break

        if not merged:
            clusters.append({"lats": [lat], "lngs": [lng], "alerts": [sight], "latest_dt": dt})

    events = []
    for cluster in clusters:
        alerts = cluster["alerts"]
        centroid_lat = sum(cluster["lats"]) / len(cluster["lats"])
        centroid_lng = sum(cluster["lngs"]) / len(cluster["lngs"])
        best_alert = max(alerts, key=lambda a: int(a.get("upvote", 0)))
        address = best_alert.get("address", "")
        create_dt = alerts[0].get("create_dt", "")
        end_dt = alerts[-1].get("create_dt", create_dt)
        dt_start = _parse_dt(create_dt)
        dt_end = _parse_dt(end_dt)
        duration_min = (dt_end - dt_start).total_seconds() / 60.0
        total_up = sum(int(a.get("upvote", 0)) for a in alerts)
        total_down = sum(int(a.get("downvote", 0)) for a in alerts)

        from ghost_utils import compute_grid_cell
        grid_cell = compute_grid_cell(centroid_lat, centroid_lng)

        events.append({
            "lat": centroid_lat, "lng": centroid_lng, "address": address,
            "create_dt": create_dt, "end_dt": end_dt,
            "duration_min": duration_min, "report_count": len(alerts),
            "total_upvotes": total_up, "total_downvotes": total_down,
            "grid_cell": grid_cell, "district": "", "region": "",
        })
    return events
