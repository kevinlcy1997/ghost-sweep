"""Hong Kong district assignment using official district boundary polygons."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from shapely.geometry import Point, shape


ROOT = Path(__file__).resolve().parent
BOUNDARY_PATH = ROOT / "analysis" / "geo" / "hksar_18_district_boundary.json"

DISTRICT_REGIONS: dict[str, str] = {
    "Central & Western": "Hong Kong Island",
    "Wan Chai": "Hong Kong Island",
    "Eastern": "Hong Kong Island",
    "Southern": "Hong Kong Island",
    "Yau Tsim Mong": "Kowloon West",
    "Sham Shui Po": "Kowloon West",
    "Kowloon City": "Kowloon West",
    "Wong Tai Sin": "Kowloon East",
    "Kwun Tong": "Kowloon East",
    "Tsuen Wan": "New Territories South",
    "Kwai Tsing": "New Territories South",
    "Sha Tin": "New Territories South",
    "Sai Kung": "New Territories South",
    "Islands": "New Territories South",
    "Tuen Mun": "New Territories North",
    "Yuen Long": "New Territories North",
    "North": "New Territories North",
    "Tai Po": "New Territories North",
}

DISTRICT_STATIONS: dict[str, dict] = {
    "Central & Western": {"lat": 22.2816, "lng": 114.1585},
    "Wan Chai": {"lat": 22.2780, "lng": 114.1720},
    "Eastern": {"lat": 22.2870, "lng": 114.2190},
    "Southern": {"lat": 22.2470, "lng": 114.1580},
    "Yau Tsim Mong": {"lat": 22.3193, "lng": 114.1694},
    "Sham Shui Po": {"lat": 22.3310, "lng": 114.1590},
    "Kowloon City": {"lat": 22.3280, "lng": 114.1870},
    "Wong Tai Sin": {"lat": 22.3420, "lng": 114.1930},
    "Kwun Tong": {"lat": 22.3130, "lng": 114.2250},
    "Tsuen Wan": {"lat": 22.3710, "lng": 114.1140},
    "Kwai Tsing": {"lat": 22.3560, "lng": 114.1300},
    "Sha Tin": {"lat": 22.3810, "lng": 114.1880},
    "Sai Kung": {"lat": 22.3170, "lng": 114.2590},
    "Islands": {"lat": 22.3080, "lng": 113.9185},
    "Tuen Mun": {"lat": 22.3910, "lng": 113.9770},
    "Yuen Long": {"lat": 22.4440, "lng": 114.0220},
    "North": {"lat": 22.5030, "lng": 114.1280},
    "Tai Po": {"lat": 22.4510, "lng": 114.1680},
}


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_m = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@lru_cache(maxsize=1)
def _load_district_boundaries() -> tuple[tuple[str, str, object], ...]:
    if not BOUNDARY_PATH.exists():
        return tuple()

    data = json.loads(BOUNDARY_PATH.read_text(encoding="utf-8-sig"))
    boundaries = []
    for feature in data.get("features", []):
        district = feature.get("properties", {}).get("District")
        if not district:
            continue
        region = DISTRICT_REGIONS.get(district, "Unknown")
        boundaries.append((district, region, shape(feature["geometry"])))
    return tuple(boundaries)


def _nearest_station_district(lat: float, lng: float) -> tuple[str, str]:
    best_dist = float("inf")
    best_name = "Unknown"
    for name, info in DISTRICT_STATIONS.items():
        dist = _haversine_m(lat, lng, info["lat"], info["lng"])
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name, DISTRICT_REGIONS.get(best_name, "Unknown")


@lru_cache(maxsize=4096)
def get_district(lat: float, lng: float) -> tuple[str, str]:
    """Return the official Hong Kong district and broader modeling region."""
    point = Point(float(lng), float(lat))
    boundaries = _load_district_boundaries()

    for district, region, geometry in boundaries:
        if geometry.covers(point):
            return district, region

    if boundaries:
        district, region, _ = min(
            (
                (district, region, geometry.distance(point))
                for district, region, geometry in boundaries
            ),
            key=lambda item: item[2],
        )
        return district, region

    return _nearest_station_district(float(lat), float(lng))


def assign_district_to_events(events: list[dict]) -> list[dict]:
    for event in events:
        district, region = get_district(event["lat"], event["lng"])
        event["district"] = district
        event["region"] = region
    return events
