# ghost_districts.py
"""Police district assignment via nearest-station proximity."""

import math
from functools import lru_cache

DISTRICT_STATIONS: dict[str, dict] = {
    "Eastern": {"lat": 22.2870, "lng": 114.2190, "region": "Hong Kong Island"},
    "Wan Chai": {"lat": 22.2780, "lng": 114.1720, "region": "Hong Kong Island"},
    "Central": {"lat": 22.2816, "lng": 114.1585, "region": "Hong Kong Island"},
    "Western": {"lat": 22.2870, "lng": 114.1420, "region": "Hong Kong Island"},
    "Wong Tai Sin": {"lat": 22.3420, "lng": 114.1930, "region": "Kowloon East"},
    "Kwun Tong": {"lat": 22.3130, "lng": 114.2250, "region": "Kowloon East"},
    "Tseung Kwan O": {"lat": 22.3170, "lng": 114.2590, "region": "Kowloon East"},
    "Sau Mau Ping": {"lat": 22.3290, "lng": 114.2320, "region": "Kowloon East"},
    "Yau Tsim": {"lat": 22.2980, "lng": 114.1720, "region": "Kowloon West"},
    "Mong Kok": {"lat": 22.3193, "lng": 114.1694, "region": "Kowloon West"},
    "Sham Shui Po": {"lat": 22.3310, "lng": 114.1590, "region": "Kowloon West"},
    "Kowloon City": {"lat": 22.3280, "lng": 114.1870, "region": "Kowloon West"},
    "Tai Po": {"lat": 22.4510, "lng": 114.1680, "region": "New Territories North"},
    "Tuen Mun": {"lat": 22.3910, "lng": 113.9770, "region": "New Territories North"},
    "Yuen Long": {"lat": 22.4440, "lng": 114.0220, "region": "New Territories North"},
    "Border": {"lat": 22.5030, "lng": 114.1280, "region": "New Territories North"},
    "Tsuen Wan": {"lat": 22.3710, "lng": 114.1140, "region": "New Territories South"},
    "Kwai Tsing": {"lat": 22.3560, "lng": 114.1300, "region": "New Territories South"},
    "Sha Tin": {"lat": 22.3810, "lng": 114.1880, "region": "New Territories South"},
    "Airport": {"lat": 22.3080, "lng": 113.9185, "region": "New Territories South"},
    "Lantau": {"lat": 22.2660, "lng": 113.9430, "region": "New Territories South"},
}


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@lru_cache(maxsize=4096)
def get_district(lat: float, lng: float) -> tuple[str, str]:
    best_dist = float("inf")
    best_name = ""
    best_region = ""
    for name, info in DISTRICT_STATIONS.items():
        d = _haversine_m(lat, lng, info["lat"], info["lng"])
        if d < best_dist:
            best_dist = d
            best_name = name
            best_region = info["region"]
    return best_name, best_region


def assign_district_to_events(events: list[dict]) -> list[dict]:
    for ev in events:
        district, region = get_district(ev["lat"], ev["lng"])
        ev["district"] = district
        ev["region"] = region
    return events
