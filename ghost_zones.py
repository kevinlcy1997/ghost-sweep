"""Zone assignment helpers for Hong Kong alert risk modeling."""

from __future__ import annotations

from h3 import cell_to_boundary, cell_to_latlng, latlng_to_cell

from ghost_districts import get_district
from ghost_utils import compute_grid_cell


DEFAULT_H3_RESOLUTION = 9


def compute_h3_zone(lat: float, lng: float, resolution: int = DEFAULT_H3_RESOLUTION) -> str:
    """Return the H3 cell ID for a latitude/longitude pair."""
    return latlng_to_cell(float(lat), float(lng), resolution)


def h3_zone_centroid(zone_id: str) -> tuple[float, float]:
    """Return the H3 cell centroid as ``(lat, lng)``."""
    lat, lng = cell_to_latlng(zone_id)
    return float(lat), float(lng)


def h3_zone_polygon(zone_id: str) -> list[list[float]]:
    """Return a GeoJSON-compatible polygon ring for an H3 cell."""
    boundary = cell_to_boundary(zone_id)
    ring = [[float(lng), float(lat)] for lat, lng in boundary]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def assign_zone(lat: float, lng: float, resolution: int = DEFAULT_H3_RESOLUTION) -> dict:
    """Assign grid, H3, district, and region fields for an alert coordinate."""
    district, region = get_district(float(lat), float(lng))
    zone_id = compute_h3_zone(float(lat), float(lng), resolution)
    zone_lat, zone_lng = h3_zone_centroid(zone_id)
    return {
        "grid_cell": compute_grid_cell(float(lat), float(lng)),
        "h3_zone": zone_id,
        "h3_resolution": resolution,
        "zone_lat": zone_lat,
        "zone_lng": zone_lng,
        "district": district,
        "region": region,
    }

