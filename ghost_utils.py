# ghost_utils.py
"""Shared utilities for Ghost Sweep modules."""


def compute_grid_cell(lat: float, lng: float) -> str:
    """Snap lat/lng to 0.005° grid cell (~500m)."""
    cell_lat = round(lat / 0.005) * 0.005
    cell_lng = round(lng / 0.005) * 0.005
    return f"{cell_lat:.3f}_{cell_lng:.3f}"
