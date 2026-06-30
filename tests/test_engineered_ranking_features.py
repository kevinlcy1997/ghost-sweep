import math

import h3
import pandas as pd

from ghost_ranking_features import add_engineered_ranking_features
from ghost_zones import compute_h3_zone


def test_add_engineered_ranking_features_adds_neighbor_momentum_and_cyclical_columns():
    zone = compute_h3_zone(22.3154, 114.1698, resolution=8)
    neighbor = next(cell for cell in h3.grid_disk(zone, 1) if cell != zone)
    far_zone = compute_h3_zone(22.38, 114.19, resolution=8)

    rows = pd.DataFrame(
        [
            {
                "target_time": pd.Timestamp("2026-06-20 06:00:00"),
                "zone_id": zone,
                "hour": 6,
                "day_of_week": 5,
                "zone_event_count_3h": 2,
                "zone_event_count_24h": 10,
                "zone_event_count_7d": 100,
                "district_event_count_3h": 5,
                "district_event_count_24h": 20,
            },
            {
                "target_time": pd.Timestamp("2026-06-20 06:00:00"),
                "zone_id": neighbor,
                "hour": 6,
                "day_of_week": 5,
                "zone_event_count_3h": 4,
                "zone_event_count_24h": 8,
                "zone_event_count_7d": 40,
                "district_event_count_3h": 5,
                "district_event_count_24h": 20,
            },
            {
                "target_time": pd.Timestamp("2026-06-20 06:00:00"),
                "zone_id": far_zone,
                "hour": 6,
                "day_of_week": 5,
                "zone_event_count_3h": 7,
                "zone_event_count_24h": 14,
                "zone_event_count_7d": 70,
                "district_event_count_3h": 5,
                "district_event_count_24h": 20,
            },
        ]
    )

    enhanced = add_engineered_ranking_features(rows)
    base = enhanced.loc[enhanced["zone_id"] == zone].iloc[0]

    assert base["neighbor_event_count_3h"] == 4
    assert base["neighbor_event_count_24h"] == 8
    assert base["neighbor_active_zones_24h"] == 1
    assert base["zone_3h_to_24h_ratio"] == 0.2
    assert base["zone_24h_to_7d_ratio"] == 0.1
    assert base["district_3h_to_24h_ratio"] == 0.25
    assert math.isclose(base["hour_sin"], 1.0, abs_tol=1e-9)
    assert math.isclose(base["hour_cos"], 0.0, abs_tol=1e-9)
