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


def test_add_engineered_ranking_features_adds_ring2_context():
    zone = compute_h3_zone(22.3154, 114.1698, resolution=8)
    ring1 = next(iter(h3.grid_ring(zone, 1)))
    ring2 = next(iter(h3.grid_ring(zone, 2)))

    rows = pd.DataFrame(
        [
            {
                "target_time": pd.Timestamp("2026-06-20 06:00:00"),
                "zone_id": zone,
                "hour": 6,
                "day_of_week": 5,
                "zone_event_count_3h": 0,
                "zone_event_count_24h": 2,
                "zone_event_count_7d": 4,
                "district_event_count_3h": 0,
                "district_event_count_24h": 10,
            },
            {
                "target_time": pd.Timestamp("2026-06-20 06:00:00"),
                "zone_id": ring1,
                "hour": 6,
                "day_of_week": 5,
                "zone_event_count_3h": 0,
                "zone_event_count_24h": 5,
                "zone_event_count_7d": 8,
                "district_event_count_3h": 0,
                "district_event_count_24h": 10,
            },
            {
                "target_time": pd.Timestamp("2026-06-20 06:00:00"),
                "zone_id": ring2,
                "hour": 6,
                "day_of_week": 5,
                "zone_event_count_3h": 0,
                "zone_event_count_24h": 7,
                "zone_event_count_7d": 11,
                "district_event_count_3h": 0,
                "district_event_count_24h": 10,
            },
        ]
    )

    enhanced = add_engineered_ranking_features(rows)
    base = enhanced.loc[enhanced["zone_id"] == zone].iloc[0]

    assert base["neighbor_event_count_24h"] == 5
    assert base["ring2_event_count_24h"] == 7
    assert base["ring2_event_count_7d"] == 11
    assert base["ring2_active_zones_24h"] == 1
    assert base["ring2_to_ring1_24h_ratio"] == 7 / 5


def test_add_engineered_ranking_features_adds_district_relative_features():
    rows = pd.DataFrame(
        [
            {
                "target_time": pd.Timestamp("2026-06-20 06:00:00"),
                "zone_id": compute_h3_zone(22.3154, 114.1698, resolution=8),
                "district": "Yau Tsim Mong",
                "hour": 6,
                "day_of_week": 5,
                "zone_event_count_3h": 0,
                "zone_event_count_24h": 8,
                "zone_event_count_7d": 20,
                "zone_same_hour_rate": 0.4,
                "district_event_count_3h": 0,
                "district_event_count_24h": 10,
            },
            {
                "target_time": pd.Timestamp("2026-06-20 06:00:00"),
                "zone_id": compute_h3_zone(22.318, 114.171, resolution=8),
                "district": "Yau Tsim Mong",
                "hour": 6,
                "day_of_week": 5,
                "zone_event_count_3h": 0,
                "zone_event_count_24h": 2,
                "zone_event_count_7d": 10,
                "zone_same_hour_rate": 0.1,
                "district_event_count_3h": 0,
                "district_event_count_24h": 10,
            },
        ]
    )

    enhanced = add_engineered_ranking_features(rows)
    hot = enhanced.sort_values("zone_event_count_24h", ascending=False).iloc[0]
    cold = enhanced.sort_values("zone_event_count_24h", ascending=True).iloc[0]

    assert hot["zone_24h_share_of_district"] == 0.8
    assert hot["zone_7d_rank_in_district"] == 1
    assert hot["zone_same_hour_percentile_in_district"] == 1.0
    assert cold["zone_24h_share_of_district"] == 0.2


def test_add_engineered_ranking_features_adds_hotspot_distance_features():
    zone = compute_h3_zone(22.3154, 114.1698, resolution=8)
    nearby = next(iter(h3.grid_ring(zone, 1)))
    far = compute_h3_zone(22.38, 114.19, resolution=8)
    zone_lat, zone_lng = h3.cell_to_latlng(zone)
    nearby_lat, nearby_lng = h3.cell_to_latlng(nearby)
    far_lat, far_lng = h3.cell_to_latlng(far)

    rows = pd.DataFrame(
        [
            {
                "target_time": pd.Timestamp("2026-06-20 06:00:00"),
                "zone_id": zone,
                "district": "Yau Tsim Mong",
                "zone_lat": zone_lat,
                "zone_lng": zone_lng,
                "hour": 6,
                "day_of_week": 5,
                "zone_event_count_3h": 0,
                "zone_event_count_24h": 0,
                "zone_event_count_7d": 0,
                "district_event_count_3h": 0,
                "district_event_count_24h": 8,
            },
            {
                "target_time": pd.Timestamp("2026-06-20 06:00:00"),
                "zone_id": nearby,
                "district": "Yau Tsim Mong",
                "zone_lat": nearby_lat,
                "zone_lng": nearby_lng,
                "hour": 6,
                "day_of_week": 5,
                "zone_event_count_3h": 1,
                "zone_event_count_24h": 2,
                "zone_event_count_7d": 5,
                "district_event_count_3h": 0,
                "district_event_count_24h": 8,
            },
            {
                "target_time": pd.Timestamp("2026-06-20 06:00:00"),
                "zone_id": far,
                "district": "Yau Tsim Mong",
                "zone_lat": far_lat,
                "zone_lng": far_lng,
                "hour": 6,
                "day_of_week": 5,
                "zone_event_count_3h": 0,
                "zone_event_count_24h": 6,
                "zone_event_count_7d": 20,
                "district_event_count_3h": 0,
                "district_event_count_24h": 8,
            },
        ]
    )

    enhanced = add_engineered_ranking_features(rows)
    base = enhanced.loc[enhanced["zone_id"] == zone].iloc[0]

    assert 0 < base["distance_to_nearest_event_3h_m"] < 1500
    assert base["distance_to_nearest_event_24h_m"] < 1500
    assert base["distance_to_district_recent_centroid_24h_m"] > 0


def test_add_engineered_ranking_features_can_join_road_context(tmp_path):
    zone = compute_h3_zone(22.3154, 114.1698, resolution=8)
    road_context = tmp_path / "road_context.csv"
    road_context.write_text(
        "\n".join(
            [
                "h3_zone,nearest_road_m,road_segment_count,road_source_mismatch,has_drivable_road",
                f"{zone},18.5,3,0,1",
            ]
        ),
        encoding="utf-8",
    )
    rows = pd.DataFrame(
        [
            {
                "target_time": pd.Timestamp("2026-06-20 06:00:00"),
                "zone_id": zone,
                "hour": 6,
                "day_of_week": 5,
                "zone_event_count_3h": 0,
                "zone_event_count_24h": 0,
                "zone_event_count_7d": 0,
                "district_event_count_3h": 0,
                "district_event_count_24h": 0,
            }
        ]
    )

    enhanced = add_engineered_ranking_features(rows, road_context_path=road_context)
    base = enhanced.iloc[0]

    assert base["nearest_road_m"] == 18.5
    assert base["road_segment_count"] == 3
    assert base["road_source_mismatch"] == 0
    assert base["has_drivable_road"] == 1
