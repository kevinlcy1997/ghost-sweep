import pandas as pd

from analysis.simulate_real_location_risk import simulate_location_risk
from ghost_zones import compute_h3_zone, h3_zone_centroid


def test_simulate_location_risk_scores_exact_zone_and_lists_hotspots():
    zone = compute_h3_zone(22.3154, 114.1698, resolution=8)
    zone_lat, zone_lng = h3_zone_centroid(zone)
    other_zone = compute_h3_zone(22.2819, 114.1589, resolution=8)
    other_lat, other_lng = h3_zone_centroid(other_zone)

    predictions = pd.DataFrame(
        [
            {
                "target_time": "2026-06-25 17:00:00",
                "zone_id": zone,
                "district": "Mong Kok",
                "region": "Kowloon West",
                "zone_lat": zone_lat,
                "zone_lng": zone_lng,
                "score": 0.91,
                "actual": 1,
            },
            {
                "target_time": "2026-06-25 17:00:00",
                "zone_id": other_zone,
                "district": "Central",
                "region": "Hong Kong Island",
                "zone_lat": other_lat,
                "zone_lng": other_lng,
                "score": 0.12,
                "actual": 0,
            },
        ]
    )

    result = simulate_location_risk(
        lat=22.3154,
        lng=114.1698,
        target_time="2026-06-25 17:00:00",
        predictions=predictions,
        resolution=8,
    )

    assert result["zone_id"] == zone
    assert result["score"] == 0.91
    assert result["rank"] == 1
    assert result["risk_band"] == "high"
    assert result["nearest_hot_zones"][0]["zone_id"] == zone
    assert "Mong Kok" in result["summary"]
