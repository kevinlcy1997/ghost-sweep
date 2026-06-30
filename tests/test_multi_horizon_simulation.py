import pandas as pd

from analysis.simulate_real_location_risk import simulate_location_multi_horizon
from ghost_zones import compute_h3_zone, h3_zone_centroid


def _predictions(score):
    zone = compute_h3_zone(22.3154, 114.1698, resolution=8)
    lat, lng = h3_zone_centroid(zone)
    return pd.DataFrame(
        [
            {
                "target_time": "2026-06-25 17:00:00",
                "zone_id": zone,
                "district": "Mong Kok",
                "region": "Kowloon West",
                "zone_lat": lat,
                "zone_lng": lng,
                "score": score,
                "actual": 1,
            }
        ]
    )


def test_simulate_location_multi_horizon_returns_all_horizons():
    result = simulate_location_multi_horizon(
        lat=22.3154,
        lng=114.1698,
        target_time="2026-06-25 17:00:00",
        predictions_by_horizon={
            "30m": _predictions(0.8),
            "1h": _predictions(0.6),
            "2h": _predictions(0.4),
        },
    )

    assert set(result["horizons"]) == {"30m", "1h", "2h"}
    assert result["horizons"]["30m"]["score"] == 0.8
    assert result["horizons"]["2h"]["score"] == 0.4
