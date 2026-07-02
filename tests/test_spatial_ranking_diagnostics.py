import pandas as pd

from analysis.diagnose_spatial_ranking import summarize_topk_near_misses


def test_summarize_topk_near_misses_counts_neighbor_hit():
    rows = pd.DataFrame(
        [
            {
                "target_time": "2026-06-01 10:00:00",
                "zone_id": "a",
                "spatial_probability": 0.9,
                "actual": 0,
                "district": "D1",
                "region": "R1",
            },
            {
                "target_time": "2026-06-01 10:00:00",
                "zone_id": "b",
                "spatial_probability": 0.1,
                "actual": 1,
                "district": "D1",
                "region": "R1",
            },
        ]
    )

    summary = summarize_topk_near_misses(
        rows,
        k=1,
        neighbor_lookup={"a": {"b"}},
        ring2_lookup={"a": set()},
    )

    assert summary.iloc[0]["exact_hits"] == 0
    assert summary.iloc[0]["ring1_hits"] == 1
    assert summary.iloc[0]["district_hits"] == 1
    assert summary.iloc[0]["region_hits"] == 1
