from analysis.run_resolution_comparison import choose_practical_resolution


def test_choose_practical_resolution_penalizes_sparse_one_off_cells():
    rows = [
        {
            "resolution": 8,
            "precision_at_20": 0.80,
            "top_decile_lift": 4.0,
            "one_off_zone_rate": 0.10,
            "active_zones": 100,
        },
        {
            "resolution": 9,
            "precision_at_20": 0.85,
            "top_decile_lift": 4.2,
            "one_off_zone_rate": 0.25,
            "active_zones": 250,
        },
        {
            "resolution": 10,
            "precision_at_20": 0.90,
            "top_decile_lift": 4.4,
            "one_off_zone_rate": 0.80,
            "active_zones": 900,
        },
    ]

    chosen = choose_practical_resolution(rows)

    assert chosen["resolution"] == 9
