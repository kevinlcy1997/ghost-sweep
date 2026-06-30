from analysis.run_model_iteration import horizon_slug, target_for_horizon


def test_horizon_slug_formats_minutes_and_hours():
    assert horizon_slug(30) == "30m"
    assert horizon_slug(60) == "1h"
    assert horizon_slug(120) == "2h"


def test_target_for_horizon_names_alert_columns():
    assert target_for_horizon(30) == "alert_next_30m"
    assert target_for_horizon(60) == "alert_next_1h"
    assert target_for_horizon(120) == "alert_next_2h"
