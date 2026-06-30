import pytest

from ghost_ranking_metrics import (
    brier_score,
    expected_calibration_error,
    group_hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    risk_band,
    top_decile_lift,
)


def test_precision_at_k_scores_top_ranked_rows():
    assert precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 2) == 0.5


def test_recall_at_k_scores_captured_positives():
    assert recall_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 2) == 0.5


def test_top_decile_lift_compares_top_bucket_to_base_rate():
    labels = [1] + [0] * 9
    scores = [0.99] + [0.1] * 9

    assert top_decile_lift(labels, scores) == 10.0


def test_ranking_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        precision_at_k([1, 0], [0.2], 1)


def test_brier_score_is_mean_squared_probability_error():
    assert brier_score([1, 0, 1], [0.8, 0.2, 0.5]) == pytest.approx(0.11)


def test_expected_calibration_error_is_bounded_and_handles_empty_input():
    assert expected_calibration_error([], []) == 0.0

    value = expected_calibration_error([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], n_bins=2)

    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(0.35)


def test_group_hit_rate_at_k_counts_positive_groups_represented_in_top_predictions():
    labels = [1, 0, 1, 0, 1]
    scores = [0.9, 0.8, 0.2, 0.7, 0.1]
    groups = ["Central", "Central", "Yau Tsim Mong", "Wan Chai", "Sha Tin"]

    assert group_hit_rate_at_k(labels, scores, groups, 2) == pytest.approx(1 / 3)


def test_risk_band_maps_probabilities_to_named_bands():
    assert risk_band(0.01) == "low"
    assert risk_band(0.03) == "elevated"
    assert risk_band(0.08) == "high"
    assert risk_band(0.2) == "critical"
