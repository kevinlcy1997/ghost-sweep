import pandas as pd

from analysis.analyze_spatial_model_errors import summarize_horizon


def test_summarize_horizon_reports_artifact_and_per_target_time_ranks():
    predictions = pd.DataFrame(
        [
            {
                "target_time": "2026-06-01 10:00:00",
                "zone_id": "a",
                "score": 0.9,
                "rank": 1,
                "actual": 0,
            },
            {
                "target_time": "2026-06-01 10:00:00",
                "zone_id": "b",
                "score": 0.8,
                "rank": 3,
                "actual": 1,
            },
            {
                "target_time": "2026-06-01 11:00:00",
                "zone_id": "c",
                "score": 0.85,
                "rank": 2,
                "actual": 1,
            },
        ]
    )

    artifact = summarize_horizon(predictions, "30m", k=1, rank_scope="artifact")
    per_target = summarize_horizon(predictions, "30m", k=1, rank_scope="per_target_time")

    assert artifact["top1_true_positives"] == 0
    assert artifact["top1_recall"] == 0.0
    assert per_target["top1_true_positives"] == 1
    assert per_target["top1_recall"] == 0.5
