from pathlib import Path
import json

from analysis.build_current_model_data_analysis_report import (
    build_report_payload,
    output_path_for_date,
    render_report_html,
    write_report,
)


def test_render_report_writes_static_dated_html(tmp_path: Path):
    payload = build_report_payload(
        raw={
            "total_alerts": 10236,
            "date_range": ["2026-06-13 09:58:34", "2026-07-10 14:32:03"],
            "field_presence": {"lat": 10236, "lng": 10236, "region": 0, "district": 0},
            "last_7_daily": [
                ["2026-07-04", 18],
                ["2026-07-05", 639],
                ["2026-07-06", 556],
            ],
            "hour_top5": [[16, 919], [20, 901]],
            "dow": {"Wed": 1986, "Thu": 1662},
        },
        reconstructed_geo={
            "usable_events": 10236,
            "unique_zones": 880,
            "unique_regions": 5,
            "unique_districts": 18,
            "top_regions": [["Kowloon West", 3688], ["New Territories North", 2724]],
            "top_districts": [["Yau Tsim Mong", 2180], ["Yuen Long", 1329]],
        },
        stage1={
            "30m": {"positive_rate": 0.5, "rows": 486, "holdout_base_rate": 0.7463},
            "60m": {"positive_rate": 0.5556, "rows": 486, "holdout_base_rate": 0.7812},
            "120m": {"positive_rate": 0.6358, "rows": 486, "holdout_base_rate": 0.9259},
        },
        stage2={
            "feature_quality": {
                "rows": 279840,
                "unique_target_times": 318,
                "unique_zones": 880,
                "null_rate_top10": [["region", 0.2566]],
                "zero_share_top10": [["zone_event_count_1h", 0.9928]],
            },
            "30m": {
                "positive_rate": 0.0037,
                "sampled_neg_to_pos_ratio": 5.1,
                "rows": 279840,
                "positives": 1027,
            },
            "60m": {
                "positive_rate": 0.0073,
                "sampled_neg_to_pos_ratio": 5.1,
                "rows": 279840,
                "positives": 2031,
            },
            "120m": {
                "positive_rate": 0.0136,
                "sampled_neg_to_pos_ratio": 5.1,
                "rows": 279840,
                "positives": 3804,
            },
        },
        writeup={
            "executive_read": "The current model still fits the data, but Stage 1 is less selective in the hotter regime.",
            "what_so_what_now_what": {
                "what": "The raw feed lost admin text fields, but coordinates still support enrichment.",
                "so_what": "The accepted design still works, but DB refresh and geo backfill are now hard dependencies.",
                "now_what": "Refresh the DB, rerun the baseline, and keep the coordinate backfill path intact.",
            },
        },
    )

    html_text = render_report_html(payload)
    out_path = write_report(payload, root_dir=tmp_path, report_date="2026-07-10")

    assert output_path_for_date(tmp_path, "2026-07-10") == out_path
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == html_text
    assert "<title>Current Model Data Analysis</title>" in html_text
    assert "Stage 1 30m positive rate" in html_text
    assert "Stage 2 30m positive rate" in html_text
    assert "What / So What / Now What" in html_text
    assert "Yau Tsim Mong" in html_text


def test_collect_report_data_uses_project_inputs(tmp_path: Path):
    from analysis.build_current_model_data_analysis_report import collect_report_data

    source = tmp_path / "ghost_alerts.json"
    source.write_text(
        json.dumps(
            {
                "alerts": {
                    "1": {
                        "lat": 22.3154,
                        "lng": 114.1698,
                        "create_dt": "2026-07-09 10:00:00",
                        "address": "Central",
                        "name": "A",
                    },
                    "2": {
                        "lat": 22.3160,
                        "lng": 114.1703,
                        "create_dt": "2026-07-10 11:00:00",
                        "address": "Wan Chai",
                        "name": "B",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    payload = collect_report_data(alerts_path=source)

    assert payload["raw"]["total_alerts"] == 2
    assert payload["reconstructed_geo"]["usable_events"] == 2
    assert "30m" in payload["stage1"]
    assert "30m" in payload["stage2"]
    assert payload["current_model_design"]["stage2_numeric_feature_count"] > 0
