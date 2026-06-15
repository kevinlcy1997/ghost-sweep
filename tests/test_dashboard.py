"""Tests for generate_dashboard.py — popup content."""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generate_dashboard import load_data, compute_stats, generate_html


@pytest.fixture
def sample_json(tmp_path):
    """Create a minimal ghost_alerts.json and return its path."""
    data = {
        "alerts": {
            "1001": {
                "lat": "22.335",
                "lng": "114.165",
                "address": "Park In Street, Kowloon City",
                "create_dt": "2025-06-10 14:30:00",
                "upvote": "3",
                "downvote": "0",
            },
            "1002": {
                "lat": "22.336",
                "lng": "114.166",
                "address": "Lung Kong Road, Kowloon City",
                "create_dt": "2025-06-10 15:45:00",
                "upvote": "1",
                "downvote": "0",
            },
        },
        "meta": {},
    }
    path = tmp_path / "alerts.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _generate(sample_json):
    alerts, meta = load_data(sample_json)
    stats = compute_stats(alerts)
    return generate_html(alerts, stats, meta)


class TestPopupContent:
    """Popups on the map must show event address and time."""

    def test_popup_contains_address(self, sample_json):
        html = _generate(sample_json)
        assert "Park In Street" in html, "Popup should contain the alert address"

    def test_popup_contains_time(self, sample_json):
        html = _generate(sample_json)
        assert "14:30" in html, "Popup should contain the alert time"

    def test_popup_contains_second_address(self, sample_json):
        html = _generate(sample_json)
        assert "Lung Kong Road" in html, "Popup should show addresses for all alerts"

    def test_marker_data_has_address_and_time(self, sample_json):
        """The JS marker data array should carry address and create_dt fields."""
        html = _generate(sample_json)
        # Find the JSON array passed to JS for markers
        # It should contain objects with 'address' and 'create_dt' keys
        assert '"address"' in html, "Marker data should include address field"
        assert '"create_dt"' in html, "Marker data should include create_dt field"
