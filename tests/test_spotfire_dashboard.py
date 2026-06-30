from analysis.build_spotfire_dashboard import render_html, write_dashboard


def test_dashboard_contains_required_interactive_sections(tmp_path):
    html = render_html(
        {
            "coverage": [],
            "sparsity": [],
            "hourProfile": [],
            "dayProfile": [],
            "neighborContext": [],
            "multiHorizon": [],
            "predictions30m": [],
            "predictions1h": [],
            "predictions2h": [],
            "manifest": {"artifact_groups": {}},
        }
    )

    assert "Ghost Sweep HK Fixed-Grid Analytics" in html
    assert "HK H3 Coverage Map" in html
    assert "Model Evaluation and Predictions" in html
    assert "Traceable Artifacts" in html
    assert "function render()" in html

    output = write_dashboard(tmp_path / "dashboard.html")
    assert output.exists()
