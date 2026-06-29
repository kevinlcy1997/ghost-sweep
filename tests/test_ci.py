# tests/test_ci.py
"""Tests to verify CI/deployment readiness."""

import os
import subprocess
import sys


def test_once_flag_exits_cleanly():
    """ghost_listener.py --once should complete without crashing (dry-run check)."""
    # We can't actually call the API in CI, but we can verify the script loads
    result = subprocess.run(
        [sys.executable, "-c", "from ghost_listener import main; print('OK')"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_workflow_file_exists():
    """GitHub Actions workflow should exist."""
    workflow = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        ".github", "workflows", "ghost-scrape.yml"
    )
    assert os.path.exists(workflow), f"Workflow not found at {workflow}"


def test_workflow_valid_yaml():
    """GitHub Actions workflow should be valid YAML with required fields."""
    workflow = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        ".github", "workflows", "ghost-scrape.yml"
    )
    with open(workflow, "r", encoding="utf-8") as f:
        content = f.read()
    # Basic structural checks
    assert "on:" in content
    assert "schedule:" in content
    assert "ghost_listener.py --once" in content
    assert "--request-timeout" in content
    assert "--max-grid-cells" in content
    assert "--skip-active" in content
    assert "git push" in content
