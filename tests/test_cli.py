from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

DATA_DIR = Path("data/raw/olist")


@pytest.mark.slow
def test_cli_end_to_end(tmp_path):
    """The full pipeline produces markdown + html + charts."""
    out = tmp_path / "reports"
    result = subprocess.run(
        [sys.executable, "-m", "investigator.cli", "--data", str(DATA_DIR), "--out", str(out), "--no-llm"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert (out / "report.md").exists()
    assert (out / "report.html").exists()
    assert (out / "chart_monthly_revenue.png").exists()