"""Out-of-box guarantee: the root shim works without an editable install."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from smoke_data import write_smoke_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_shim_runs_in_clean_subprocess(tmp_path):
    data_dir = write_smoke_dataset(tmp_path / "challenge_data")
    out = tmp_path / "predictions.csv"
    # Deliberately run as a fresh subprocess without importing cwfc first,
    # mirroring a reviewer on a clean checkout with dependencies installed.
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "predict.py"),
         "--data", str(data_dir), "--output", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    written = pd.read_csv(out)
    assert len(written) == 24
    assert list(written.columns) == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
