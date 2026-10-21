"""End-to-end tests on a tiny synthetic release-shaped dataset.

Mirrors what reviewers may do: swap the public data for a small internal smoke
dataset with the same file/column contract and run the documented command.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cwfc.config import GRID_SECONDS, HORIZON
from cwfc.predict import main as predict_main
from cwfc.pipeline import run_pipeline


def _write_smoke_dataset(data_dir: Path, n_vms: int = 12, n_ts: int = 64) -> None:
    rng = np.random.default_rng(42)
    timestamps = np.arange(n_ts) * GRID_SECONDS
    rows = []
    for j, t in enumerate(timestamps):
        level = 8.0 + 0.05 * j + np.sin(j / 4.0)
        for i in range(n_vms):
            avg = float(np.clip(level + rng.normal(0, 1.0), 0.02, 99.9))
            rows.append(
                {
                    "timestamp": int(t),
                    "vm_id": f"vm{i:04d}",
                    "min_cpu": max(0.0, avg - 0.5),
                    "max_cpu": min(100.0, avg + 0.5),
                    "avg_cpu": avg,
                }
            )
    readings = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        {
            "vm_id": [f"vm{i:04d}" for i in range(n_vms)],
            "subscription_id": ["sub-1"] * n_vms,
            "deployment_id": ["dep-1"] * n_vms,
            "timestamp_vm_created": 0,
            "timestamp_vm_deleted": timestamps[-1] + GRID_SECONDS,
            "max_cpu": 50.0,
            "avg_cpu": 9.0,
            "p95_max_cpu": 40.0,
            "vm_category": ["Interactive"] * n_vms,
            "virtual_core_bucket": "2",
            "memory_gb_bucket": "4",
        }
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    readings.to_csv(data_dir / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
    metadata.to_csv(data_dir / "vm_metadata.csv.gz", index=False, compression="gzip")


def test_pipeline_runs_on_smoke_dataset(tmp_path):
    data_dir = tmp_path / "challenge_data"
    _write_smoke_dataset(data_dir)
    result = run_pipeline(data_dir)

    preds = result.predictions
    assert list(preds.columns) == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    assert len(preds) == HORIZON
    assert preds["timestamp"].is_monotonic_increasing
    assert preds["timestamp"].is_unique
    assert np.isfinite(preds["aggregate_cpu_utilization"]).all()
    assert preds["aggregate_cpu_utilization"].between(0, 100).all()
    assert (preds["active_vm_count"] >= 0).all()
    # holdout timestamps must be the final 24 grid timestamps
    expected_ts = np.arange(40, 64) * GRID_SECONDS
    np.testing.assert_array_equal(preds["timestamp"].to_numpy(), expected_ts)


def test_cli_writes_prediction_file(tmp_path):
    data_dir = tmp_path / "challenge_data"
    _write_smoke_dataset(data_dir)
    out = tmp_path / "out" / "predictions.csv"
    artifacts = tmp_path / "artifacts"
    predict_main(
        ["--data", str(data_dir), "--output", str(out), "--artifacts-dir", str(artifacts)]
    )
    written = pd.read_csv(out)
    assert len(written) == HORIZON
    assert list(written.columns) == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    leaderboards = list(artifacts.glob("cv_leaderboard_*.csv"))
    assert len(leaderboards) == 2
    for path in leaderboards:
        board = pd.read_csv(path)
        assert {"model", "rmse_mean", "mae_mean"}.issubset(board.columns)
        # selection metric must be sorted ascending
        assert board["rmse_mean"].is_monotonic_increasing
