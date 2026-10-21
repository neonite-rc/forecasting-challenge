"""End-to-end tests on a tiny synthetic release-shaped dataset.

Mirrors what reviewers may do: swap the public data for a small internal smoke
dataset with the same file/column contract and run the documented command.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cwfc.config import GRID_SECONDS, HORIZON
from cwfc.predict import main as predict_main
from cwfc.pipeline import run_pipeline
from smoke_data import write_smoke_dataset


def test_pipeline_runs_on_smoke_dataset(tmp_path):
    data_dir = write_smoke_dataset(tmp_path / "challenge_data")
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
    data_dir = write_smoke_dataset(tmp_path / "challenge_data")
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
