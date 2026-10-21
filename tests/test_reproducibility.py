"""Determinism and end-to-end analysis guarantees."""
from __future__ import annotations

import warnings

import pandas as pd

from cwfc.predict import main as predict_main
from cwfc.pipeline import run_pipeline
from smoke_data import write_smoke_dataset

warnings.filterwarnings("ignore")


def test_pipeline_is_deterministic(tmp_path):
    data_dir = write_smoke_dataset(tmp_path / "challenge_data")
    first = run_pipeline(data_dir).predictions
    second = run_pipeline(data_dir).predictions
    pd.testing.assert_frame_equal(first, second)


def test_cli_and_analysis_run_out_of_the_box(tmp_path):
    data_dir = write_smoke_dataset(tmp_path / "challenge_data")
    out = tmp_path / "predictions.csv"
    predict_main(
        [
            "--data", str(data_dir),
            "--output", str(out),
            "--artifacts-dir", str(tmp_path / "artifacts"),
        ]
    )
    assert out.exists()

    # The full analysis script must also run on a small release-shaped dataset.
    from analysis.run_analysis import main as analysis_main

    analysis_main(str(data_dir), str(tmp_path / "reports"))
    figures = list((tmp_path / "reports" / "figures").glob("*.png"))
    tables = list((tmp_path / "reports" / "tables").glob("*"))
    assert len(figures) == 8
    assert len(tables) >= 5
