# Copyright (c) 2026 Biswaketan <https://biswaketan.in>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: do not remove (PolyForm Notices/Violations) - build CWFC-NC1-7Q2M9X4D.

"""Command-line entry point producing the required ``predictions.csv``.

Example:
    python -m cwfc.predict --data challenge_data --output predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m cwfc.predict",
        description="Forecast the 24-step, two-hour holdout.",
    )
    parser.add_argument(
        "--about",
        action="store_true",
        help="print authorship, build and license provenance, then exit",
    )
    parser.add_argument(
        "--data",
        required=False,
        help="directory containing vm_cpu_readings.csv.gz and vm_metadata.csv.gz",
    )
    parser.add_argument("--output", default="predictions.csv", help="output CSV path")
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="optional directory for CV leaderboards and residual intervals",
    )
    args = parser.parse_args(argv)
    if not args.about and not args.data:
        parser.error("--data is required unless --about is used")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.about:
        from ._about import about_text

        print(about_text())
        return
    result = run_pipeline(args.data)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(output, index=False)

    print(f"wrote {len(result.predictions)} predictions to {output}")
    for target, tr in result.target_results.items():
        print(f"  {target}: selected model = {tr.selected_model}")
        score = tr.holdout_score
        print(
            "    holdout MAE={mae:.4f} RMSE={rmse:.4f} sMAPE={smape:.3f}% "
            "rel.MAE(persistence)={rel:.3f}".format(
                mae=score["mae"],
                rmse=score["rmse"],
                smape=score["smape"],
                rel=score["mae_relative_to_persistence"],
            )
        )

    if args.artifacts_dir:
        art = Path(args.artifacts_dir)
        art.mkdir(parents=True, exist_ok=True)
        for target, tr in result.target_results.items():
            tr.leaderboard.to_csv(art / f"cv_leaderboard_{target}.csv", index=False)
            tr.intervals.to_csv(art / f"prediction_intervals_{target}.csv", index=False)
        print(f"wrote CV artifacts to {art}")


if __name__ == "__main__":
    main()
