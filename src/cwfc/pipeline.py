"""End-to-end forecasting pipeline shared by the CLI, tests and analysis.

Order of operations (leakage-safe by construction):

1. Build the complete aggregate target frame from the VM-level panel.
2. Apply the chronological 110/24 split.
3. Select each target's model purely via rolling-origin CV on the first 110
   observations.
4. Refit the selected model on the full 110-point training window and forecast
   the 24 holdout steps. Holdout targets are never touched until scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HORIZON, TARGETS, TRAIN_SIZE
from .data import audit_panel, build_target_frame, load_readings, split_train_holdout
from .evaluation import (
    capacity_trigger,
    cross_validate,
    empirical_interval,
    pooled_cv_residuals,
    residual_intervals,
    score_holdout,
    select_best_model,
)
from .models import available_models, build_model

# Rolling-origin CV inside the 110-point training window: ten origins
# (48, 52, ..., 86), each evaluating a full 24-step horizon. The last fold
# ends exactly at the training/holdout boundary, replicating the production
# task. Ten folds yield 240 out-of-sample residuals for model selection and
# interval calibration. On smaller smoke datasets the window shrinks
# adaptively while keeping at least one full-horizon fold.
CV_MIN_TRAIN = 48
CV_STEP = 4
CV_SMOKE_FLOOR = 12


def _cv_window(n_train: int, horizon: int) -> tuple[int, int]:
    min_train = min(CV_MIN_TRAIN, n_train - horizon)
    if min_train < CV_SMOKE_FLOOR:
        # Very small smoke dataset: keep one fold with the longest viable
        # history, but never below four observations per model fit.
        min_train = max(n_train - horizon, 4)
    if min_train + horizon > n_train or min_train < 4:
        raise ValueError(
            f"training window too short for CV: {n_train} observations, "
            f"need at least {horizon + 4}"
        )
    step = min(CV_STEP, n_train - min_train - horizon + 1)
    return min_train, max(step, 1)


@dataclass
class TargetResult:
    target: str
    selected_model: str
    leaderboard: pd.DataFrame
    forecast: np.ndarray
    holdout_actual: np.ndarray
    holdout_score: dict
    intervals: pd.DataFrame
    folds: list = field(default_factory=list)


@dataclass
class PipelineResult:
    predictions: pd.DataFrame
    target_results: dict[str, TargetResult]
    audit: dict


def run_pipeline(data_dir: str | Path) -> PipelineResult:
    readings = load_readings(data_dir)
    audit = audit_panel(readings)
    target_frame = build_target_frame(readings)
    train, holdout = split_train_holdout(target_frame, TRAIN_SIZE, HORIZON)
    if len(holdout) != HORIZON:
        raise ValueError(f"expected {HORIZON} holdout rows, found {len(holdout)}")

    results: dict[str, TargetResult] = {}
    forecasts: dict[str, np.ndarray] = {}
    candidates = available_models()

    cv_min_train, cv_step = _cv_window(len(train), HORIZON)

    for target in TARGETS:
        history = train[target].to_numpy(dtype=float)
        actual = holdout[target].to_numpy(dtype=float)

        best_name, board = select_best_model(
            history, candidates, HORIZON, cv_min_train, cv_step
        )
        folds = cross_validate(history, best_name, HORIZON, cv_min_train, cv_step)
        forecaster = build_model(best_name).fit(history)
        forecast = np.asarray(forecaster.forecast(HORIZON), dtype=float)

        if target == "active_vm_count":
            forecast = np.rint(forecast).clip(min=0.0)
        else:
            forecast = np.clip(forecast, 0.0, 100.0)

        persistence = build_model("persistence").fit(history).forecast(HORIZON)
        score = score_holdout(actual, forecast, persistence)

        pooled = pooled_cv_residuals(folds)
        lower, upper = empirical_interval(forecast, pooled)
        intervals = pd.DataFrame(
            {
                "step": np.arange(1, HORIZON + 1),
                "forecast": forecast,
                "actual": actual,
                "lower_p10": lower,
                "upper_p90": upper,
                "capacity_upper_p90": capacity_trigger(forecast, pooled),
                "capacity_upper_p90_normal": capacity_trigger(
                    forecast, pooled, method="normal"
                ),
            }
        )
        step_diagnostics = residual_intervals(folds, HORIZON)

        forecasts[target] = forecast
        results[target] = TargetResult(
            target=target,
            selected_model=best_name,
            leaderboard=board,
            forecast=forecast,
            holdout_actual=actual,
            holdout_score=score,
            intervals=intervals,
            folds=folds,
        )
        results[target].step_diagnostics = step_diagnostics

    predictions = pd.DataFrame(
        {
            "timestamp": holdout["timestamp"].to_numpy(),
            "active_vm_count": forecasts["active_vm_count"].astype(int),
            "aggregate_cpu_utilization": np.round(
                forecasts["aggregate_cpu_utilization"], 6
            ),
        }
    )
    return PipelineResult(predictions=predictions, target_results=results, audit=audit)
