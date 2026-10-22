"""Leakage-safe model evaluation: rolling-origin CV and holdout scoring."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import mae, rmse, smape
from .models import BaseForecaster, build_model
from .validation import rolling_origin_indices


@dataclass
class FoldResult:
    fold: int
    origin: int
    mae: float
    rmse: float
    smape: float
    residuals: np.ndarray  # actual - forecast, length == horizon


def cross_validate(
    values: np.ndarray,
    model_name: str,
    horizon: int,
    min_train_size: int,
    step: int | None = None,
) -> list[FoldResult]:
    """Expanding-window CV using only the series passed in.

    Callers pass the *training* series (110 points); the holdout never enters
    model selection.
    """
    values = np.asarray(values, dtype=float)
    folds: list[FoldResult] = []
    for i, split in enumerate(
        rolling_origin_indices(len(values), horizon, min_train_size, step)
    ):
        history = values[split.train_start:split.train_end]
        actual = values[split.test_start:split.test_end]
        forecaster: BaseForecaster = build_model(model_name).fit(history)
        pred = forecaster.forecast(horizon)
        folds.append(
            FoldResult(
                fold=i,
                origin=split.origin,
                mae=mae(actual, pred),
                rmse=rmse(actual, pred),
                smape=smape(actual, pred),
                residuals=actual - pred,
            )
        )
    return folds


def summarize_folds(folds: list[FoldResult]) -> dict:
    return {
        "n_folds": len(folds),
        "mae_mean": float(np.mean([f.mae for f in folds])),
        "rmse_mean": float(np.mean([f.rmse for f in folds])),
        "smape_mean": float(np.mean([f.smape for f in folds])),
    }


def select_best_model(
    values: np.ndarray,
    candidate_names: list[str],
    horizon: int,
    min_train_size: int,
    step: int | None = None,
    metric: str = "rmse_mean",
) -> tuple[str, pd.DataFrame]:
    """Return the best model name and the full CV leaderboard (train data only)."""
    rows = []
    for name in candidate_names:
        folds = cross_validate(values, name, horizon, min_train_size, step)
        row = {"model": name, **summarize_folds(folds)}
        rows.append(row)
    board = pd.DataFrame(rows).sort_values(metric).reset_index(drop=True)
    return str(board.iloc[0]["model"]), board


def score_holdout(actual: np.ndarray, pred: np.ndarray, baseline_pred: np.ndarray) -> dict:
    """Final, one-time holdout metrics including relative and peak analysis."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    baseline_pred = np.asarray(baseline_pred, dtype=float)
    model_mae, base_mae = mae(actual, pred), mae(actual, baseline_pred)
    # Peak periods: top tercile of observed demand in the holdout. For a
    # constant series (e.g. active_vm_count in this panel) every period is
    # identical, so peak and off-peak metrics both equal the overall metric.
    if np.isclose(np.ptp(actual), 0.0):
        peak_mae = offpeak_mae = model_mae
    else:
        peak_threshold = np.quantile(actual, 2.0 / 3.0)
        peak_mask = actual >= peak_threshold
        peak_mae = mae(actual[peak_mask], pred[peak_mask])
        offpeak_mae = mae(actual[~peak_mask], pred[~peak_mask])
    out = {
        "mae": model_mae,
        "rmse": rmse(actual, pred),
        "smape": smape(actual, pred),
        "baseline_mae": base_mae,
        "baseline_rmse": rmse(actual, baseline_pred),
        # Relative MAE; <1 means the model beats persistence. Guarded against
        # a zero baseline denominator (e.g. constant active_vm_count).
        "mae_relative_to_persistence": (
            model_mae / base_mae if base_mae > 0 else (0.0 if model_mae == 0 else np.inf)
        ),
        "peak_mae": peak_mae,
        "offpeak_mae": offpeak_mae,
        "max_abs_error": float(np.max(np.abs(actual - pred))),
        "mean_error_bias": float(np.mean(pred - actual)),
    }
    return out


def residual_intervals(folds: list[FoldResult], horizon: int) -> pd.DataFrame:
    """Empirical residual statistics by horizon step (diagnostic table).

    With only a few CV folds, per-step quantiles are noisy; the pooled band
    helpers below are therefore used for the reported prediction intervals.
    Folds end at or before the training boundary, so no holdout information
    enters any of these numbers.
    """
    matrix = np.vstack([f.residuals for f in folds])  # (n_folds, horizon)
    rows = []
    for step in range(horizon):
        r = matrix[:, step]
        rows.append(
            {
                "step": step + 1,
                "residual_p10": float(np.quantile(r, 0.10)),
                "residual_p50": float(np.quantile(r, 0.50)),
                "residual_p90": float(np.quantile(r, 0.90)),
                "residual_std": float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def pooled_cv_residuals(folds: list[FoldResult]) -> np.ndarray:
    """All out-of-sample CV residuals pooled across folds and horizon steps."""
    return np.concatenate([f.residuals for f in folds])


def empirical_interval(
    forecast: np.ndarray, pooled_residuals: np.ndarray, alpha: float = 0.20
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-width 1-alpha empirical interval from pooled CV residuals.

    Constant width (rather than per-step quantiles) is deliberate: with three
    folds the per-step quantiles are unstable and can invert.
    """
    lower_q, upper_q = np.quantile(pooled_residuals, [alpha / 2, 1 - alpha / 2])
    forecast = np.asarray(forecast, dtype=float)
    return forecast + lower_q, forecast + upper_q


def capacity_trigger(
    forecast: np.ndarray,
    pooled_residuals: np.ndarray,
    method: str = "empirical",
    alpha: float = 0.90,
) -> np.ndarray:
    """One-sided upper safety line for capacity provisioning.

    Residuals are defined as actual minus forecast, so the upper tail
    quantile is the headroom that covers ``alpha`` of historical out-of-sample
    surprises.

    ``method="empirical"`` (default, consistent with the pooled-empirical
    interval philosophy) adds the empirical alpha-quantile of CV residuals and
    makes no distributional assumption. ``method="normal"`` adds
    ``z_alpha * sigma`` and is retained only as a comparison; it under-covers
    when the residual distribution is skewed.
    """
    forecast = np.asarray(forecast, dtype=float)
    pooled_residuals = np.asarray(pooled_residuals, dtype=float)
    if method == "empirical":
        offset = float(np.quantile(pooled_residuals, alpha))
    elif method == "normal":
        from scipy.stats import norm

        sigma = float(np.std(pooled_residuals, ddof=1))
        offset = float(norm.ppf(alpha) * sigma)
    else:
        raise ValueError("method must be 'empirical' or 'normal'")
    return forecast + offset


def calibration_table(
    pooled_residuals: np.ndarray,
    nominal_levels: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90),
) -> pd.DataFrame:
    """Nominal vs empirical coverage of pooled-residual quantile bands.

    The bands and the residuals come from the same pool of out-of-sample CV
    residuals, so the empirical column is an in-sample-of-the-CV calibration
    check; the true out-of-sample check is the band's coverage on the holdout,
    computed separately in the analysis script.
    """
    pooled_residuals = np.asarray(pooled_residuals, dtype=float)
    rows = []
    for level in nominal_levels:
        lower_q, upper_q = np.quantile(
            pooled_residuals, [(1 - level) / 2, 1 - (1 - level) / 2]
        )
        inside = (pooled_residuals >= lower_q) & (pooled_residuals <= upper_q)
        rows.append(
            {
                "nominal_coverage": level,
                "empirical_coverage": float(inside.mean()),
                "outside_rate": float(1.0 - inside.mean()),
                "lower_quantile": float(lower_q),
                "upper_quantile": float(upper_q),
                "band_width": float(upper_q - lower_q),
            }
        )
    return pd.DataFrame(rows)


def fold_band_breach_rates(
    folds: list[FoldResult], level: float = 0.80
) -> dict:
    """Per-fold fraction of residuals outside the pooled band at ``level``.

    Reveals whether a band's aggregate coverage is uniform across origins or
    propped up by a few easy folds.
    """
    pooled = pooled_cv_residuals(folds)
    lower_q, upper_q = np.quantile(pooled, [(1 - level) / 2, 1 - (1 - level) / 2])
    rates = [
        float(np.mean((f.residuals < lower_q) | (f.residuals > upper_q)))
        for f in folds
    ]
    return {
        "level": level,
        "mean_outside_rate": float(np.mean(rates)),
        "sd_outside_rate": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
        "per_fold_outside_rate": rates,
    }
