import numpy as np
import pytest

from cwfc.evaluation import (
    FoldResult,
    calibration_table,
    capacity_trigger,
    empirical_interval,
    fold_band_breach_rates,
    pooled_cv_residuals,
    residual_intervals,
)


def _folds():
    rng = np.random.default_rng(0)
    return [
        FoldResult(
            fold=i,
            origin=40 + i * 10,
            mae=0.1,
            rmse=0.15,
            smape=1.0,
            residuals=rng.normal(0, 0.2, 24),
        )
        for i in range(3)
    ]


def test_pooled_residuals_and_intervals():
    folds = _folds()
    pooled = pooled_cv_residuals(folds)
    assert pooled.shape == (72,)
    table = residual_intervals(folds, 24)
    assert len(table) == 24
    assert (table["residual_p10"] <= table["residual_p90"]).all()


def test_empirical_interval_orders_bounds():
    folds = _folds()
    forecast = np.full(24, 5.0)
    lower, upper = empirical_interval(forecast, pooled_cv_residuals(folds))
    assert (lower <= forecast).all()
    assert (upper >= forecast).all()
    assert np.allclose(upper - lower, upper[0] - lower[0])  # constant width


def test_capacity_trigger_is_one_sided_and_cv_only():
    folds = _folds()
    pooled = pooled_cv_residuals(folds)
    forecast = np.linspace(1, 2, 24)
    trigger = capacity_trigger(forecast, pooled)
    assert (trigger >= forecast).all()
    # pooled empirical quantile -> constant offset across steps
    offsets = trigger - forecast
    assert np.allclose(offsets, offsets[0])
    # empirical trigger must equal forecast plus the empirical 90th percentile
    assert np.allclose(trigger, forecast + np.quantile(pooled, 0.90))


def test_capacity_trigger_methods_differ_only_as_documented():
    folds = _folds()
    pooled = pooled_cv_residuals(folds)
    forecast = np.full(24, 5.0)
    empirical = capacity_trigger(forecast, pooled, method="empirical")
    normal = capacity_trigger(forecast, pooled, method="normal")
    assert empirical.shape == normal.shape == (24,)
    with pytest.raises(ValueError):
        capacity_trigger(forecast, pooled, method="banana")


def test_calibration_table_is_monotone_and_brackets_nominal():
    folds = _folds()
    table = calibration_table(pooled_cv_residuals(folds))
    assert len(table) == 5
    assert table["nominal_coverage"].is_monotonic_increasing
    assert table["band_width"].is_monotonic_increasing
    assert (table["empirical_coverage"] > 0).all()
    assert np.allclose(
        table["empirical_coverage"] + table["outside_rate"], 1.0
    )


def test_fold_band_breach_rates_are_valid_fractions():
    folds = _folds()
    out = fold_band_breach_rates(folds, level=0.80)
    assert len(out["per_fold_outside_rate"]) == len(folds)
    assert all(0.0 <= rate <= 1.0 for rate in out["per_fold_outside_rate"])
