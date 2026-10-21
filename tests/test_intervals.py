import numpy as np

from cwfc.evaluation import (
    FoldResult,
    capacity_trigger,
    empirical_interval,
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
    forecast = np.linspace(1, 2, 24)
    trigger = capacity_trigger(forecast, pooled_cv_residuals(folds))
    assert (trigger >= forecast).all()
    # constant sigma -> constant offset across steps
    offsets = trigger - forecast
    assert np.allclose(offsets, offsets[0])
