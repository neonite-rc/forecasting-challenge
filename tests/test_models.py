import numpy as np
import pytest

from cwfc.models import available_models, build_model


@pytest.mark.parametrize("name", available_models())
def test_all_models_produce_finite_forecasts(name):
    rng = np.random.default_rng(7)
    history = 10 + np.cumsum(rng.normal(0, 0.1, 80))
    model = build_model(name).fit(history)
    pred = model.forecast(24)
    assert pred.shape == (24,)
    assert np.isfinite(pred).all()


def test_constant_series_forecasts_constant():
    history = np.full(50, 10000.0)
    for name in available_models():
        pred = build_model(name).fit(history).forecast(24)
        assert np.allclose(pred, 10000.0)


def test_persistence_matches_last_value():
    history = np.arange(10, dtype=float)
    pred = build_model("persistence").fit(history).forecast(5)
    assert np.allclose(pred, 9.0)


def test_seasonal_naive_repeats_last_cycle():
    history = np.arange(24, dtype=float)
    pred = build_model("seasonal_naive_12").fit(history).forecast(3)
    assert np.allclose(pred, [12.0, 13.0, 14.0])


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_model("does_not_exist")
