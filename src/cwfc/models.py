"""Forecaster zoo with a uniform fit/forecast interface.

All forecasters ingest a one-dimensional history (the target series up to the
forecast origin) and produce a multi-step forecast. Multi-step forecasts are
genuinely out-of-sample: recursive models feed their own one-step predictions
back in as lags, and no forecaster ever sees holdout target values.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import DEFAULT_LAGS, build_ar_design, recursive_feature_vector

try:  # statsmodels is an explicit project dependency, but keep imports defensive
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:  # pragma: no cover - exercised only in broken environments
    ExponentialSmoothing = None
    SARIMAX = None


def _fit_ets(history: np.ndarray):
    if ExponentialSmoothing is None:
        raise RuntimeError("statsmodels is required for ETS forecasting")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExponentialSmoothing(
            history, trend="add", damped_trend=True, initialization_method="estimated"
        ).fit()


class BaseForecaster(ABC):
    name = "base"

    def fit(self, history: np.ndarray) -> "BaseForecaster":
        self.history_ = np.asarray(history, dtype=float).copy()
        self.constant_ = bool(np.isclose(np.ptp(self.history_), 0.0))
        if not self.constant_:
            self._fit(self.history_)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not hasattr(self, "history_"):
            raise RuntimeError("forecaster must be fit before forecasting")
        if self.constant_:
            return np.full(steps, self.history_[-1], dtype=float)
        return self._forecast(steps)

    @abstractmethod
    def _fit(self, history: np.ndarray) -> None: ...

    @abstractmethod
    def _forecast(self, steps: int) -> np.ndarray: ...


class PersistenceForecaster(BaseForecaster):
    name = "persistence"

    def _fit(self, history: np.ndarray) -> None:
        self.last_ = history[-1]

    def _forecast(self, steps: int) -> np.ndarray:
        return np.full(steps, self.last_, dtype=float)


class MovingAverageForecaster(BaseForecaster):
    name = "moving_average_6"

    def __init__(self, window: int = 6) -> None:
        self.window = window
        self.name = f"moving_average_{window}"

    def _fit(self, history: np.ndarray) -> None:
        self.last_ = history[-self.window:].mean()

    def _forecast(self, steps: int) -> np.ndarray:
        return np.full(steps, self.last_, dtype=float)


class SeasonalNaiveForecaster(BaseForecaster):
    name = "seasonal_naive_12"

    def __init__(self, season_length: int = 12) -> None:
        self.season_length = season_length
        self.name = f"seasonal_naive_{season_length}"

    def _fit(self, history: np.ndarray) -> None:
        # On short smoke histories, use the longest available cycle.
        self.effective_season_ = min(self.season_length, len(history))
        self.cycle_ = history[-self.effective_season_:].copy()

    def _forecast(self, steps: int) -> np.ndarray:
        return np.asarray(
            [self.cycle_[i % self.effective_season_] for i in range(steps)],
            dtype=float,
        )


class RidgeARForecaster(BaseForecaster):
    """Recursive autoregressive Ridge regression on lagged targets.

    Lag set and regularization are fixed from rolling-origin CV conducted on
    training data only. CV showed lag-only models materially outperform the
    same model with deterministic trend/cyclical features, which drift when
    predictions are fed back recursively over a 24-step horizon.
    """

    name = "ridge_ar"

    def __init__(self, lags: tuple[int, ...] = DEFAULT_LAGS, alpha: float = 10.0) -> None:
        self.lags = tuple(lags)
        self.alpha = alpha

    def _fit(self, history: np.ndarray) -> None:
        # Shrink the lag set on very short histories (smoke datasets) so the
        # design matrix always has at least two rows.
        effective_lags = tuple(lag for lag in self.lags if lag <= len(history) - 3)
        if not effective_lags:
            effective_lags = (1,) if len(history) >= 4 else ()
        self.effective_lags_ = effective_lags
        if effective_lags:
            X, y = build_ar_design(history, effective_lags)
            self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
            self.model_.fit(X, y)
        else:
            # Extremely short history: degrade to the global mean.
            self.model_ = None
            self.fallback_mean_ = float(history.mean())

    def _forecast(self, steps: int) -> np.ndarray:
        if self.model_ is None:
            return np.full(steps, self.fallback_mean_, dtype=float)
        known = list(self.history_)
        out: list[float] = []
        for _ in range(steps):
            pos = len(known)
            x = recursive_feature_vector(known, pos, self.effective_lags_).reshape(1, -1)
            yhat = float(self.model_.predict(x)[0])
            out.append(yhat)
            known.append(yhat)
        return np.asarray(out, dtype=float)


class ETSDampedForecaster(BaseForecaster):
    """Holt's linear trend with damped trend, no seasonal component."""

    name = "ets_damped"

    def _fit(self, history: np.ndarray) -> None:
        self.model_ = _fit_ets(history)

    def _forecast(self, steps: int) -> np.ndarray:
        return np.asarray(self.model_.forecast(steps), dtype=float)


class ARIMAForecaster(BaseForecaster):
    """ARMA(1,1) with linear trend; falls back to damped ETS if MLE fails."""

    name = "arima_101"

    def _fit(self, history: np.ndarray) -> None:
        if SARIMAX is None:
            raise RuntimeError("statsmodels is required for ARIMAForecaster")
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_ = SARIMAX(
                    history, order=(1, 0, 1), trend="t", enforce_stationarity=True
                ).fit(disp=False, maxiter=200)
            self._fallback = False
        except Exception:
            self.model_ = _fit_ets(history)
            self._fallback = True

    def _forecast(self, steps: int) -> np.ndarray:
        return np.asarray(self.model_.forecast(steps), dtype=float)


_FORECASTER_REGISTRY: dict[str, type[BaseForecaster]] = {
    "persistence": PersistenceForecaster,
    "moving_average_6": MovingAverageForecaster,
    "seasonal_naive_12": SeasonalNaiveForecaster,
    "ridge_ar": RidgeARForecaster,
    "ets_damped": ETSDampedForecaster,
    "arima_101": ARIMAForecaster,
}


def available_models() -> list[str]:
    return list(_FORECASTER_REGISTRY)


def build_model(name: str) -> BaseForecaster:
    try:
        return _FORECASTER_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(
            f"unknown model '{name}'; choose from {available_models()}"
        ) from exc
