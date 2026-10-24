# Copyright (c) 2026 Biswaketan <https://biswaketan.in>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: do not remove (PolyForm Notices/Violations) - build CWFC-NC1-7Q2M9X4D.

"""Leakage-safe lag features for autoregressive forecasting.

Every feature for target position ``t`` uses only values observed before ``t``.
The source timestamps are *relative seconds* with no wall-clock calendar and
only ~11 hours of history, so calendar/seasonal features have no defensible
interpretation; rolling-origin CV also showed deterministic trend features
destabilizing recursive multi-step forecasts. The production feature set is
therefore lag-only, which is fully known at the forecast origin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAGS = (1, 2)  # chosen by the training-only pre-study (analysis/model_experiments.py)


def lag_feature_table(
    history: np.ndarray, lags: tuple[int, ...] = DEFAULT_LAGS
) -> pd.DataFrame:
    """Supervised lag table built strictly from an observed history array."""
    history = np.asarray(history, dtype=float)
    max_lag = max(lags)
    data = {
        f"lag_{lag}": history[np.arange(max_lag, len(history)) - lag]
        for lag in lags
    }
    return pd.DataFrame(data)


def build_ar_design(
    history: np.ndarray, lags: tuple[int, ...] = DEFAULT_LAGS
):
    """Return ``(X, y)`` where row ``i`` predicts one step after its lag window."""
    history = np.asarray(history, dtype=float)
    table = lag_feature_table(history, lags)
    y = history[max(lags):]
    return table.to_numpy(dtype=float), y


def recursive_feature_vector(
    known: list[float],
    forecast_position: int,
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> np.ndarray:
    """Lag vector for predicting the value at ``forecast_position``.

    ``known`` holds observed values followed by already-produced forecasts, so
    recursive multi-step prediction never reaches for unobserved data.
    """
    return np.asarray(
        [float(known[forecast_position - lag]) for lag in lags], dtype=float
    )
