"""Time-aware validation utilities.

No random shuffling anywhere: every split is an expanding-window,
rolling-origin design that mirrors production (forecast the next ``horizon``
steps using only data available at the origin).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    ts = pd.to_datetime(frame[timestamp_col], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out


@dataclass(frozen=True)
class IndexSplit:
    """Integer-position expanding-window split."""

    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive

    @property
    def origin(self) -> int:
        return self.test_start


def rolling_origin_indices(
    n: int,
    horizon: int,
    min_train_size: int,
    step: int | None = None,
) -> list[IndexSplit]:
    """Expanding-window splits on positional indices.

    Produces every origin ``o`` with ``o >= min_train_size`` and
    ``o + horizon <= n``, spaced ``step`` observations apart.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    if min_train_size + horizon > n:
        raise ValueError(
            f"not enough observations ({n}) for min_train={min_train_size} "
            f"and horizon={horizon}"
        )
    step = horizon if step is None else step
    splits: list[IndexSplit] = []
    origin = min_train_size
    while origin + horizon <= n:
        splits.append(IndexSplit(0, origin, origin, origin + horizon))
        origin += step
    return splits
