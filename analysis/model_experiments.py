"""Leakage-safe model pre-study on the 110-point training window only.

This script is the evidence base for the modeling decisions documented in
JOURNEY.md. It never touches the final 24 holdout observations: every
candidate is scored exclusively with expanding-window rolling-origin CV on
the training series. Run:

    python analysis/model_experiments.py --data challenge_data
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from cwfc.config import HORIZON
from cwfc.data import build_target_frame, load_readings, split_train_holdout
from cwfc.models import build_model
from cwfc.validation import rolling_origin_indices

warnings.filterwarnings("ignore")


def _design(h, lags, diff=False):
    L = max(lags)
    X, y = [], []
    base = np.diff(h) if diff else h  # differenced series has length n-1
    for t in range(L, len(base)):
        X.append([base[t - lag] for lag in lags])
        y.append(base[t])
    return np.array(X), np.array(y)


class RecLinear:
    """Recursive multi-step linear autoregression on lagged values."""

    def __init__(self, lags=(1, 2, 3), alpha=10.0, kind="ridge", diff=False):
        self.lags, self.alpha, self.kind, self.diff = lags, alpha, kind, diff

    def fit(self, h):
        self.h = np.asarray(h, dtype=float)
        est = (
            Ridge(alpha=self.alpha)
            if self.kind == "ridge"
            else ElasticNet(alpha=self.alpha, l1_ratio=0.5, max_iter=2000)
        )
        X, yv = _design(h, self.lags, diff=self.diff)
        self.model = make_pipeline(StandardScaler(), est).fit(X, yv)
        return self

    def forecast(self, steps):
        known = list(self.h)
        out = []
        for _ in range(steps):
            if self.diff:
                d = np.diff(np.asarray(known))
                v = np.array([d[len(d) - lag] for lag in self.lags]).reshape(1, -1)
                pred = known[-1] + float(self.model.predict(v)[0])
            else:
                v = np.array([known[len(known) - lag] for lag in self.lags]).reshape(1, -1)
                pred = float(self.model.predict(v)[0])
            out.append(pred)
            known.append(pred)
        return np.array(out)


class HGBR:
    def __init__(self, lags=(1, 2, 3)):
        self.lags = lags

    def fit(self, h):
        self.h = list(np.asarray(h, dtype=float))
        X, yv = _design(h, self.lags)
        self.model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=3, learning_rate=0.05, random_state=0
        ).fit(X, yv)
        return self

    def forecast(self, steps):
        known = list(self.h)
        out = []
        for _ in range(steps):
            v = np.array([known[len(known) - lag] for lag in self.lags]).reshape(1, -1)
            out.append(float(self.model.predict(v)[0]))
            known.append(out[-1])
        return np.array(out)


class SES:
    def fit(self, h):
        self.model = SimpleExpSmoothing(
            np.asarray(h, float), initialization_method="estimated"
        ).fit()
        return self

    def forecast(self, steps):
        return np.asarray(self.model.forecast(steps))


class Drift:
    def fit(self, h):
        h = np.asarray(h, float)
        self.last = h[-1]
        self.slope = (h[-1] - h[0]) / (len(h) - 1)
        return self

    def forecast(self, steps):
        return self.last + self.slope * np.arange(1, steps + 1)


class Ensemble:
    def __init__(self, factories, weights=None):
        self.factories, self.weights = factories, weights

    def fit(self, h):
        self.models = [f().fit(h) for f in self.factories]
        return self

    def forecast(self, steps):
        preds = np.array([m.forecast(steps) for m in self.models])
        return np.average(preds, axis=0, weights=self.weights)


def _ridge(lags, alpha=10.0):
    return lambda: RecLinear(lags, alpha)


def candidates():
    return {
        "persistence": lambda: build_model("persistence"),
        "moving_average_6": lambda: build_model("moving_average_6"),
        "seasonal_naive_12": lambda: build_model("seasonal_naive_12"),
        "ets_damped": lambda: build_model("ets_damped"),
        "arima_101": lambda: build_model("arima_101"),
        "ses": SES,
        "drift": Drift,
        "ridge_l1_a1": _ridge((1,), 1.0),
        "ridge_l1_a10": _ridge((1,), 10.0),
        "ridge_l12_a1": _ridge((1, 2), 1.0),
        "ridge_l12_a10": _ridge((1, 2), 10.0),
        "ridge_l123_a1": _ridge((1, 2, 3), 1.0),
        "ridge_l123_a10": _ridge((1, 2, 3), 10.0),
        "ridge_l1236_a10": _ridge((1, 2, 3, 6), 10.0),
        "ridge_l1236_a100": _ridge((1, 2, 3, 6), 100.0),
        "ridge_diff_l123": lambda: RecLinear((1, 2, 3), 10.0, diff=True),
        "elasticnet_l123": lambda: RecLinear((1, 2, 3), 0.01, kind="enet"),
        "hgbr_l123": lambda: HGBR((1, 2, 3)),
        "ens_ridge_persistence": lambda: Ensemble(
            [_ridge((1, 2, 3)), lambda: build_model("persistence")]
        ),
        "ens_ridge_ets": lambda: Ensemble(
            [_ridge((1, 2, 3)), lambda: build_model("ets_damped")]
        ),
        "ens_ridge_persistence_ses": lambda: Ensemble(
            [_ridge((1, 2, 3)), lambda: build_model("persistence"), SES]
        ),
    }


def run_cv(series, cands, min_train, step, horizon=HORIZON):
    rows = {}
    for name, factory in cands.items():
        rmses, maes = [], []
        for sp in rolling_origin_indices(len(series), horizon, min_train, step):
            model = factory().fit(series[sp.train_start:sp.train_end])
            actual = series[sp.test_start:sp.test_end]
            pred = model.forecast(horizon)
            rmses.append(np.sqrt(np.mean((actual - pred) ** 2)))
            maes.append(np.mean(np.abs(actual - pred)))
        rows[name] = (np.mean(rmses), np.mean(maes), np.std(rmses, ddof=1))
    return rows


def main(data_dir: str) -> None:
    readings = load_readings(data_dir)
    target = build_target_frame(readings)
    train, _ = split_train_holdout(target)
    series = train["aggregate_cpu_utilization"].to_numpy()
    cands = candidates()
    schemes = [
        ("dense 10-fold (min_train=48, step=4)", 48, 4),
        ("expanded 7-fold (min_train=48, step=6)", 48, 6),
        ("original 3-fold (min_train=62, step=12)", 62, 12),
    ]
    for label, min_train, step in schemes:
        print(f"\n== {label} ==")
        rows = run_cv(series, cands, min_train, step)
        for name, (rmse, mae_v, sd) in sorted(rows.items(), key=lambda kv: kv[1][0]):
            print(f"{name:28s} RMSE={rmse:.4f} MAE={mae_v:.4f} fold_sd={sd:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="challenge_data")
    args = parser.parse_args()
    main(args.data)
