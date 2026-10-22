"""Leakage-safe model pre-study on the 110-point training window only.

This script is the evidence base for the modeling decisions documented in
JOURNEY.md. It never touches the final 24 holdout observations: every
candidate is scored exclusively with expanding-window rolling-origin CV on
the training series. Run:

    python analysis/model_experiments.py --data challenge_data
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Allow direct execution (`python analysis/model_experiments.py`) pre-install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
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


class RecLinearBiasCorrected:
    """Ridge AR with an intercept correction (refinement candidate).

    The forecast is shifted by an estimate of recent one-step bias. Two
    variants are supported, both computable at the forecast origin:

    * ``source="insample"``: mean in-sample residual over the last ``window``
      fitted observations;
    * ``source="prior"``: mean out-of-sample residual of CV folds that ended
      at or before the current origin (supplied externally via ``prior_resid``).
    """

    def __init__(self, lags=(1, 2), alpha=1.0, source="insample", window=12,
                 prior_resid=None):
        self.lags, self.alpha = lags, alpha
        self.source, self.window = source, window
        self.prior_resid = prior_resid

    def fit(self, h):
        self.h = np.asarray(h, dtype=float)
        X, yv = _design(h, self.lags)
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(X, yv)
        if self.source == "insample":
            fitted = self.model.predict(X)
            resid = yv - fitted
            self.bias = float(np.mean(resid[-self.window:]))
        else:
            self.bias = float(np.mean(self.prior_resid)) if self.prior_resid else 0.0
        return self

    def forecast(self, steps):
        known = list(self.h)
        out = []
        for _ in range(steps):
            v = np.array([known[len(known) - lag] for lag in self.lags]).reshape(1, -1)
            out.append(float(self.model.predict(v)[0]) + self.bias)
            known.append(out[-1])
        return np.array(out)


class ARIMACSS:
    """ARMA(1,1) fitted by the innovations (CSS-like) estimator.

    statsmodels' statespace SARIMAX only offers Kalman MLE optimizers on short
    series; the higher-level ARIMA class exposes ``innovations_mle``, the
    conditional/innovations likelihood estimator (the modern equivalent of
    classical conditional sum of squares). Falls back to statespace MLE if the
    innovations estimator rejects the specification.
    """

    def fit(self, h):
        try:
            self.model = ARIMA(
                np.asarray(h, float), order=(1, 0, 1), trend="t"
            ).fit(method="innovations_mle")
        except Exception:
            self.model = ARIMA(
                np.asarray(h, float), order=(1, 0, 1), trend="t"
            ).fit(method="statespace")
        return self

    def forecast(self, steps):
        return np.asarray(self.model.forecast(steps))


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
        "arima_innovations_mle": ARIMACSS,
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


def _fold_rmses(series, factory, min_train, step, horizon=HORIZON):
    out = []
    for sp in rolling_origin_indices(len(series), horizon, min_train, step):
        model = factory().fit(series[sp.train_start:sp.train_end])
        actual = series[sp.test_start:sp.test_end]
        out.append(float(np.sqrt(np.mean((actual - model.forecast(horizon)) ** 2))))
    return out


def bias_correction_study(series, min_train=48, step=4, horizon=HORIZON):
    """Compare intercept-correction variants against the plain Ridge AR.

    All variants are evaluated inside the same rolling-origin loop, so the
    prior-fold correction only ever sees residuals from folds that ended
    before the current origin.
    """
    variants = {
        "ridge_no_correction": lambda prior: RecLinear((1, 2), 1.0),
        "bias_insample_last12": lambda prior: RecLinearBiasCorrected(
            (1, 2), 1.0, source="insample", window=12
        ),
        "bias_insample_all": lambda prior: RecLinearBiasCorrected(
            (1, 2), 1.0, source="insample", window=10**6
        ),
        "bias_prior_folds": lambda prior: RecLinearBiasCorrected(
            (1, 2), 1.0, source="prior", prior_resid=prior
        ),
    }
    rows = {name: [] for name in variants}
    prior_residuals: list[float] = []
    for sp in rolling_origin_indices(len(series), horizon, min_train, step):
        history = series[sp.train_start:sp.train_end]
        actual = series[sp.test_start:sp.test_end]
        for name, builder in variants.items():
            model = builder(prior_residuals).fit(history)
            pred = model.forecast(horizon)
            rows[name].append(float(np.sqrt(np.mean((actual - pred) ** 2))))
        # only after scoring do these residuals become available to later folds
        base = RecLinear((1, 2), 1.0).fit(history)
        prior_residuals.extend((actual - base.forecast(horizon)).tolist())
    return {name: (float(np.mean(v)), float(np.std(v, ddof=1))) for name, v in rows.items()}


def paired_selection_test(series, min_train=48, step=4, horizon=HORIZON):
    """Paired tests on per-fold RMSE of the two statistically-tied configs."""
    a = _fold_rmses(series, lambda: RecLinear((1, 2), 1.0), min_train, step, horizon)
    b = _fold_rmses(series, lambda: RecLinear((1, 2, 3), 10.0), min_train, step, horizon)
    a, b = np.array(a), np.array(b)
    if len(a) >= 2:
        t_p = float(stats.ttest_rel(a, b).pvalue)
        try:
            w_p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
        except ValueError:
            # all differences identical/zero (e.g. tiny smoke datasets)
            w_p = 1.0
    else:  # single fold: no paired test is identifiable
        t_p, w_p = float("nan"), float("nan")
    return {
        "folds": len(a),
        "rmse_lags12_alpha1": float(a.mean()),
        "rmse_lags123_alpha10": float(b.mean()),
        "mean_paired_diff": float((a - b).mean()),
        "folds_won_by_lags12": int((a < b).sum()),
        "folds_won_by_lags123": int((b < a).sum()),
        "paired_t_pvalue": t_p,
        "wilcoxon_pvalue": w_p,
    }


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

    print("\n== intercept-correction study (dense 10-fold) ==")
    for name, (rmse, sd) in bias_correction_study(series).items():
        print(f"{name:28s} RMSE={rmse:.4f} fold_sd={sd:.4f}")

    print("\n== paired test: lags(1,2) a=1 vs lags(1,2,3) a=10 (dense 10-fold) ==")
    for key, value in paired_selection_test(series).items():
        print(f"{key:28s} {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="challenge_data")
    args = parser.parse_args()
    main(args.data)
