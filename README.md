# Cloud Workload Forecasting Challenge — Solution

End-to-end, leakage-safe forecasting solution for the curated **Microsoft Azure
Public Dataset V2** VM trace (10,000 VMs × 134 five-minute CPU observations).
Forked from [`atomityhq/forecasting-challenge`](https://github.com/atomityhq/forecasting-challenge).

- Full write-up: **[`reports/final_report.md`](reports/final_report.md)**
- Decision journal (rationale, dead ends, selection robustness): **[`JOURNEY.md`](JOURNEY.md)**

## Headline result

| Target | Selected model | Holdout MAE | Holdout RMSE | vs. persistence |
|---|---|---:|---:|---:|
| `aggregate_cpu_utilization` | recursive Ridge AR, lags 1–2 (α=1) | 0.124 pp | 0.149 pp | **−15.5 % MAE** |
| `active_vm_count` | persistence (constant 10,000) | 0.000 | 0.000 | exact |

Model selection used only expanding-window rolling-origin CV (10 folds, 240
out-of-sample residuals) inside the first 110 observations; the final 24
observations were touched once for reporting. A 21-candidate pre-study
(`analysis/model_experiments.py`) found nothing better than the regularized
linear autoregression.

## Repository layout

```text
src/cwfc/
  config.py        # challenge contract constants (grid, 110/24 split, targets)
  data.py          # loading, panel audit, target construction, chronological split
  features.py      # leakage-safe lag features (past values only)
  models.py        # 8 forecasters: persistence, MA, seasonal-naive, SES, drift,
                   #                Ridge AR, damped ETS, ARIMA
  validation.py    # rolling-origin index/timestamp splitters (no shuffling)
  evaluation.py    # CV, holdout scoring, empirical intervals, capacity trigger
  pipeline.py      # end-to-end orchestration (selection -> refit -> forecast)
  predict.py       # CLI producing predictions.csv
analysis/
  run_analysis.py      # regenerates every EDA table and figure in reports/
  model_experiments.py # training-only 21-candidate pre-study (evidence for choices)
tests/             # 29 tests: targets, leakage, models, intervals, determinism,
                   #         smoke-dataset end-to-end CLI + full analysis run
reports/
  final_report.md  # 4–6 page report (9 required sections)
  tables/          # CV leaderboards, metrics, interval diagnostics (CSV/JSON)
  figures/         # 8 figures referenced by the report
JOURNEY.md         # narrative of the reasoning, iterations and failed approaches
Makefile           # install / test / predict / analysis / experiments targets
predictions.csv    # required submission artifact (24 rows, exact contract schema)
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"          # or: pip install -r requirements.txt
```

## Dataset

Download `data-v1.tar.gz` from the
[data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)
and extract it so the data lives under `challenge_data/` (never committed):

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
cd challenge_data && sha256sum -c SHA256SUMS
```

## Reproduce the submission

One command, from a clean checkout, produces `predictions.csv`:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv \
    --artifacts-dir reports/tables
# equivalent: make predict
```

Output schema (exactly 24 rows, chronological, no index column):

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

Other workflows:

```bash
make test          # 29 tests, incl. synthetic smoke-dataset end-to-end run
make analysis      # all EDA tables and 8 report figures
make experiments   # rerun the 21-candidate training-only pre-study
make all           # test + predict + analysis
```

Run the full suite directly with `python -m pytest -q`. The smoke-dataset
tests verify the pipeline is not hard-coded to the supplied files and that
repeated runs are byte-for-byte deterministic.

## Main modeling decisions

1. **Targets first, split second.** Aggregate targets are built from the full
   panel; the chronological 110/24 split is applied afterwards, making leakage
   structurally impossible.
2. **Native five-minute cadence retained** — only 134 observations exist, so
   coarsening would throw away signal.
3. **Lag-only features (1, 2).** Autocorrelation is short-memory (lag-1
   ACF 0.66, ≈ 0 by lag 24); trend/calendar terms have no defensible
   interpretation with relative timestamps and an 11.1-hour window, and CV
   showed them destabilizing recursive forecasts.
4. **Ridge with `alpha=1`** chosen by the training-only pre-study over lag
   sets `(1)`–`(1,2,3,6)`, four strengths, and seven other model families.
5. **Eight-model leaderboard under identical 10-fold rolling-origin CV**, with
   persistence as the required baseline; sophisticated ETS/ARIMA/boosting
   models lose honestly rather than being excluded.
6. **Uncertainty from 240 out-of-sample CV residuals:** constant-width 80 %
   empirical intervals and a one-sided p90 capacity trigger (forecast +
   ~0.30 pp). Per-step quantiles were tried and rejected as unstable with few
   folds (documented in `JOURNEY.md`).
7. **No holdout target values** are used in any fit, feature, tuning, or
   selection step; tests assert feature rows cannot see their own target, and
   selection-robustness is reported rather than hidden.

## Dependencies

numpy, pandas, scikit-learn, statsmodels (runtime); pytest, matplotlib
(analysis/dev). See `pyproject.toml` / `requirements.txt`.
