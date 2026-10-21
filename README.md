# Cloud Workload Forecasting Challenge — Solution

End-to-end, leakage-safe forecasting solution for the curated **Microsoft Azure
Public Dataset V2** VM trace (10,000 VMs × 134 five-minute CPU observations).
Forked from [`atomityhq/forecasting-challenge`](https://github.com/atomityhq/forecasting-challenge).

The full write-up is in **[`reports/final_report.md`](reports/final_report.md)**.

## Headline result

| Target | Selected model | Holdout MAE | Holdout RMSE | vs. persistence |
|---|---|---:|---:|---:|
| `aggregate_cpu_utilization` | recursive Ridge AR (lags 1–3) | 0.101 pp | 0.122 pp | **−31.4 % MAE** |
| `active_vm_count` | persistence (constant 10,000) | 0.000 | 0.000 | exact |

Model selection used only expanding-window rolling-origin CV inside the first
110 observations; the final 24 observations were touched once for reporting.

## Repository layout

```text
src/cwfc/
  config.py        # challenge contract constants (grid, 110/24 split, targets)
  data.py          # loading, panel audit, target construction, chronological split
  features.py      # leakage-safe lag features (past values only)
  models.py        # persistence, moving-average, seasonal-naive, ETS, ARIMA, Ridge AR
  validation.py    # rolling-origin index/timestamp splitters (no shuffling)
  evaluation.py    # CV, holdout scoring, empirical intervals, capacity trigger
  pipeline.py      # end-to-end orchestration (selection -> refit -> forecast)
  predict.py       # CLI producing predictions.csv
analysis/
  run_analysis.py  # regenerates every EDA table and figure in reports/
tests/             # 24 tests: targets, leakage, models, intervals, smoke-dataset CLI
reports/
  final_report.md  # 4–6 page report (9 required sections)
  tables/          # CV leaderboards, metrics, interval diagnostics (CSV/JSON)
  figures/         # 8 figures referenced by the report
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
```

Output schema (exactly 24 rows, chronological, no index column):

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

Regenerate all analysis tables and figures:

```bash
python analysis/run_analysis.py --data challenge_data
```

Run the tests (includes an end-to-end run on a tiny synthetic release-shaped
smoke dataset, verifying the pipeline is not hard-coded to the supplied files):

```bash
python -m pytest -q
```

## Main modeling decisions

1. **Targets first, split second.** Aggregate targets are built from the full
   panel; the chronological 110/24 split is applied afterwards, making leakage
   structurally impossible.
2. **Native five-minute cadence retained** — only 134 observations exist, so
   coarsening would throw away signal.
3. **Lag-only features (1, 2, 3).** Autocorrelation is short-memory (lag-1
   ACF 0.66, ≈ 0 by lag 24); trend/calendar terms have no defensible
   interpretation with relative timestamps and an 11.1-hour window, and CV
   showed them destabilizing recursive forecasts.
4. **Ridge with `alpha=10`** chosen by training-only CV over lag sets and
   regularization strengths; shrinkage controls recursive error accumulation.
5. **Six-model leaderboard under identical rolling-origin CV**, with
   persistence as the required baseline; sophisticated ETS/ARIMA models lose
   honestly rather than being excluded.
6. **Uncertainty from 72 out-of-sample CV residuals:** constant-width 80 %
   empirical intervals and a one-sided p90 capacity trigger (forecast +
   0.18 pp), breached in only 1/24 holdout periods.
7. **No holdout target values** are used in any fit, feature, tuning, or
   selection step; tests assert feature rows cannot see their own target.

## Dependencies

numpy, pandas, scikit-learn, statsmodels (runtime); pytest, matplotlib
(analysis/dev). See `pyproject.toml` / `requirements.txt`.
