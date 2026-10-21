# Cloud Workload Forecasting — Final Report

**Dataset:** Microsoft Azure Public Dataset V2, curated `data-v1` challenge release
**Series:** 134 five-minute observations (~11.1 hours) aggregated from 10,000 VMs / 1,340,000 CPU readings
**Holdout:** final 24 observations (2 hours), never used for fitting, feature selection, tuning, or model selection
**Selected model:** recursive autoregressive Ridge regression on lags 1–3 (`ridge_ar`), selected by rolling-origin CV on the training window

---

## 1. Executive summary

- The curated panel is **complete and regular**: 10,000 VMs × 134 timestamps on an exact 300-second grid, with zero missing cells, zero duplicate `(vm, timestamp)` rows, and no churn inside the window. The `active_vm_count` target is therefore the constant 10,000 and is forecast perfectly with zero error.
- Aggregate CPU utilization is a **low-level, low-volatility, short-memory series**: mean **9.39 %**, standard deviation **0.195 pp** (coefficient of variation **2.08 %**), range 8.98–9.91 %. Aggregation over 10,000 VMs smooths away most VM-level noise (the median per-VM temporal CV is 28.7 %, versus 2.1 % for the aggregate).
- A deliberately simple **lag-1/2/3 Ridge autoregression** wins leakage-safe rolling-origin validation (CV RMSE 0.1935 vs. 0.2174 for persistence) and, on the untouched 24-step holdout, achieves **MAE 0.101 pp, RMSE 0.122 pp, sMAPE 1.09 %** — a **31.4 % MAE reduction versus persistence**. Trend and seasonal models (ETS, ARIMA, seasonal-naïve) all underperform, because the data has no reliable seasonal component in an 11-hour window and only weak, noisy trend.
- Residual-scale analysis yields a concrete operational rule: provision to **forecast + 0.18 pp** (a one-sided p90 safety margin). This line was breached by only **1 of 24** holdout periods (4.2 %) while adding under 2 % relative headroom.
- The biggest operational opportunity is *not* aggregate headroom but **right-sizing**: 49 % of all VM readings are below 5 % CPU and the median VM averages only 5.48 %.

## 2. Data understanding and preprocessing

### 2.1 Panel audit (`reports/tables/eda_summary.json`)

| Check | Result |
|---|---|
| Rows / VMs / timestamps | 1,340,000 / 10,000 / 134 |
| Timestamp range | 0–39,900 s, exact 300 s step (verified against the full expected grid) |
| Observations per VM | exactly 134 for every VM (min = max = 134) |
| Duplicate `(vm_id, timestamp)` | 0 |
| Null cells | 0 |
| Distinct active VMs per timestamp | 10,000 at **every** timestamp |

No imputation, gap filling, or resampling was required. Targets are constructed from the complete panel *before* any split, so split policy cannot leak into target construction.

### 2.2 Target definitions (per `DATASET.md`)

- `active_vm_count(t)` = distinct VM ids with a reading at `t` (each VM counted once). Constant at 10,000.
- `aggregate_cpu_utilization(t)` = unweighted mean of VM-level `avg_cpu` at `t`, clipped to `[0, 100]`. Documented missingness policy: had readings been missing, the mean would be computed over observed VMs (missing-at-random within a timestamp); no back-filling. This case never arises in the curated data.
- The native five-minute cadence is retained; coarser aggregation would discard 134 already-scarce observations and is not justified.

### 2.3 Metadata and lifetimes

All 10,000 VMs have `timestamp_vm_created = 0` and the earliest deletion is 40,500 s — after the last observation at 39,900 s. The panel is a closed cohort with no birth/death churn inside the window. The VMs belong to 2,324 subscriptions and 2,359 deployments; 75 % are 2-vCPU instances.

## 3. Workload analysis

### 3.1 The fleet is mostly idle, but the aggregate looks stable

| Statistic | Value |
|---|---|
| Mean aggregate utilization | 9.388 % |
| Aggregate std / CV | 0.195 pp / 2.08 % |
| Readings below 1 % / 5 % / above 50 % CPU | 7.6 % / 49.0 % / 3.1 % |
| Median per-VM mean utilization | 5.48 % |
| Median per-VM temporal CV | 28.7 % |

The mean of per-VM means (9.39 %) equals the aggregate mean, while the *median* VM averages only 5.48 %: a minority of busier VMs pulls the mean up. The aggregate is smooth because idiosyncratic VM spikes diversify across 10,000 units — a textbook pooling effect. Operationally, aggregate headroom planning and VM-level right-sizing are different problems, and both matter here.

### 3.2 Temporal structure: short memory, no usable seasonality

The training-window autocorrelation function (`reports/figures/02_acf.png`) starts at 0.66 at lag 1 and decays steadily (0.60 lag 2, 0.45 lag 6, 0.33 lag 12, ≈ 0 at lag 24). There is no significant 12-step (hourly) or 24-step seasonal peak. Hourly block means peak in hours 4–5 (9.62–9.71 %) and bottom out in hour 9 (9.14 %), but with under 12 hours of data this could equally be drift; the weak lag-12 autocorrelation makes seasonal models unsafe. The series minimum (8.98 %) occurs at holdout step 2 and the maximum (9.91 %) at training step 55 — the holdout opens at a slightly lower level (holdout mean 9.27 % vs. train 9.41 %, a −0.145 pp shift).

### 3.3 Workload segments (`reports/figures/03_distributions.png`, `04_segment_series.png`)

| `vm_category` | VMs | Mean util. | Std (pp) |
|---|---:|---:|---:|
| Delay-insensitive | 6,180 | 8.24 % | 0.21 |
| Interactive | 3,091 | 11.49 % | 0.26 |
| Unknown | 729 | 10.23 % | 0.64 |

Interactive VMs run ~39 % hotter than delay-insensitive ones; the small Unknown group is three times more volatile. Segment-aware monitoring is therefore worthwhile even though the required forecasts are aggregate-level.

### 3.4 Why these observations matter operationally

1. With half of all readings under 5 % CPU, the fleet is heavily over-provisioned at the VM level — right-sizing (fewer/smaller instances) offers larger savings than tweaking aggregate buffers.
2. Short-memory dynamics mean near-term forecasts are feasible but long-horizon predictions converge quickly to the local level; planning should use short, frequently refreshed horizons with uncertainty bands.
3. A constant active count means capacity risk in this window is driven by *utilization*, not VM arrivals — but the metadata shows churn exists outside the window, so arrival modeling matters in production.

## 4. Forecasting approach

### 4.1 Candidate models (all implemented behind one `fit/forecast` interface)

1. **Persistence** — last observed value (primary baseline).
2. **Moving average (6)** — mean of the last six observations.
3. **Seasonal-naïve (12)** — repeats the last hour.
4. **ETS (damped Holt)** — additive damped trend, no seasonal term.
5. **ARIMA(1,0,1) with trend** — statistical benchmark, with an ETS fallback if MLE fails.
6. **Ridge AR (`ridge_ar`, selected)** — recursive multi-step autoregression on lags 1, 2, 3 with standardized inputs and Ridge (`alpha = 10`).

### 4.2 Why this feature set

Features for predicting position `t` use only values observed before `t`: lags `y_{t-1}, y_{t-2}, y_{t-3}`. Deterministic trend and cyclical features were prototyped and **rejected by CV**: with relative (not wall-clock) timestamps and under 12 hours of data they have no defensible calendar interpretation, and in recursive forecasting they drift as predictions feed back over 24 steps — the version with a trend term and lag 12 scored RMSE 0.49–1.22 versus 0.19 for the lag-only model. Lag set `(1, 2, 3)` and `alpha = 10` were chosen by training-only CV over lag sets `(1)` through `(1,2,3,6,12)` and alphas 0.1–100. Multi-step forecasts are recursive: each one-step prediction is appended and reused as a lag, so no future information is ever touched.

### 4.3 The constant target

For `active_vm_count`, every candidate produces identical zero-error forecasts; persistence is selected by the same CV machinery (all models tie), and the integer count is emitted unchanged. No modeling effort is spent where no signal exists.

## 5. Validation methodology

- **No random splitting anywhere.** The 134-point series is split chronologically into 110 training and 24 holdout observations; the holdout is opened once, after all model choices are frozen.
- **Model selection uses expanding-window rolling-origin CV inside the 110 training points** (`reports/figures/05_rolling_origin.png`): origins at positions 62, 74 and 86, each forecasting the next 24 steps. Fold 3 ends exactly at the training/holdout boundary, replicating the production task. Three folds × 24 steps give 72 out-of-sample CV residuals per model.
- All six models are scored under the identical scheme, and the same folds are reused for residual-based uncertainty estimation. On smoke-sized datasets the window shrinks adaptively while retaining at least one full-horizon fold (see tests).

## 6. Results versus baseline

### 6.1 Rolling-origin CV (training data only, mean RMSE / MAE across 3 folds)

| Model | CV RMSE | CV MAE |
|---|---:|---:|
| **ridge_ar (selected)** | **0.1935** | **0.1678** |
| persistence | 0.2174 | 0.1848 |
| seasonal-naïve 12 | 0.2755 | 0.2460 |
| ETS damped | 0.2764 | 0.2432 |
| moving average 6 | 0.2790 | 0.2491 |
| ARIMA(1,0,1) | 0.4124 | 0.3642 |

### 6.2 Final 24-step holdout (seen once)

| Metric | ridge_ar | Persistence |
|---|---:|---:|
| MAE (pp) | **0.1008** | 0.1468 |
| RMSE (pp) | **0.1218** | 0.1848 |
| sMAPE (%) | **1.086** | 1.585 |
| Relative MAE (model / persistence) | **0.686** | 1.000 |
| Max absolute error (pp) | 0.304 | — |
| Mean error bias (pp, + = over-forecast) | +0.031 | — |
| `active_vm_count` MAE / RMSE | 0.0 / 0.0 | 0.0 / 0.0 |

The model selected *without holdout access* delivers a 31.4 % holdout MAE improvement over the baseline — consistent with the 11 % CV improvement, i.e. no evidence of overfitting to validation folds.

### 6.3 Uncertainty and the capacity trigger

Pooling the 72 CV residuals gives σ = 0.14 pp. A constant-width **80 % empirical interval** (10th/90th pooled residual quantiles) covers **75 %** of holdout actuals (18/24), close to its nominal level given only three folds. A one-sided **p90 safety line = forecast + 1.2816σ ≈ forecast + 0.18 pp** is breached in only **1 of 24** holdout periods (4.2 %, nominal 10 %) — see `reports/figures/08_capacity_policy.png`.

## 7. Error and failure analysis

- **Abrupt innovations are irreducible.** The holdout's worst miss (step 2, error 0.21 pp) coincides with the series-wide minimum (8.98 %), a sharp drop immediately after the origin; no model using only past values can forecast a one-step shock. Likewise the single capacity-trigger breach is an isolated spike to 9.65 % with no precursor.
- **No horizon degradation.** Mean absolute CV residuals stay in a 0.08–0.24 pp band across steps 1–24 (`reports/tables/error_by_horizon_step.csv`); shrinkage keeps recursive error accumulation under control, which is precisely why regularized Ridge beats unconstrained ARIMA/ETS here.
- **Level-shift risk.** The holdout mean sits 0.145 pp below the training mean. The model carries a small positive bias (+0.031 pp); a sustained regime shift would be tracked only after a few observations once the rolling forecast refreshes. This argues for frequent re-fitting and the safety margin, not longer static horizons.
- **Peak vs. off-peak accuracy is statistically identical** (MAE 0.1006 vs. 0.1008 pp) because the aggregate lacks strong peak regimes; segment-level volatility (Unknown category, σ 0.64 pp) is where peak-specific risk concentrates.
- **Why the sophisticated models lost:** seasonal-naïve assumes an hourly cycle that the ACF does not support; ETS/ARIMA extrapolate a noisy local trend and overshoot; ARIMA MLE is fragile on 110 points (non-stationarity warnings, fallback paths). The Ridge AR model expresses exactly the short-memory structure the ACF shows.

## 8. Capacity-planning recommendation

**Policy (refreshed every 5 minutes, rolling 2-hour horizon):**

1. **Baseline capacity** = point forecast of aggregate utilization.
2. **Provisioned capacity** = forecast **+ 0.18 pp** one-sided p90 margin (≈ 1.9 % relative buffer on a 9.4 % mean). On this holdout that rule absorbs 23 of 24 periods.
3. **Autoscale trigger:** add capacity when the upper line approaches a resource ceiling (e.g. forecasted utilization > 70 % of a fleet-wide limit); never scale down below the point forecast within the horizon.
4. **Segment rule:** monitor Interactive VMs (11.5 % mean, hotter) and the volatile Unknown category on separate dashboards; apply a wider margin (their residual σ is ~2–3× the fleet aggregate) if they are scheduled onto dedicated pools.
5. **Right-sizing program:** independently of forecasting, audit the 49 % of VM-intervals under 5 % CPU for downsizing or consolidation — the expected savings dwarf the ~2 % aggregate safety buffer.

**Trade-off.** Under-provisioning risks throttling and SLA breaches during unforecastable spikes; over-provisioning wastes cost. With a fleet already averaging <10 % CPU, a 0.18 pp insurance margin is cheap, while the single observed breach shows why a zero-margin policy is unwise. Intervals should widen with horizon in production once more folds/data allow step-specific residual estimation.

## 9. Limitations and future work

- **Window length:** 11.1 hours cannot establish daily/weekly seasonality or support calendar features; all seasonality claims are deliberately avoided.
- **Few CV folds (3):** interval quantiles are noisy; the pooled, constant-width design is a deliberate robustness choice. A longer trace would enable step-specific and seasonal residual bands.
- **Closed cohort:** no VM churn occurs in-window, so `active_vm_count` predictability says little about arrival/departure forecasting in production.
- **Aggregation masks tail risk:** fleet-level smoothness coexists with high per-VM volatility; VM/cluster-level forecasting and quantile regression are natural extensions.
- **Single snapshot:** generalization across traces or clouds (e.g. Google ClusterData 2019) is untested.
- **Next steps:** (i) gradient boosting on lag + segment-aggregate features once more history permits; (ii) online/recursive re-estimation to adapt to level shifts; (iii) VM-level clustering and right-sizing models; (iv) probabilistic forecasting with calibrated multi-step quantiles.

---

### Reproducibility

`python -m cwfc.predict --data challenge_data --output predictions.csv --artifacts-dir reports/tables` regenerates the submission and CV artifacts; `python analysis/run_analysis.py --data challenge_data` regenerates every table and figure in this report; `pytest -q` runs 24 tests including leakage-safe feature checks and a smoke-dataset end-to-end run.
