# Cloud Workload Forecasting — Final Report

**Dataset:** Microsoft Azure Public Dataset V2, curated `data-v1` challenge release
**Series:** 134 five-minute observations (~11.1 hours) aggregated from 10,000 VMs / 1,340,000 CPU readings
**Holdout:** final 24 observations (2 hours), never used for fitting, feature selection, tuning, or model selection
**Selected model:** recursive autoregressive Ridge regression on lags 1–2, `alpha = 1` (`ridge_ar`), selected by a 21-candidate pre-study and 10-fold rolling-origin CV on the training window
**Decision journal:** see [`JOURNEY.md`](../JOURNEY.md) for the full rationale, failed approaches and selection-robustness analysis

---

## 1. Executive summary

- The curated panel is **complete and regular**: 10,000 VMs × 134 timestamps on an exact 300-second grid, with zero missing cells, zero duplicate `(vm, timestamp)` rows, and no churn inside the window. The `active_vm_count` target is therefore the constant 10,000 and is forecast perfectly with zero error.
- Aggregate CPU utilization is a **low-level, low-volatility, short-memory series**: mean **9.39 %**, standard deviation **0.195 pp** (coefficient of variation **2.08 %**), range 8.98–9.91 %. Aggregation over 10,000 VMs smooths away most VM-level noise (the median per-VM temporal CV is 28.7 %, versus 2.1 % for the aggregate).
- A deliberately simple **lag-1/2 Ridge autoregression** wins leakage-safe rolling-origin validation across three different CV designs (best 10-fold CV RMSE 0.228 vs. 0.283 for persistence). On the untouched 24-step holdout it achieves **MAE 0.124 pp, RMSE 0.149 pp, sMAPE 1.34 %** — a **15.5 % MAE reduction versus persistence**. A wider 21-candidate pre-study (gradient boosting, ElasticNet, ETS, ARIMA, SES, drift, differenced AR, ensembles) found nothing better; see §4 and `JOURNEY.md`.
- Residual-scale analysis yields a concrete operational rule: provision to **forecast + 0.30 pp** (a one-sided p90 safety margin from 240 CV residuals). This line was breached by **0 of 24** holdout periods while adding ~3 % relative headroom.
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

### 4.1 The pre-study: 21 candidates, training data only

Before fixing the production registry, a broad pre-study (`analysis/model_experiments.py`) scored 21 candidates under three rolling-origin designs, never touching the holdout. Families tested: persistence, moving averages, seasonal-naïve, simple/damped exponential smoothing, drift, ARIMA(1,0,1), Ridge AR over lag sets (1) through (1,2,3,6) and four regularization strengths, ElasticNet, differenced AR, histogram gradient boosting, and three equal-weight ensembles. Findings:

- **Regularized linear AR wins decisively** (best CV RMSE 0.228 vs. 0.283 persistence).
- Gradient boosting overfits ~50 supervised rows per fold (0.268); ElasticNet's sparsity adds nothing; ensembles are dragged down by weaker, highly correlated members (0.239+); differenced and drift models amplify noise; ARIMA is the worst family (0.442) due to fragile MLE and trend overshoot on 110 points.
- Among Ridge configs the gap is tiny; lags **(1,2), α = 1** has the best average rank across all three CV schemes and wins the densest 10-fold design. It is fixed as the production configuration.

### 4.2 Production model registry (eight models, one interface)

`persistence`, `moving_average_6`, `seasonal_naive_12`, `ses`, `drift`, **`ridge_ar` (selected)**, damped `ets_damped`, `arima_101`. All share one `fit/forecast` interface and run through identical CV. The selected model is a recursive autoregression: standardized lag features `y_{t-1}, y_{t-2}` → Ridge (`alpha = 1`). Each one-step prediction is appended and reused as a lag for the next step, so the 24-step forecast never reaches for unobserved data.

### 4.3 Why this feature set

Features for predicting position `t` use only values observed before `t`. Deterministic trend and cyclical features were prototyped and **rejected by CV**: with relative (not wall-clock) timestamps and under 12 hours of data they have no defensible calendar interpretation, and in recursive forecasting they drift as predictions feed back over 24 steps — the rich-feature version scored CV RMSE 0.49–1.22 versus ~0.23 for lag-only. Autocorrelation being effectively zero beyond a handful of lags is what makes the two-lag model sufficient.

### 4.4 The constant target

For `active_vm_count`, every candidate produces identical zero-error forecasts; persistence is selected by the same CV machinery (all models tie), and the integer count is emitted unchanged. No modeling effort is spent where no signal exists.

## 5. Validation methodology

- **No random splitting anywhere.** The 134-point series is split chronologically into 110 training and 24 holdout observations; the holdout is opened once, after all model choices are frozen.
- **Model selection uses expanding-window rolling-origin CV inside the 110 training points** (`reports/figures/05_rolling_origin.png`): ten origins at positions 48, 52, …, 86 (step 4), each forecasting the next 24 steps. Fold 10 ends exactly at the training/holdout boundary, replicating the production task. Ten folds × 24 steps give **240 out-of-sample CV residuals** per model — enough for stable leaderboards and interval calibration. On smoke-sized datasets the window shrinks adaptively while retaining at least one full-horizon fold.
- The pre-study additionally cross-checked every candidate under a 7-fold and the original 3-fold design; the chosen Ridge config ranks top-three in all three.

## 6. Results versus baseline

### 6.1 Rolling-origin CV (10-fold, training data only)

| Model | CV RMSE | CV MAE |
|---|---:|---:|
| **ridge_ar (selected)** | **0.2278** | **0.1976** |
| ses | 0.2681 | 0.2355 |
| persistence | 0.2826 | 0.2467 |
| moving average 6 | 0.2830 | 0.2510 |
| ETS damped | 0.2948 | 0.2609 |
| seasonal-naïve 12 | 0.3199 | 0.2832 |
| drift | 0.3261 | 0.2864 |
| ARIMA(1,0,1) | 0.4420 | 0.3821 |

### 6.2 Final 24-step holdout (seen once)

| Metric | ridge_ar | Persistence |
|---|---:|---:|
| MAE (pp) | **0.1240** | 0.1468 |
| RMSE (pp) | **0.1491** | 0.1848 |
| sMAPE (%) | **1.336** | 1.585 |
| Relative MAE (model / persistence) | **0.845** | 1.000 |
| Max absolute error (pp) | 0.279 | — |
| Mean error bias (pp, + = over-forecast) | +0.087 | — |
| `active_vm_count` MAE / RMSE | 0.0 / 0.0 | 0.0 / 0.0 |

### 6.3 Model-selection uncertainty, stated honestly

A neighboring Ridge configuration (lags 1–3, α = 10) wins exactly 5 of the 10
paired CV folds, with a mean RMSE gap of just 0.003 — the two are
statistically equivalent, and on this *single* holdout draw that alternative
scores MAE 0.101 versus 0.124. We do **not** switch to it: re-selecting on
holdout numbers is the leakage the protocol forbids, and a 24-point sample
cannot distinguish two models that tie in ten folds. The pipeline therefore
commits to the rigorous CV winner and treats holdout differences of this size
as selection noise. Full fold-by-fold evidence is in `JOURNEY.md` §7.

### 6.4 Uncertainty and the capacity trigger

Pooling the 240 CV residuals gives σ = 0.24 pp. A constant-width **80 % empirical interval** (10th/90th pooled residual quantiles) covers **100 %** of holdout actuals (24/24) on this draw. A one-sided **p90 safety line = forecast + 1.2816σ ≈ forecast + 0.30 pp** is breached in **0 of 24** holdout periods — see `reports/figures/08_capacity_policy.png`. Earlier validation with a coarser 3-fold design showed a 1/24 breach, so the margin is deliberately conservative rather than tuned to this holdout.

## 7. Error and failure analysis

- **Abrupt innovations are irreducible.** The holdout's series-wide minimum (8.98 %) arrives one step after the origin with no precursor; no model using only past values can forecast a one-step shock. Errors are largest in early holdout steps and shrink as the recursive model tracks the new level.
- **No horizon degradation.** Mean absolute CV residuals stay bounded across steps 1–24 (`reports/tables/error_by_horizon_step.csv`); shrinkage keeps recursive error accumulation under control, which is precisely why regularized Ridge beats unconstrained ARIMA/ETS here.
- **Level-shift risk.** The holdout mean sits 0.145 pp below the training mean, producing the model's +0.087 pp upward bias. A sustained regime shift is tracked only after a few observations once the rolling forecast refreshes — arguing for frequent re-fitting and the safety margin, not longer static horizons.
- **Peak vs. off-peak.** Peak-tercile MAE (0.064 pp) is actually lower than off-peak MAE (0.154 pp) on this holdout, because the early-holdout trough dominates the errors; segment-level volatility (Unknown category, σ 0.64 pp) remains where peak-specific risk concentrates.
- **Why the sophisticated models lost:** seasonal-naïve assumes an hourly cycle the ACF does not support; ETS/ARIMA extrapolate a noisy local trend and overshoot; gradient boosting cannot extrapolate and overfits ~50 training rows. The Ridge AR model expresses exactly the short-memory structure the ACF shows.

## 8. Capacity-planning recommendation

**Policy (refreshed every 5 minutes, rolling 2-hour horizon):**

1. **Baseline capacity** = point forecast of aggregate utilization.
2. **Provisioned capacity** = forecast **+ 0.30 pp** one-sided p90 margin (≈ 3.2 % relative buffer on a 9.4 % mean), calibrated from 240 CV residuals. On this holdout the rule absorbs all 24 periods.
3. **Autoscale trigger:** add capacity when the upper line approaches a resource ceiling (e.g. forecasted utilization > 70 % of a fleet-wide limit); never scale down below the point forecast within the horizon.
4. **Segment rule:** monitor Interactive VMs (11.5 % mean, hotter) and the volatile Unknown category on separate dashboards; apply a wider margin (their residual σ is ~2–3× the fleet aggregate) if they are scheduled onto dedicated pools.
5. **Right-sizing program:** independently of forecasting, audit the 49 % of VM-intervals under 5 % CPU for downsizing or consolidation — the expected savings dwarf the ~3 % aggregate safety buffer.

**Trade-off.** Under-provisioning risks throttling and SLA breaches during unforecastable spikes; over-provisioning wastes cost. With a fleet already averaging <10 % CPU, a 0.30 pp insurance margin is cheap, while the pre-study's evidence of occasional breaches shows why a zero-margin policy is unwise. Intervals should be re-calibrated as more folds/data accumulate.

## 9. Limitations and future work

- **Window length:** 11.1 hours cannot establish daily/weekly seasonality or support calendar features; all seasonality claims are deliberately avoided.
- **Model-selection noise:** top Ridge configurations are statistically tied in CV; a single 24-point holdout cannot rank them, and we explicitly refuse to try.
- **Closed cohort:** no VM churn occurs in-window, so `active_vm_count` predictability says little about arrival/departure forecasting in production.
- **Aggregation masks tail risk:** fleet-level smoothness coexists with high per-VM volatility; VM/cluster-level forecasting and quantile regression are natural extensions.
- **Single snapshot:** generalization across traces or clouds (e.g. Google ClusterData 2019) is untested.
- **Next steps:** (i) online/recursive re-estimation to adapt to level shifts; (ii) VM-level clustering and right-sizing models; (iii) probabilistic forecasting with calibrated step-specific quantiles once a longer trace supplies enough folds; (iv) an automated backtest-stability dashboard flagging leaderboard changes across fold designs (the pre-study machinery is already 80 % of this).

---

### Reproducibility

`python -m cwfc.predict --data challenge_data --output predictions.csv --artifacts-dir reports/tables` regenerates the submission and CV artifacts; `python analysis/run_analysis.py --data challenge_data` regenerates every table and figure; `python analysis/model_experiments.py --data challenge_data` reruns the 21-candidate pre-study; `pytest -q` runs 29 tests including leakage-safe feature checks, determinism, and a smoke-dataset end-to-end run of both the CLI and the full analysis. Or simply `make all`.
