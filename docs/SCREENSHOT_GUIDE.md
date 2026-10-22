# Runbook & Screenshot Guide

A step-by-step guide to running the full solution from a clean checkout and
capturing screenshots for documentation. Every command below was verified
against the repository; **expected output** blocks show exactly what a
successful run looks like on the `data-v1` release, so you can compare your
screen against them.

- Total hands-on time: ~15 minutes (plus the 111 MB dataset download).
- OS-specific terminals: **Terminal** (macOS/Linux) or **PowerShell**
  (Windows). Windows variants are noted where they differ.
- Suggested screenshot folder: `docs/screenshots/` with the filenames used in
  the checklist below.

---

## 0. Prerequisites

| Requirement | Check command | Expected |
|---|---|---|
| Python 3.11+ | `python3 --version` (Windows: `py --version`) | e.g. `Python 3.13.x` |
| Git | `git --version` | any recent version |
| ~250 MB free disk | — | dataset + packages |
| Network access | — | PyPI + the GitHub release |

## 1. Get the code

```bash
git clone https://github.com/neonite-rc/forecasting-challenge.git
cd forecasting-challenge
```

(Or download the ZIP from the GitHub page and extract it.)

## 2. Create a virtual environment and install dependencies

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

**Windows PowerShell:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you cannot do the editable install, the fallback is
`pip install -r requirements.txt` — the root `predict.py` shim and
`conftest.py` make every documented command work without `-e`.

## 3. Download and verify the dataset

The dataset is **not** in Git; it is a GitHub Release asset.

```bash
mkdir -p challenge_data
# download data-v1.tar.gz from:
# https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1
tar -xzf data-v1.tar.gz -C challenge_data
cd challenge_data && sha256sum -c SHA256SUMS && cd ..
```

Windows (PowerShell, `tar` ships with Windows 10+):

```powershell
tar -xzf data-v1.tar.gz -C challenge_data
certutil -hashfile challenge_data\vm_cpu_readings.csv.gz SHA256
# compare the printed hash against challenge_data\SHA256SUMS
```

Expected checksum result (Linux/macOS):

```text
vm_cpu_readings.csv.gz: OK
vm_metadata.csv.gz: OK
DATA_CARD.md: OK
```

The folder must now contain `vm_cpu_readings.csv.gz`, `vm_metadata.csv.gz`,
`DATA_CARD.md`, and `SHA256SUMS`.

---

## 4. Run the test suite — 📸 screenshot 01

```bash
python -m pytest -q
# or: make test
```

**Expected (last line):**

```text
33 passed in 4.56s
```

What this proves: target-contract tests, leakage-safe feature checks, all
eight forecasters, interval/calibration helpers, determinism, and an
end-to-end run on a synthetic smoke dataset. If you skipped the editable
install, this still works thanks to `conftest.py`.

## 5. Generate the submission — 📸 screenshot 02

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv \
    --artifacts-dir reports/tables
# or: make predict
# install-free fallback:
# python predict.py --data challenge_data --output predictions.csv --artifacts-dir reports/tables
```

**Expected console output:**

```text
wrote 24 predictions to predictions.csv
  active_vm_count: selected model = persistence
    holdout MAE=0.0000 RMSE=0.0000 sMAPE=0.000% rel.MAE(persistence)=0.000
  aggregate_cpu_utilization: selected model = ridge_ar
    holdout MAE=0.1240 RMSE=0.1491 sMAPE=1.336% rel.MAE(persistence)=0.845
wrote CV artifacts to reports/tables
```

Then show the file itself — 📸 screenshot 03:

```bash
head -5 predictions.csv && echo "..." && tail -2 predictions.csv && wc -l predictions.csv
```

**Expected:** 25 lines (1 header + 24 predictions), timestamps 33000 → 39900
in 300-second steps, and `active_vm_count = 10000` on every row:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
33000,10000,9.203695
33300,10000,9.238027
33600,10000,9.26596
...
39600,10000,9.401887
39900,10000,9.40248
25 predictions.csv
```

## 6. Regenerate every analysis table and figure — 📸 screenshot 04

```bash
python analysis/run_analysis.py --data challenge_data
# or: make analysis
```

**Expected final console block (JSON metrics):**

```text
analysis artifacts written to reports/
{
  "mae": 0.12396365606339439,
  "rmse": 0.149101195229035,
  ...
}
{
  "interval_80pct_holdout_coverage": 1.0,
  "capacity_trigger_holdout_breach_rate_empirical": 0.0,
  ...
}
```

Confirm the artifacts — same screenshot or a new one (05):

```bash
ls reports/figures reports/tables
```

**Expected:** 9 PNGs (`01_…` through `09_calibration.png`) and 15 table files
(CSV/JSON). Open the figures in any image viewer — see the figure map below.

## 7. Rerun the 22-candidate pre-study — 📸 screenshot 06

```bash
python analysis/model_experiments.py --data challenge_data
# or: make experiments
```

**Expected tail (the refinement-round evidence):**

```text
== intercept-correction study (dense 10-fold) ==
ridge_no_correction          RMSE=0.2278 fold_sd=0.0682
bias_insample_last12         RMSE=0.2721 fold_sd=0.1187
...
== paired test: lags(1,2) a=1 vs lags(1,2,3) a=10 (dense 10-fold) ==
paired_t_pvalue              0.5944540010854108
wilcoxon_pvalue              0.845703125
```

This is the script that justifies every modeling choice; its three CV blocks
each print a full ranked leaderboard.

## 8. One-command full run (optional) — 📸 screenshot 07

```bash
make all     # test + predict + analysis
```

---

## Screenshot checklist

| # | File to save | What to capture | Success signal to show |
|---|---|---|---|
| 01 | `01_pytest.png` | terminal running pytest | `33 passed` |
| 02 | `02_predict_cli.png` | the predict command + metrics | `MAE=0.1240 ... rel.MAE=0.845` |
| 03 | `03_predictions_csv.png` | `head`/`tail` of predictions.csv | 25 lines, timestamps 33000–39900 |
| 04 | `04_analysis_cli.png` | analysis script finishing | `analysis artifacts written to reports/` |
| 05 | `05_artifacts_listing.png` | `ls reports/figures reports/tables` | 9 figures + 15 tables |
| 06 | `06_experiments.png` | pre-study leaderboard tail | paired-test p-values 0.59 / 0.85 |
| 07 | `07_make_all.png` | `make all` completing | tests pass + predictions written |
| 08–16 | `08_fig01_target.png` … `16_fig09_calibration.png` | the nine report figures opened full-size | see figure map below |

### Figure map (for captions)

| File | What it shows |
|---|---|
| `01_target_series.png` | Aggregate utilization over 134 steps with the 110/24 split line |
| `02_acf.png` | Autocorrelation decay — short memory, no seasonality |
| `03_distributions.png` | Per-VM utilization distribution + category boxplots |
| `04_segment_series.png` | Utilization time series by VM category |
| `05_rolling_origin.png` | The ten expanding-window CV folds (blue train / orange validation) |
| `06_holdout_forecast.png` | 2-hour forecast vs actuals with 80% band |
| `07_error_by_step.png` | Absolute error per forecast horizon step |
| `08_capacity_policy.png` | Empirical p90 vs Gaussian capacity trigger lines |
| `09_calibration.png` | Nominal vs empirical interval-coverage calibration curve |

### How to capture screenshots

- **Windows:** `Win + Shift + S` (snip), then paste/save into
  `docs/screenshots/`.
- **macOS:** `Cmd + Shift + 4`, drag to select; files land on the Desktop —
  move them into `docs/screenshots/`.
- **Linux (GNOME):** `Print Screen` or `Shift + Print Screen` for a region.
- For figures, open the PNG full-size (double-click, or
  `xdg-open reports/figures/06_holdout_forecast.png`) and snip the viewer.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'cwfc'` | The editable install is missing. Run `pip install -e ".[dev]"`, or use the install-free `python predict.py ...` shim and `python analysis/...` (tests work either way via `conftest.py`). |
| `No module named 'statsmodels'` (or sklearn/scipy) | Dependencies not installed in the active environment. Run `pip install -r requirements.txt` and confirm the venv is activated (prompt prefix `(.venv)`). |
| `FileNotFoundError: vm_cpu_readings.csv.gz` | Wrong `--data` path or the release was not extracted into `challenge_data/`; repeat step 3. |
| Checksum mismatch | Re-download `data-v1.tar.gz`; never edit the gzip files. |
| `ValueError: need more than ... observations` | You pointed `--data` at an incomplete dataset; the release has exactly 134 timestamps. |
| Numbers differ slightly from this guide | Confirm you are on the latest commit (`git log --oneline -1`) and rerun; the pipeline is deterministic, so identical code + data give byte-identical `predictions.csv`. |
| `make: command not found` (Windows) | Use the raw `python -m ...` commands shown beside each Make target; Make is optional. |

## Sanity check before you finish

```bash
git status        # predictions.csv and reports/ regenerated; no data files staged
python -m pytest -q   # 33 passed
```

The 111 MB `challenge_data/*.csv.gz` files are git-ignored and must never be
committed; only code, docs, small tables, figures, and `predictions.csv`
belong in the repository.
