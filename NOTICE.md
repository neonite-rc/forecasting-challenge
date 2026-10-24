# Notice — License Scope

This repository combines upstream open-source starter material with an
original, restrictively licensed solution. Two licenses therefore coexist:

| Material | License | File |
|---|---|---|
| Original solution work (listed below) | Restricted proprietary, all rights reserved | [`LICENSE`](LICENSE) |
| Upstream starter code and assignment documents from `atomityhq/forecasting-challenge` | MIT | [`LICENSE-MIT`](LICENSE-MIT) |

The MIT license cannot be revoked on upstream material, so those files remain
under MIT regardless of the proprietary notice applied to the new work.

## Covered by the restrictive LICENSE (proprietary)

- `src/cwfc/`: `config.py`, `data.py`, `features.py`, `models.py`,
  `evaluation.py`, `pipeline.py`, `predict.py`
- Root: `predict.py`, `conftest.py`, `Makefile`, `requirements.txt`,
  `JOURNEY.md`
- `analysis/` (all programs)
- `tests/`: `smoke_data.py`, `test_targets.py`, `test_features.py`,
  `test_models.py`, `test_intervals.py`, `test_pipeline_cli.py`,
  `test_reproducibility.py`, `test_entrypoints.py`
- `reports/final_report.md` and all generated tables and figures under
  `reports/`
- `predictions.csv` and `docs/`

Under the restrictive LICENSE, commercial use, derivative works,
redistribution, and production deployment are not granted; a limited license
for evaluation and non-commercial study is provided in Section 1 of `LICENSE`.

## Covered by LICENSE-MIT (upstream, MIT)

- Assignment and dataset contract documents: `ASSIGNMENT.md`, `DATASET.md`,
  `SUBMISSION.md`, `DATA_CARD_TEMPLATE.md`, `DATA_RELEASE.md`, `SOURCES.md`,
  `CODE_OF_CONDUCT.md`, `challenge_data/README.md`
- Original starter modules and their direct modifications:
  `src/cwfc/__init__.py`, `src/cwfc/baselines.py`, `src/cwfc/metrics.py`,
  `src/cwfc/validation.py`
- Starter tests: `tests/test_metrics.py`, `tests/test_validation.py`
- Project scaffolding originating upstream: `pyproject.toml`, `.gitignore`,
  and the original `README.md` structure

The challenge dataset itself is distributed separately via the upstream
GitHub Release and is not committed to this repository.

---

Copyright (c) 2026 Biswaketan (GitHub: neonite-rc). All rights reserved
except where an MIT notice applies.
