# Notice — License Scope

This repository combines upstream open-source starter material with an
original work released under the PolyForm Noncommercial License 1.0.0. Two
licenses therefore coexist:

| Material | License | File |
|---|---|---|
| Original solution work (listed below) | PolyForm Noncommercial 1.0.0 | [`LICENSE`](LICENSE) |
| Upstream starter code and assignment documents from `atomityhq/forecasting-challenge` | MIT | [`LICENSE-MIT`](LICENSE-MIT) |

The MIT license cannot be revoked on upstream material, so those files remain
under MIT regardless of the license applied to the new work.

## What PolyForm Noncommercial 1.0.0 permits and forbids

Permitted without further permission: viewing, cloning, running, reviewing,
testing, research and personal study, making noncommercial changes, and
noncommercial redistribution with this notice retained. The top-level
LICENSE additionally grants the challenge recruitment team an explicit,
narrow license to run and evaluate the submission for hiring assessment.

Not permitted without prior written authorization: any **commercial use**,
including operating the software (or a derivative) in a commercial
production environment, incorporating it into a proprietary product or
service, or commercial redistribution or sublicensing. PolyForm
Noncommercial is source-available, not OSI open source, because an OSI
license may not forbid commercial use.

## Covered by the PolyForm Noncommercial LICENSE

- `src/cwfc/`: `config.py`, `data.py`, `features.py`, `models.py`,
  `evaluation.py`, `pipeline.py`, `predict.py`, `_about.py`
- Root: `predict.py`, `conftest.py`, `Makefile`, `requirements.txt`,
  `JOURNEY.md`
- `analysis/` (all programs)
- `tests/`: `smoke_data.py`, `test_targets.py`, `test_features.py`,
  `test_models.py`, `test_intervals.py`, `test_pipeline_cli.py`,
  `test_reproducibility.py`, `test_entrypoints.py`
- `reports/final_report.md` and all generated tables and figures under
  `reports/`
- `predictions.csv` and `docs/`

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

Required Notice: Copyright (c) 2026 Biswaketan (https://biswaketan.in)
