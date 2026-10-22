# Cloud Workload Forecasting Challenge — developer workflow
# Requires: Python 3.11+, and the data-v1 release extracted into challenge_data/

PYTHON ?= python3
DATA ?= challenge_data

.PHONY: help install test predict analysis experiments clean all

help:
	@echo "make install    - install the package with dev dependencies"
	@echo "make test       - run the full pytest suite (32 tests)"
	@echo "make predict    - regenerate predictions.csv and CV artifacts"
	@echo "make analysis   - regenerate all EDA tables and report figures"
	@echo "make experiments - rerun the training-only model pre-study"
	@echo "make all        - test + predict + analysis"
	@echo "make clean      - remove generated caches"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

predict:
	$(PYTHON) -m cwfc.predict --data $(DATA) --output predictions.csv \
		--artifacts-dir reports/tables

analysis:
	$(PYTHON) analysis/run_analysis.py --data $(DATA)

experiments:
	$(PYTHON) analysis/model_experiments.py --data $(DATA)

all: test predict analysis

clean:
	rm -rf .pytest_cache src/*.egg-info src/cwfc/__pycache__ tests/__pycache__ \
		analysis/__pycache__ .ipynb_checkpoints
	find . -type d -name __pycache__ -not -path "./.git/*" -exec rm -rf {} +
