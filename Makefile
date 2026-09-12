# Reproducible project targets (uv-based)
PY ?= uv run

.PHONY: setup fetch test eval report report-retail report-llm html clean

## Install the environment (runtime + dev)
setup:
	uv sync --extra dev

## Fetch the real Olist dataset into data/raw/olist
fetch:
	$(PY) python scripts/download_olist.py

## Run the full test suite (unit + integration + ground truth)
test:
	$(PY) python -m pytest

## Re-run the ground-truth evaluation (synthetic anomalies, seed 42)
eval:
	$(PY) python -m scripts.evaluate --orders 700 --customers 3500

## Investigate the real Olist dataset (deterministic, no LLM)
report:
	$(PY) investigator --data data/raw/olist --no-llm --out reports

## Same, but with the LLM decision layer (needs OPENROUTER_API_KEY in .env)
report-llm:
	$(PY) investigator --data data/raw/olist --out reports

## Investigate the synthetic retail (Spanish-schema) dataset
report-retail:
	$(PY) python scripts/synth_retail.py --out .eval/retail --anomalies
	$(PY) investigator --data .eval/retail --no-llm --out reports/retail

## Remove generated artifacts (reports, evaluation output, caches)
clean:
	rm -rf reports .eval .pytest_cache