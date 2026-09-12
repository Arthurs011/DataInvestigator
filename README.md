# DataInvestigator

An autonomous AI "investigator" for real-world data. Give it a folder of CSVs and it
investigates the dataset and returns an **Investigation Report** of evidence-backed
findings — not just "average sales", but *"revenue decreased 18.3% in Q3, concentrated
in Region X → Product A → Customer segment B (orders -31%), confidence 91%."*

```
                   DATA
                    │
                    ▼
             Data understanding
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    Statistical              Graph
    investigation          discovery
          │                   │
          └─────────┬─────────┘
                    ▼
              Hypothesis engine
                    │
                    ▼
              Causal testing
                    │
                    ▼
             Evidence ranking
                    │
                    ▼
              INVESTIGATION
                 REPORT
```

Pipeline stages map to modules in `src/investigator/`:

| Stage | Module |
|---|---|
| Data understanding | `loader.py`, `profile.py` |
| Statistical investigation | `stats_engine.py` |
| Graph discovery | `graph_engine.py` |
| Hypothesis engine | `hypothesis_engine.py` |
| Causal testing | `causal_engine.py` |
| Evidence ranking | `ranking.py` |
| Report | `llm_reporter.py`, `report.py` |

The engine is deterministic (reproducible); the LLM (OpenRouter) selects/prioritizes
candidate findings and writes narrative prose. Every finding ships with DuckDB **SQL
evidence** and pandas **Python evidence** so the numbers can be re-run by hand.

## Setup

```bash
uv sync
cp .env.example .env        # add OPENROUTER_API_KEY
```

Fetch the real test dataset (Olist Brazilian e-commerce, 9 CSVs, ~50 MB):

```bash
uv run python scripts/download_olist.py
```

## Run

```bash
uv run investigator --data data/raw/olist --out reports/
uv run investigator --data data/raw/olist --no-llm     # deterministic only (no API key)
```

## Development

```bash
uv sync --extra dev            # install pytest
uv run pytest                  # full suite (unit + integration on Olist data)
uv run pytest -m unit          # fast tests only
```

## Evaluation (ground truth)

`scripts/synth_data.py` generates a synthetic dataset in Olist's schema with three
**known** injected anomalies — the ground truth the investigator must rediscover:

| Key | Anomaly | How it's detected |
|---|---|---|
| A | 2020Q3 revenue drop (price ×0.45) | quarterly `time_change` |
| B | `furniture` ×5.0 revenue in state `TS` | geo×category `graph` over-index |
| C | 2020-06 revenue spike (price ×4.0) | monthly `time_anomaly` (rolling z ≥ 2) |

The harness builds a *clean* copy and an *injected* copy of the same seeded dataset,
runs the full pipeline on both, and asserts **recall = 3/3 with zero ghosting** (an
anomaly must appear only on the injected data):

```bash
uv run python -m scripts.evaluate --orders 700 --customers 3500
```

Current result (seed 42): A rank 2, B rank 10, C rank 9 — all recovered, none ghosted.
The same assertions ship as integration tests in `tests/test_ground_truth.py`.

> Note: `build_dataset` seeds both the default and per-draw RNG, so every run is
> reproducible; the pipeline must recover the anomalies deterministically.