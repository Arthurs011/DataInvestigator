# DataInvestigator — Evaluation & Ground Truth

How do we know the investigator is *correct* and not just plausibility-sounding?
This project answers that with a **ground-truth evaluation**: we inject
anomalies with known parameters into a synthetic dataset, run the full pipeline,
and require **recall = 3/3 with zero ghosting**. We also prove the loader and
detectors generalize to a completely different naming scheme.

## 1. Methodology

The evaluation in `scripts/evaluate.py` builds two copies of the same seeded
dataset:

- a **clean** copy (no anomalies), and
- an **injected** copy with three deterministic anomalies.

The full pipeline runs on both copies (`--no-llm`, i.e. the deterministic engine
only). Each anomaly must satisfy two checks:

1. **Recovered**: the injected run reports a finding that matches the anomaly's
   signature (a `match` predicate per anomaly), and
2. **Not ghosted**: the clean run reports nothing matching that signature.

An anomaly that appears in the clean run is a **false positive (ghost)**; an
anomaly missing from the injected run is a **miss**. The evaluation passes only
with 3 recovered and 0 ghosted. Exit code 0 in CI.

## 2. The injected anomalies (scripts/synth_data.py)

Data: Olist-schema synthetic e-commerce, 8 quarters (2019Q1–2020Q4),
700 orders/quarter, 3500 customers, seed 42. Prices are drawn from a seeded
lognormal per category.

| Key | Injection | Signature the pipeline must find |
|---|---|---|
| **A** | 2020Q3 `price × 0.45` (all states) | `time_change`, revenue, period `2020Q3`, prev `2020Q2`, Δ < −15% |
| **B** | `furniture` `price × 5.0` in state `TS` | `graph` finding, state `TS`, category `furniture` |
| **C** | 2020-06 `price × 4.0` (all states) | `time_anomaly`, period `2020-06`, z ≥ 2 (spike) |

The match predicates (in `scripts/evaluate.py`) encode the detection language
the engine is expected to use, not the injection internals — e.g. C matches any
`time_anomaly` in 2020-06 with z ≥ 2, i.e. an *upward* spike (a downward z-score
is not a match).

## 3. Results

Run:

```bash
uv run python -m scripts.evaluate --orders 700 --customers 3500
```

Result (seed 42, deterministic):

| Anomaly | Recovered | Ghosted | Rank | Score |
|---|---|---|---|---|
| A: revenue drop in 2020Q3 (price ×0.45) | YES | NO | 2 | 0.717 |
| B: 'furniture' ×5.0 revenue in state TS | YES | NO | 10 | 0.505 |
| C: revenue spike in 2020-06 (price ×4.0) | YES | NO | 9 | 0.513 |

**Final status: PASSED (recall 3/3, zero ghosting).** The same assertions ship
as an integration test in `tests/test_ground_truth.py`.

### What the three anomalies exercise

- **A** exercises the quarterly-change detector, its period-completeness guard,
  the drill-down chain, SQL evidence, and the causal regression.
- **B** exercises the category×state graph index and the ranking diversity pass
  (it lands at rank 10 only because diversity protects under-represented kinds).
- **C** exercises the rolling median/MAD z-score detector and its volume guard
  (the spike month must still contain ≥ 200 orders or it is legitimately
  skipped).

## 4. Found and fixed along the way

The evaluation caught real bugs:

1. **Non-reproducible data** — one price draw used the global (unseeded)
   `np.random`, so every dataset build produced different data and B would
   randomly fall out of the top-10. Moved every draw onto the seeded generator;
   datasets are now byte-identical across runs.
2. **Ghosted C** — the C matcher only checked the month, not the direction; the
   clean data naturally contains a *dip* in 2020-06 (z ≈ −2.1) above the
   |z| ≥ 2 threshold. Required z ≥ 2 (upward spike) to disambiguate.
3. **Suppressed singular-matrix warnings** in the bootstrap and primary OLS fit
   so evaluation/debug output stays readable and degenerate panels don't
   mislead.

## 5. Generalization: second dataset (`tests/test_retail_dataset.py`)

`scripts/synth_retail.py` generates the *same kind* of data but in a different
**naming scheme** — Spanish-mapped retail tables (`transacciones`,
`lineas_pedido`, `clientes`, `catalogo`, `categorias`, `tiendas`) with columns
like `txn_id`, `fulfillment_status`, `sales_datetime`, `outlet_id`, `amount`,
`province`, `department`. It also injects A (2020Q3 drop ×0.5) and C (2020-06
spike ×4.0).

The integration test asserts that, with **zero changes** to the analytical code:

- the loader canonicalizes the Spanish schema onto the internal names,
- the fact table is built with revenue and dimensions,
- the pipeline detects the 2020Q3 drop and the 2020-06 spike,
- some finding carries SQL evidence that cross-verifies against the engine.

This is the evidence that the discovery stages are general: they depend on the
canonical schema, not on the source dataset's vocabulary.

## 6. What we do not evaluate (limitations)

- **No end-to-end false-positive rate** across many seeds: the eval proves the
  three signatures are recoverable on seed 42, not that the engine's detectors
  have optimal precision on arbitrary data.
- **LLM selection is not quantitatively scored** here — narratives are judged
  qualitatively; the *numbers* the LLM narrates are protected by the verified
  evidence layer.
- **Small data may yield `insufficient_data`** causal verdicts; that is honest
  behavior, not a false positive, but it means causal *proof* is conditional on
  panel size.
- Both synthetic datasets use clean, complete, well-behaved joins; real-world
  duplicates and dirty keys are only partially exercised (the Olist run is the
  realistic stress test).