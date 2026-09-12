# DataInvestigator — System Design

This document describes the architecture and design decisions behind
**DataInvestigator**, an autonomous agent that investigates a folder of raw CSVs
and returns an *Investigation Report* of evidence-backed findings. It is written
to support the major-project thesis: every stage is grounded in a reproducible,
testable mechanism, and the LLM is deliberately constrained to *selection and
narrative* — never to number generation.

## 1. Objective and guarantees

Given a directory of relational CSVs (no metadata, mixed naming schemes), the
system must, without human guidance:

1. Discover tables and their relationships.
2. Detect anomalies: sudden metric changes, segment shifts, regional
   over-indexing, monthly outliers, and numeric correlations.
3. Attribute each finding to a cause-location chain (drill-down) and attach
   **reproducible evidence** — DuckDB SQL *and* pandas code that reproduce the
   headline numbers.
4. Test whether a detected relationship is causal after controlling for
   plausible confounders.
5. Rank, select, and narrate the findings for a non-technical reader.

Guarantees:

- **Reproducible**: the deterministic engine is seeded and pure with respect to
  its inputs; the same data + configuration always yields the same findings and
  evidence.
- **Verifiable**: headline numbers are cross-checked against a second,
  independent computation path (SQL views vs. pandas), and each finding is
  marked `verified` only when they agree within tolerance.
- **Honest**: the LLM receives only that reproducible evidence; it is forbidden
  from inventing numbers, and causal verdicts are surfaced verbatim.

## 2. Pipeline

```
        DATA (directory of CSVs)
                 │
                 ▼
   ┌────────────────────────────┐
   │ 1. Loader + canonicalizer  │  loader.py
   │    schema-agnostic intake  │
   └────────────────────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │ 2. Profiling & joins       │  profile.py
   │    table stats, FK graph   │
   └────────────────────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │ 3. Statistical engine      │  stats_engine.py
   │    period tests, segment   │
   │    movers, correlations,   │
   │    rolling z anomalies     │
   └────────────────────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │ 4. Graph discovery         │  graph_engine.py
   │    geo×category lift,      │
   │    co-purchase affinity    │
   └────────────────────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │ 5. Hypothesis engine       │  hypothesis_engine.py
   │    fuse signals → findings │
   │    + DuckDB SQL evidence   │
   └────────────────────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │ 6. Causal testing          │  causal_engine.py
   │    OLS + controls, PC      │
   │    skeleton, bootstrap CI  │
   └────────────────────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │ 7. Evidence ranking        │  ranking.py
   │    multi-component score   │
   └────────────────────────────┘
                 │
                 ▼
   ┌────────────────────────────┐
   │ 8. LLM decision layer      │  llm_reporter.py
   │    select, order, narrate; │
   │    executive summary       │
   └────────────────────────────┘
                 │
                 ▼
        INVESTIGATION REPORT     report.py (md + self-contained html)
```

### Stage 1 — Loader & canonicalizer (`loader.py`)

Two datasets are used to prove generality:

- **Olist** (Brazilian e-commerce, `olist_*` files, English columns).
- **Retail** (synthetic, Spanish file/column names: `transacciones`,
  `lineas_pedido`, `clientes`, `catalogo`, `categorias`, `tiendas`).

`load_raw` reads every CSV, then `canonicalize_raw` maps arbitrary table stems
and column names onto an internal canonical schema (the Olist names). This is an
*alias table* per role (orders / items / customers / products / sellers /
translations / payments / reviews) plus a date-column fallback. Every later
stage — fact table, DuckDB evidence views, SQL generation — is written against
the canonical names, so a new naming scheme is addressed by the alias table, not
by new analysis code.

The **fact table** is the order-item grain:

- `revenue = price + freight_value` (freight is 0 when absent).
- canceled orders are excluded.
- category names are translated to English when a translation table exists.

`FactSchema` (models.py) centralizes the column names that the analytical stages
depend on (`time_col`, `revenue_col`, `order_col`, `status_col`, `dimensions`,
`numeric_cols`), so the analytical code never touches dataset-specific names.

### Stages 3–4 — statistical & graph discovery

- `quarterly_series` / `monthly_series` aggregate revenue and distinct orders per
  period from the fact table.
- `detect_period_changes` compares consecutive quarters (min absolute delta,
  min distinct days/period to avoid partial-period bias) and `drilldown_change`
  decomposes a change across dimensions (state → city → category).
- `segment_movers` compares the two most recent quarters per segment and keeps
  statistically significant movers.
- `correlation_hits` reports numeric column pairs with |r| ≥ threshold.
- `category_geo_lift` computes a regional index (category share in a state
  relative to national share); `co_occurrence_lift` finds categories bought
  together in the same order significantly more often than chance.
- Monthly outliers use a centered rolling median/MAD **z-score** (z ≥ 2),
  skipping the final, possibly incomplete month and months below a volume guard.

### Stage 5 — Hypothesis engine & evidence

The engine fuses the signals above into `Finding` objects. Every finding ships
with:

- **SQL evidence**: a query run against DuckDB *views* registered from the same
  canonical frames used by the fact table (`DBEV`). It is marked `verified` when
  it reproduces the engine's headline number within tolerance. Because the views
  and the pandas fact table read from the identical in-memory frames, the two
  paths are genuinely independent but cannot diverge accidentally (row sampling
  changes both equally).
- **Python evidence**: a short snippet reproducing the number from the fact
  table, for hand-rechecking.

### Stage 6 — Causal testing (`causal_engine.py`)

Each headline finding is tested as a treatment/control regression:

- Panel = region × quarter × category aggregation.
- `target ~ treatment + controls` via OLS; controls are a fixed set of plausible
  confounders (`n_items`, `avg_price`, `freight_ratio`, review metrics), each
  dropped if null in the panel.
- A **bootstrap CI** on the treatment coefficient (collinearity warnings are
  suppressed; degenerate fits are skipped).
- An optional **PC skeleton** (causal-learn) marks whether treatment and target
  are directly connected in the graphical model.

Output verdict: `supported` / `confounded` / `spurious` / `insufficient_data`,
attached to the finding and shown to the reader. (Small panels — e.g. synthetic
data without reviews — routinely return `insufficient_data`, which is treated as
an honest "cannot tell", not as a false positive.)

### Stage 7 — Ranking (`ranking.py`)

`score_finding` combines six components with fixed weights:

| component | weight |
|---|---|
| significance (best p-value) | 0.25 |
| effect size | 0.20 |
| causal support | 0.25 |
| bootstrap-CI robustness | 0.10 |
| evidence verified | 0.10 |
| coverage | 0.10 |

Ranks are assigned deterministically. A **diversity pass** ensures the top set
is not dominated by one finding kind (graph / segment / association / time
anomaly) by swapping in an unrepresented kind, evicting the lowest-scoring
redundant member of an over-represented kind.

### Stage 8 — LLM decision layer (`llm_reporter.py`)

The LLM (OpenRouter, default `openai/gpt-4o-mini`) is the *decision* layer:

1. Receives **all** ranked candidates (id, title, kind, score, key numbers,
   drill-down, stats, causal verdict, verification flag).
2. Selects which, orders them, writes each narrative (≤90 words), assigns
   confidence ∈ [0,1], and adds caveats.
3. Returns an **executive summary** for the report.

Selected findings are re-ranked in the LLM's stated order and capped at
`top_findings`; any LLM-requests are reconciled with the deterministic scores
(non-selected candidates join in score order). If no API key is set, or the call
fails, the pipeline falls back to the deterministic ranking and **templated**
narratives and summary — so the LLM is an enhancement, never a correctness
dependency (this is what the ground-truth evaluation, `--no-llm`, exercises).

## 3. Data model (key structures, `models.py`)

- `Finding`: id, title, kind, target_metric, headline (key numbers), drill_down
  (cause-location chain), stats (tests), causal (verdict), evidence (SQL +
  python), coverage, narrative, confidence, score, rank, caveats, agents.
- `Evidence`: kind, code, result_head, verified.
- `CausalVerdict`: verdict, method, effect, p_value, ci, details.
- `InvestigationResult`: data_dir, profiles, join_graph, findings, ran_llm,
  duration_s, executive_summary.

## 4. Reproducibility

- All randomness is seeded (`seed=42` default) and confined to `numpy`
  generators — including the synthetic data generators (`scripts/synth_data.py`,
  `scripts/synth_retail.py`), so the evaluation datasets are byte-identical
  across runs.
- The deterministic engine has no randomness at all.
- Evidence SQL is plain text embedded in each finding, so a committee can re-run
  every number by hand.