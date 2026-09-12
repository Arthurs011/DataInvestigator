"""Schema-agnostic loader proof: a Spanish-named retail dataset.

The loader must canonicalize arbitrary file/column names onto its internal
schema, then the Olist-written detectors (quarterly change, monthly spike,
DuckDB evidence) must fire with zero code changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.synth_retail import DROP_Q, SPIKE_MONTH, build_retail
from investigator.config import Config
from investigator.loader import load_dataset, load_raw
from investigator.pipeline import run_pipeline

pytestmark = pytest.mark.integration

SEED = 42
ORDERS = 700
CUSTOMERS = 3500


@pytest.fixture(scope="module")
def retail_dir(tmp_path_factory) -> Path:
    return build_retail(
        Path(tmp_path_factory.mktemp("retail")),
        seed=SEED,
        orders_per_quarter=ORDERS,
        n_customers=CUSTOMERS,
        anomalies=True,
    )


def test_load_raw_canonicalizes_spanish_schema(retail_dir):
    raw = load_raw(retail_dir)
    assert "olist_orders_dataset" in raw
    assert "olist_order_items_dataset" in raw
    assert "olist_customers_dataset" in raw
    assert "olist_products_dataset" in raw
    orders = raw["olist_orders_dataset"]
    assert {"order_id", "customer_id", "order_status", "order_purchase_timestamp"} <= set(orders.columns)


def test_fact_table_builds_and_has_revenue(retail_dir):
    ds = load_dataset(retail_dir)
    assert ds.fact["revenue"].notna().all()
    assert ds.fact["customer_state"].notna().all()
    assert len(ds.fact) > 0


def test_pipeline_recovers_anomalies_on_retail(retail_dir):
    result = run_pipeline(retail_dir, Config(top_findings=10, seed=SEED), no_llm=True)
    assert len(result.findings) >= 1

    changes = [f for f in result.findings if f.kind == "time_change" and f.headline.get("period") == DROP_Q]
    spikes = [f for f in result.findings if f.kind == "time_anomaly" and f.headline.get("period") == SPIKE_MONTH]
    assert changes, "quarterly 2020Q3 drop was not detected on the retail schema"
    assert spikes, "monthly 2020-06 spike was not detected on the retail schema"
    assert spikes[0].headline.get("z_score", 0) >= 2.0

    assert any(
        f.evidence and any(e.verified for e in f.evidence) for f in result.findings
    ), "no finding carried SQL evidence that cross-verified against the engine numbers"