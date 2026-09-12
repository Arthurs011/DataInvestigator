"""Ground-truth evaluation as a regression test.

Synthetic data with three known injected anomalies must be recovered by the
pipeline (and must NOT appear when the injections are absent). This makes the
investigator's correctness measurable rather than anecdotal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.synth_data import ANOMALIES, GEO_CAT, GEO_STATE, SPIKE_MONTH, build_dataset
from scripts.evaluate import DROP_Q, match
from investigator.config import Config
from investigator.models import Finding
from investigator.pipeline import run_pipeline

pytestmark = pytest.mark.integration

SEED = 42
ORDERS = 700  # monthly spike guard needs >=200 orders in the spike month
CUSTOMERS = 3500


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = Path(tmp_path_factory.mktemp("eval"))
    clean_dir = build_dataset(out / "clean", seed=SEED, orders_per_quarter=ORDERS, n_customers=CUSTOMERS, anomalies=False)
    inj_dir = build_dataset(out / "injected", seed=SEED, orders_per_quarter=ORDERS, n_customers=CUSTOMERS, anomalies=True)
    cfg = Config(top_findings=10, seed=SEED)
    clean = run_pipeline(clean_dir, cfg, no_llm=True)
    injected = run_pipeline(inj_dir, cfg, no_llm=True)
    return clean, injected


def test_clean_run_has_no_injected_signatures(runs):
    clean, injected = runs
    for key in ANOMALIES:
        assert not any(match(f, key) for f in clean.findings), f"anomaly {key} ghosted on clean data"


def test_all_injected_anomalies_recovered(runs):
    clean, injected = runs
    recovered = {key for key in ANOMALIES if any(match(f, key) for f in injected.findings)}
    assert recovered == set(ANOMALIES), f"missing: {set(ANOMALIES) - recovered}"


def test_drop_anomaly_is_top_ranked(runs):
    _, injected = runs
    hit = next(f for f in injected.findings if match(f, "A_quarterly_revenue_drop"))
    assert hit.rank <= 5
    assert any(e.verified for e in hit.evidence)  # SQL evidence cross-checks the drop


def test_spike_anomaly_detectable(runs):
    _, injected = runs
    hit = next(f for f in injected.findings if match(f, "C_monthly_spike"))
    assert hit.headline["z_score"] >= 2.0