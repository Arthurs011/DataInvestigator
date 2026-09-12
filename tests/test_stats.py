from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investigator.models import FactSchema
from investigator.stats_engine import (
    correlation_hits,
    daily_period_tests,
    detect_period_changes,
    drilldown_change,
    monthly_series,
    quarterly_series,
    run_tests,
)

pytestmark = pytest.mark.unit

SCHEMA = FactSchema()
_RNG = np.random.default_rng(1234)

_QUARTER_FEATURES = {
    "2024-01-01": ("Q1", 40.0),
    "2024-04-01": ("Q2", 80.0),
    "2024-07-01": ("Q3", 55.0),
    "2024-10-01": ("Q4", 60.0),
}
_STATES = ["SP", "RJ", "MG"]


def _build_fact(rows_per_quarter: int = 90) -> pd.DataFrame:
    """Every state present in every quarter, price jumps in Q2 -> clean deltas."""
    rows: list[dict] = []
    i = 0
    for start, (_q, base_price) in _QUARTER_FEATURES.items():
        for k in range(rows_per_quarter):
            state = _STATES[k % 3]
            rows.append(
                {
                    "order_id": f"O{i}",
                    "order_purchase_timestamp": pd.Timestamp(start) + pd.Timedelta(days=k),
                    "price": base_price + _RNG.normal(0, 3),
                    "freight_value": 8.0 + _RNG.normal(0, 2),
                    "order_status": "delivered",
                    "customer_state": state,
                    "customer_city": state.lower(),
                    "product_category_name_english": "books" if k % 2 == 0 else "electronics",
                    "seller_state": state,
                }
            )
            i += 1
    fact = pd.DataFrame(rows)
    fact["revenue"] = fact["price"] + fact["freight_value"]
    return fact


@pytest.fixture(scope="module")
def fact():
    return _build_fact()


# ---------- quarterly_series -----------------------------------------------
def test_quarterly_series_has_correct_rows(fact):
    qs = quarterly_series(fact, SCHEMA)
    assert len(qs) == 4
    assert {"revenue", "orders"} <= set(qs.columns)


# ---------- monthly_series -------------------------------------------------
def test_monthly_series_has_full_year(fact):
    ms = monthly_series(fact, SCHEMA)
    assert len(ms) == 12


# ---------- detect_period_changes ------------------------------------------
def test_detect_period_changes_finds_q2_revenue_jump(fact):
    changes = detect_period_changes(fact, SCHEMA, min_abs_delta=0.15, min_rows=50)
    rev = [c for c in changes if c.metric == "revenue"]
    assert rev
    q2 = next((c for c in rev if c.period == "2024Q2" and c.prev_period == "2024Q1"), None)
    assert q2 is not None
    assert q2.delta_pct > 0.5  # roughly doubling


def test_detect_period_changes_orders_flat(fact):
    changes = detect_period_changes(fact, SCHEMA, min_abs_delta=0.15, min_rows=50)
    assert not any(c.metric == "orders" for c in changes)


# ---------- drilldown_change -----------------------------------------------
def test_drilldown_change_returns_steps(fact):
    steps, _ = drilldown_change(fact, "2024Q2", "2024Q1", "revenue", schema=SCHEMA)
    assert len(steps) >= 1
    assert steps[0].dimension in set(SCHEMA.dimensions)


def test_drilldown_change_contributions_sum_sensibly(fact):
    steps, _ = drilldown_change(fact, "2024Q2", "2024Q1", "revenue", schema=SCHEMA)
    total = sum(s.contribution_pct for s in steps)
    assert 0.0 < total <= 1.1  # top contributing states explain most of the change


# ---------- run_tests / daily_period_tests ---------------------------------
def test_run_tests_battery(fact):
    base = fact[fact.order_purchase_timestamp < "2024-04-01"]["revenue"]
    compv = fact[
        (fact.order_purchase_timestamp >= "2024-04-01") & (fact.order_purchase_timestamp < "2024-07-01")
    ]["revenue"]
    tests, ci = run_tests(base, compv)
    names = {t.name for t in tests}
    assert "Welch t-test" in names
    assert "Mann-Whitney U" in names
    assert ci is not None


def test_daily_period_tests_returns_list(fact):
    dpt = daily_period_tests(fact, "2024Q2", "2024Q1", schema=SCHEMA)
    assert len(dpt) >= 2


# ---------- correlation_hits -----------------------------------------------
def test_correlation_hits_detects_price_revenue(fact):
    hits = correlation_hits(fact, SCHEMA, min_abs=0.5)
    assert any(a == "price" and b == "revenue" for a, b, _r, _p in hits)