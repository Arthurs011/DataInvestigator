from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investigator.graph_engine import category_geo_lift, co_occurrence_lift, community_overview
from investigator.models import FactSchema

pytestmark = pytest.mark.unit


def _fact() -> pd.DataFrame:
    # 4 orders, each with 2 categories; strong category geo concentration in RJ
    rows = []
    rng = np.random.default_rng(7)
    for i in range(12):
        rows.append(
            {
                "order_id": f"O{i%3}",
                "customer_state": "RJ",
                "seller_state": "RJ",
                "product_category_name_english": "audio" if i % 2 == 0 else "games",
                "price": 50 + rng.normal(0, 2),
                "freight_value": 10.0,
            }
        )
    fact = pd.DataFrame(rows)
    fact["revenue"] = fact["price"] + fact["freight_value"]
    return fact


def test_co_occurrence_lift_returns_affinity_edges():
    fact = _fact()
    edges = co_occurrence_lift(fact, FactSchema(), min_count=1)
    assert len(edges) >= 1
    for e in edges:
        assert e.lift >= 0
        assert e.n_orders >= 1


def test_category_geo_lift_returns_over_index():
    fact = _fact()
    geo = category_geo_lift(fact, FactSchema(), min_orders=1)
    assert isinstance(geo, list)
    for g in geo:
        assert g.state == "RJ"


def test_community_overview_runs():
    edges = co_occurrence_lift(_fact(), FactSchema(), min_count=1)
    communities, _ = community_overview(edges, top_edges=50)
    assert isinstance(communities, list)