from __future__ import annotations

import pandas as pd
import pytest

from investigator.loader import build_fact_table
from investigator.models import FactSchema
from investigator.profile import profile_tables

pytestmark = pytest.mark.unit


def _raw() -> dict[str, pd.DataFrame]:
    orders = pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "customer_id": ["c1", "c2", "c3"],
            "order_status": ["delivered", "canceled", "delivered"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2024-01-05", "2024-02-05", "2024-03-05"]
            ),
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "a"],
            "order_item_id": [1, 1, 1, 2],
            "product_id": ["p1", "p2", "p3", "p4"],
            "seller_id": ["s1", "s2", "s3", "s1"],
            "price": [10.0, 5.0, 30.0, 40.0],
            "freight_value": [3.0, 2.0, 1.0, 5.0],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3"],
            "customer_state": ["SP", "RJ", "MG"],
            "customer_city": ["sao paulo", "rio de janeiro", "belo horizonte"],
        }
    )
    products = pd.DataFrame(
        {"product_id": ["p1", "p2", "p3", "p4"], "product_category_name": ["x", "y", "x", "z"]}
    )
    return {
        "olist_orders_dataset": orders,
        "olist_order_items_dataset": items,
        "olist_customers_dataset": customers,
        "olist_products_dataset": products,
    }


def test_build_fact_table_drops_canceled():
    raw = _raw()
    fact = build_fact_table(raw)
    assert set(fact["order_id"]) == {"a", "c"}
    assert len(fact) == 3  # two items from order a + one from c


def test_build_fact_table_computes_revenue():
    fact = build_fact_table(_raw())
    assert (fact["revenue"] == fact["price"] + fact["freight_value"]).all()


def test_build_fact_table_has_dimensions():
    fact = build_fact_table(_raw())
    for col in ("customer_state", "product_category_name_english"):
        assert col in fact.columns


def test_build_fact_table_requires_tables():
    with pytest.raises(KeyError):
        build_fact_table({"olist_orders_dataset": pd.DataFrame()})


def test_profile_tables_profiles_each():
    raw = _raw()
    profiles = profile_tables(raw)
    assert len(profiles.tables) == len(raw)
    p = profiles.table("olist_orders_dataset")
    assert p.n_rows == 3