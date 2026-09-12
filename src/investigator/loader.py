"""Load heterogeneous CSVs from a directory and assemble a fact table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .models import Dataset, FactSchema

DATE_COLS = {
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "review_creation_date",
    "review_answer_timestamp",
}


def _read_csv(path: Path, row_limit: int) -> pd.DataFrame:
    usecols = None
    df = pd.read_csv(path)
    if row_limit and len(df) > row_limit:
        df = df.sample(row_limit, random_state=42)
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
    return df


def load_raw(data_dir: str | Path, row_limit: int = 0) -> dict[str, pd.DataFrame]:
    """Load every *.csv in the directory into a name -> DataFrame map."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("*.csv")):
        out[path.stem] = _read_csv(path, row_limit)
    if not out:
        raise ValueError(f"no CSVs found in {data_dir}")
    return out


def build_fact_table(raw: dict[str, pd.DataFrame], schema: FactSchema | None = None) -> pd.DataFrame:
    """Join orders -> order_items -> customers/products/sellers into order-item grain.

    Adds a `revenue` = price + freight_value and translated product category.
    """
    schema = schema or FactSchema()
    required = {
        "olist_orders_dataset": "orders",
        "olist_order_items_dataset": "items",
        "olist_customers_dataset": "customers",
        "olist_products_dataset": "products",
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, alias in required.items():
        table = raw.get(key)
        if table is None:
            table = raw.get(key.removeprefix("olist_"))
        if table is None:
            table = raw.get(key.replace("olist_", "", 1))
        if table is None:
            raise KeyError(f"fact table requires '{key}' but it is missing from the data")
        tables[alias] = table

    orders = tables["orders"]
    items = tables["items"]
    customers = tables["customers"]
    products = tables["products"]

    fact = items.merge(
        orders[["order_id", "customer_id", "order_status", schema.time_col]],
        on="order_id",
        how="left",
    )
    fact = fact.merge(customers[["customer_id", "customer_state", "customer_city"]], on="customer_id", how="left")
    fact = fact.merge(
        products[["product_id", "product_category_name"]],
        on="product_id",
        how="left",
    )

    cat = "product_category_name_translation"
    if cat in raw:
        fact = fact.merge(
            raw[cat][["product_category_name", "product_category_name_english"]],
            on="product_category_name",
            how="left",
        )
    else:
        fact["product_category_name_english"] = fact["product_category_name"]

    sellers = "olist_sellers_dataset"
    if sellers in raw:
        fact = fact.merge(
            raw[sellers][["seller_id", "seller_state"]],
            on="seller_id",
            how="left",
        )
    else:
        fact["seller_state"] = None

    reviews = "olist_order_reviews_dataset"
    if reviews in raw and "review_score" not in fact.columns:
        review_score = (
            raw[reviews].groupby("order_id")["review_score"].first()
        )
        fact = fact.merge(review_score.rename("review_score"), on="order_id", how="left")
    else:
        fact["review_score"] = np.nan

    fact[schema.revenue_col] = fact["price"] + fact["freight_value"]
    if "order_status" in fact.columns:
        fact = fact[fact["order_status"] != "canceled"].copy()
    return fact


def load_dataset(data_dir: str | Path, row_limit: int = 0) -> Dataset:
    raw = load_raw(data_dir, row_limit)
    fact = build_fact_table(raw)
    return Dataset(raw=raw, fact=fact)