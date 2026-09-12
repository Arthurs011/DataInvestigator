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

# Canonical table stems (Olist names) that the rest of the pipeline keys on.
_CANONICAL_STEM = {
    "orders": "olist_orders_dataset",
    "items": "olist_order_items_dataset",
    "customers": "olist_customers_dataset",
    "products": "olist_products_dataset",
    "translations": "product_category_name_translation",
    "sellers": "olist_sellers_dataset",
    "payments": "olist_order_payments_dataset",
    "reviews": "olist_order_reviews_dataset",
}

# Alternate file stems accepted for each canonical role (normalized, no separators).
_TABLE_ALIASES: dict[str, tuple[str, ...]] = {
    "orders": ("orders", "orderheaders", "transactions", "salesheaders", "purchases", "saleheaders",
               "transacciones", "pedidos", "ventas"),
    "items": ("orderitems", "lineitems", "orderlines", "transactionlines", "salesitems", "purchaselines",
              "lineaspedido", "lineaspedidos", "detallepedido", "articulos"),
    "customers": ("customers", "accounts", "buyers", "clients", "customerdata", "clientes"),
    "products": ("products", "catalog", "productcatalog", "stock", "catalogo", "productos"),
    "translations": ("productcategorynametranslation", "categorytranslation", "categorytranslations",
                     "categories", "categorias", "traducciones"),
    "sellers": ("sellers", "outlets", "stores", "vendors", "suppliers", "tiendas", "vendedores", "comercios"),
    "payments": ("payments", "orderpayments", "pagos"),
    "reviews": ("reviews", "orderreviews", "resenas"),
}

# Alternate column names for each canonical column, per table role.
_COLUMN_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "orders": {
        "order_id": ("order_id", "id", "txn_id", "transaction_id", "sale_id", "purchase_id", "invoice_id"),
        "customer_id": ("customer_id", "account_id", "client_id", "buyer_id", "cust_id", "user_id", "shopper_id"),
        "order_status": ("order_status", "status", "fulfillment_status", "order_state", "shipment_status"),
        "order_purchase_timestamp": (
            "order_purchase_timestamp", "order_date", "purchase_date", "created_at", "created_on",
            "order_datetime", "sale_datetime", "sale_date", "invoice_date", "ordered_on",
            "sales_datetime", "fechapedido", "fecha", "date", "timestamp",
        ),
    },
    "items": {
        "order_id": ("order_id", "id", "txn_id", "transaction_id", "sale_id", "purchase_id", "invoice_id"),
        "order_item_id": ("order_item_id", "line_number", "line_no", "line", "item_number", "item_no", "lineno"),
        "product_id": ("product_id", "product_ref", "product_sku", "sku", "item_id", "upc", "product_code", "part_number"),
        "seller_id": ("seller_id", "store_id", "outlet_id", "vendor_id", "supplier_id", "shop_id", "warehouse_id"),
        "price": ("price", "unit_price", "amount", "total", "line_total", "item_total", "revenue", "sales_amount", "value"),
        "freight_value": ("freight_value", "shipping", "shipping_cost", "freight", "delivery_fee", "shipping_fee", "postage"),
    },
    "customers": {
        "customer_id": ("customer_id", "account_id", "client_id", "buyer_id", "cust_id", "user_id"),
        "customer_state": ("customer_state", "state", "region", "province", "state_code", "geo_region"),
        "customer_city": ("customer_city", "city", "town", "municipality"),
    },
    "products": {
        "product_id": ("product_id", "product_ref", "product_sku", "sku", "item_id", "upc", "product_code"),
        "product_category_name": (
            "product_category_name", "category", "product_category", "category_name", "product_type",
            "department", "product_group", "type",
        ),
    },
    "translations": {
        "product_category_name": ("product_category_name", "category", "native_category", "native_label", "category_name", "department"),
        "product_category_name_english": (
            "product_category_name_english", "category_en", "category_english", "english_category",
            "product_category_english", "en_label",
        ),
    },
    "sellers": {
        "seller_id": ("seller_id", "store_id", "outlet_id", "vendor_id", "supplier_id", "shop_id"),
        "seller_state": ("seller_state", "state", "region", "province", "state_code"),
    },
    "payments": {
        "order_id": ("order_id", "id", "txn_id", "transaction_id"),
        "payment_type": ("payment_type", "type", "payment_method", "method"),
        "payment_value": ("payment_value", "value", "amount", "total"),
    },
    "reviews": {
        "order_id": ("order_id", "id", "txn_id"),
        "review_score": ("review_score", "score", "rating", "stars", "review_rating"),
        "review_creation_date": ("review_creation_date", "created_at", "review_date", "date"),
    },
}


def _norm(name: str) -> str:
    """Lowercase and strip punctuation/whitespace for tolerant matching."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _detect_role(stem: str) -> str | None:
    key = _norm(stem)
    for role, aliases in _TABLE_ALIASES.items():
        if key in aliases:
            return role
    return None


def _build_column_rename(df: pd.DataFrame, role: str) -> dict[str, str]:
    """Map a table's real column names onto the canonical role columns."""
    cols = {_norm(c): c for c in df.columns}
    rename: dict[str, str] = {}

    def find(canonical: str) -> str | None:
        for alias in _COLUMN_ALIASES[role].get(canonical, ()):
            if _norm(alias) in cols:
                return cols[_norm(alias)]
        return None

    for canonical in _COLUMN_ALIASES[role]:
        orig = find(canonical)
        if orig is not None and orig != canonical:
            rename[orig] = canonical

    # fallback: orders need a time column; if none matched, take the first date-like one
    if role == "orders" and "order_purchase_timestamp" not in rename and "order_purchase_timestamp" not in df.columns:
        for c in df.columns:
            if not str(df[c].dtype).startswith("object"):
                continue
            try:
                if pd.to_datetime(df[c].head(200), errors="coerce", format="ISO8601").notna().mean() > 0.9:
                    rename[c] = "order_purchase_timestamp"
                    break
            except Exception:
                continue
    return rename


def canonicalize_raw(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Normalize table stems and column names onto the canonical (Olist) schema.

    The rest of the pipeline — fact-table builder, DuckDB evidence views, SQL
    generation — is written against canonical names, so arbitrary file/column
    naming schemes (e.g. a Spanish retail dataset) become addressable.
    """
    out: dict[str, pd.DataFrame] = {}
    for stem, df in raw.items():
        role = _detect_role(stem)
        if role is None:
            out[stem] = df
            continue
        rename = _build_column_rename(df, role)
        df = df.rename(columns=rename)
        out[_CANONICAL_STEM[role]] = df
    # canonical time columns must be datetime for PeriodIndex / DuckDB usage
    for name, col in (("olist_orders_dataset", "order_purchase_timestamp"),
                      ("olist_order_reviews_dataset", "review_creation_date")):
        if name in out and col in out[name].columns:
            out[name][col] = pd.to_datetime(out[name][col], errors="coerce")
    return out


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
    """Load every *.csv in the directory into a name -> DataFrame map.

    Table stems and column names are canonicalized onto the internal (Olist)
    schema so that heterogeneous datasets with different naming schemes work.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("*.csv")):
        out[path.stem] = _read_csv(path, row_limit)
    if not out:
        raise ValueError(f"no CSVs found in {data_dir}")
    return canonicalize_raw(out)


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

    fact[schema.revenue_col] = fact["price"] + fact.get("freight_value", 0.0).fillna(0.0)
    if "order_status" in fact.columns:
        fact = fact[fact["order_status"] != "canceled"].copy()
    return fact


def load_dataset(data_dir: str | Path, row_limit: int = 0) -> Dataset:
    raw = load_raw(data_dir, row_limit)
    fact = build_fact_table(raw)
    return Dataset(raw=raw, fact=fact)