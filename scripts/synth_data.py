"""Generate a synthetic e-commerce dataset (Olist-schema) with injectable,
known ground-truth anomalies for evaluating the investigator pipeline.

Running the pipeline on `injected` data must recover exactly the anomalies
defined by `ANOMALIES` (and not find them when `anomalies=False`).

Usage:
    uv run python scripts/synth_data.py --out /tmp/synth --anomalies --orders 700
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Ground-truth anomalies -----------------------------------------------
DROP_Q = "2020Q3"           # A: revenue drops ~55% in this quarter (all states)
GEO_STATE = "TS"            # B: 'furniture' over-indexes in this state
GEO_CAT = "furniture"
GEO_FACTOR = 5.0            # applied to every TS furniture order
SPIKE_MONTH = "2020-06"     # C: a month where revenue is ~4x the trend
SPIKE_FACTOR = 4.0

ANOMALIES = {
    "A_quarterly_revenue_drop": {
        "kind": "time_change",
        "describe": f"revenue drop in {DROP_Q} (price x0.45)",
    },
    "B_state_category_overindex": {
        "kind": "graph",
        "describe": f"'{GEO_CAT}' x{GEO_FACTOR} revenue in state {GEO_STATE}",
    },
    "C_monthly_spike": {
        "kind": "time_anomaly",
        "describe": f"revenue spike in {SPIKE_MONTH} (price x{SPIKE_FACTOR})",
    },
}

_STATES = {
    "SP": 0.30, "RJ": 0.18, "MG": 0.12, "BA": 0.08, "RS": 0.08,
    "PR": 0.08, "SC": 0.05, "TS": 0.05, "PE": 0.06,
}
_CATEGORIES = {
    "electronics": 900.0, "computers": 1200.0, "furniture": 420.0,
    "sports": 200.0, "fashion": 120.0, "home": 150.0, "books": 60.0,
    "toys": 90.0, "health": 80.0, "food": 45.0,
}
_QUARTERS = ["2019Q1", "2019Q2", "2019Q3", "2019Q4",
             "2020Q1", "2020Q2", "2020Q3", "2020Q4"]
_LAST3 = {"2020Q2", "2020Q3", "2020Q4"}


def _quarter_range(q: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    y, n = int(q[:4]), int(q[5])
    return pd.Timestamp(f"{y}-{(n - 1) * 3 + 1:02d}-01"), pd.Timestamp(
        f"{y}-{n * 3:02d}-01"
    )


def build_dataset(
    outdir: str | Path,
    seed: int = 42,
    orders_per_quarter: int = 700,
    n_customers: int = 4000,
    anomalies: bool = False,
) -> Path:
    """Write the 6 CSVs Olist's loader + SQL views need and return the dir."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    state_pool = list(_STATES)
    state_w = np.array(list(_STATES.values()))
    cat_pool = list(_CATEGORIES)
    cat_w = np.array([1.0] * len(cat_pool))
    cat_w = cat_w / cat_w.sum()

    states = rng.choice(state_pool, size=n_customers, p=state_w)
    customers = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n_customers)],
            "customer_state": states,
            "customer_city": [f"city_{i % 40}" for i in range(n_customers)],
        }
    )
    purchases: list[pd.Timestamp] = []
    cids: list[str] = []
    statuses: list[str] = []
    order_i = 0
    for q in _QUARTERS:
        start, end = _quarter_range(q)
        for k in range(orders_per_quarter):
            day = k % 88  # >= 55 distinct days per quarter (period_days check)
            ts = start + pd.Timedelta(days=day, hours=rng.integers(0, 23), minutes=rng.integers(0, 60))
            purchases.append(ts)
            cids.append(f"c{order_i % n_customers}")
            statuses.append("delivered" if rng.random() > 0.025 else "canceled")
            order_i += 1
    orders = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(order_i)],
            "customer_id": cids,
            "order_status": statuses,
            "order_purchase_timestamp": pd.Series(purchases, dtype="datetime64[ns]"),
        }
    )
    cust_state = customers.set_index("customer_id")["customer_state"].to_dict()

    items = []
    sellers = []
    pid = {c: f"p{i}" for i, c in enumerate(cat_pool)}
    sid = 0
    for idx, row in orders.iterrows():
        state = cust_state[row["customer_id"]]
        q = f"{row['order_purchase_timestamp'].year}Q{(row['order_purchase_timestamp'].month - 1) // 3 + 1}"
        month = f"{row['order_purchase_timestamp'].year:04d}-{row['order_purchase_timestamp'].month:02d}"
        n_items = int(rng.choice([1, 1, 2, 2, 3]))
        for it in range(n_items):
            cat = str(rng.choice(cat_pool, p=cat_w))
            price = float(rng.lognormal(np.log(_CATEGORIES[cat]), 0.22))
            if anomalies:
                if q == DROP_Q:
                    price *= 0.45
                if state == GEO_STATE and cat == GEO_CAT:
                    price *= GEO_FACTOR
                if month == SPIKE_MONTH:
                    price *= SPIKE_FACTOR
            freight = float(5.0 + 8.0 * n_items + abs(rng.normal(0, 2.0)))
            items.append(
                {
                    "order_id": row["order_id"],
                    "order_item_id": it + 1,
                    "product_id": pid[cat],
                    "seller_id": f"s{sid}",
                    "price": round(price, 2),
                    "freight_value": round(freight, 2),
                }
            )
            sellers.append({"seller_id": f"s{sid}", "seller_state": state})
            sid += 1

    order_items = pd.DataFrame(items)
    sellers = pd.DataFrame(sellers).drop_duplicates("seller_id")
    products = pd.DataFrame(
        {"product_id": [pid[c] for c in cat_pool], "product_category_name": list(cat_pool)}
    )
    cats = pd.DataFrame(
        {
            "product_category_name": list(cat_pool),
            "product_category_name_english": list(cat_pool),
        }
    )

    orders.to_csv(out / "olist_orders_dataset.csv", index=False)
    order_items.to_csv(out / "olist_order_items_dataset.csv", index=False)
    customers.to_csv(out / "olist_customers_dataset.csv", index=False)
    products.to_csv(out / "olist_products_dataset.csv", index=False)
    cats.to_csv(out / "product_category_name_translation.csv", index=False)
    sellers.to_csv(out / "olist_sellers_dataset.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".eval/dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--orders", type=int, default=700)
    ap.add_argument("--customers", type=int, default=4000)
    flags = ap.add_mutually_exclusive_group()
    flags.add_argument("--anomalies", action="store_true")
    flags.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    build_dataset(
        args.out,
        seed=args.seed,
        orders_per_quarter=args.orders,
        n_customers=args.customers,
        anomalies=args.anomalies,
    )
    print(f"Synth dataset written to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()