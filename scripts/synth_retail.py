"""Generate a synthetic *retail* dataset with a different naming scheme.

The Olist dataset uses ``olist_*`` files and English column names. This dataset
uses Spanish file/column names and a different relational wording
(``transacciones`` / ``lineas_pedido`` / ``clientes`` / ``catalogo`` /
``categorias`` / ``tiendas``) to prove the loader canonicalizes arbitrary
schemas onto the internal fact table and DuckDB evidence views.

Injectable anomalies mirror the main evaluation (scripts/synth_data.py):
  A: 2020Q3 price drop  (price x0.5)
  C: 2020-06 price spike (price x4.0)
so the same detectors must fire after schema canonicalization.

Usage:
    uv run python scripts/synth_retail.py --out /tmp/retail --anomalies
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DROP_Q = "2020Q3"
SPIKE_MONTH = "2020-06"
DROP_FACTOR = 0.5
SPIKE_FACTOR = 4.0

_STATES = {
    "MD": 0.05, "VD": 0.04, "AL": 0.03, "BA": 0.03, "GR": 0.03, "GT": 0.02,
}
# six states summing to 0.20 shown above would break rng.choice; rebalance to sum 1:
_STATES = {k: v / sum(_STATES.values()) for k, v in _STATES.items()}
_CATEGORIES = {
    "electronica": 900.0, "informatica": 1200.0, "muebles": 420.0,
    "deportes": 200.0, "moda": 120.0, "hogar": 150.0, "libros": 60.0,
    "juguetes": 90.0, "salud": 80.0, "alimentacion": 45.0,
}
_QUARTERS = ["2019Q1", "2019Q2", "2019Q3", "2019Q4",
             "2020Q1", "2020Q2", "2020Q3", "2020Q4"]


def _quarter_range(q: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    y, n = int(q[:4]), int(q[5])
    return pd.Timestamp(f"{y}-{(n - 1) * 3 + 1:02d}-01"), pd.Timestamp(f"{y}-{n * 3:02d}-01")


def build_retail(
    outdir: str | Path,
    seed: int = 42,
    orders_per_quarter: int = 700,
    n_customers: int = 3500,
    anomalies: bool = False,
) -> Path:
    """Write the 6 CSVs using the retail (Spanish) naming scheme."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    state_pool = list(_STATES)
    state_w = np.array(list(_STATES.values()))
    cat_pool = list(_CATEGORIES)

    states = rng.choice(state_pool, size=n_customers, p=state_w)
    customers = pd.DataFrame(
        {
            "account_id": [f"acc{i}" for i in range(n_customers)],
            "province": states,
            "city": [f"city_{i % 40}" for i in range(n_customers)],
        }
    )
    purchases, cids, statuses = [], [], []
    order_i = 0
    for q in _QUARTERS:
        start, end = _quarter_range(q)
        for k in range(orders_per_quarter):
            day = k % 88
            ts = start + pd.Timedelta(days=day, hours=rng.integers(0, 23), minutes=rng.integers(0, 60))
            purchases.append(ts)
            cids.append(f"acc{order_i % n_customers}")
            statuses.append("entregado" if rng.random() > 0.025 else "cancelado")
            order_i += 1
    orders = pd.DataFrame(
        {
            "txn_id": [f"txn{i}" for i in range(order_i)],
            "account_id": cids,
            "fulfillment_status": statuses,
            "sales_datetime": pd.Series(purchases, dtype="datetime64[ns]"),
        }
    )
    cust_state = customers.set_index("account_id")["province"].to_dict()

    items, sellers = [], []
    pid = {c: f"sku{i:03d}" for i, c in enumerate(cat_pool)}
    sid = 0
    for idx, row in orders.iterrows():
        state = cust_state[row["account_id"]]
        q = f"{row['sales_datetime'].year}Q{(row['sales_datetime'].month - 1) // 3 + 1}"
        month = f"{row['sales_datetime'].year:04d}-{row['sales_datetime'].month:02d}"
        n_items = int(rng.choice([1, 1, 2, 2, 3]))
        for it in range(n_items):
            cat = cat_pool[it % len(cat_pool)]
            price = float(rng.lognormal(np.log(_CATEGORIES[cat]), 0.22))
            if anomalies:
                if q == DROP_Q:
                    price *= DROP_FACTOR
                if month == SPIKE_MONTH:
                    price *= SPIKE_FACTOR
            freight = float(5.0 + 8.0 * n_items + abs(rng.normal(0, 2.0)))
            items.append(
                {
                    "txn_id": row["txn_id"],
                    "line_number": it + 1,
                    "product_sku": pid[cat],
                    "outlet_id": f"shop{sid}",
                    "amount": round(price, 2),
                    "shipping": round(freight, 2),
                }
            )
            sellers.append({"outlet_id": f"shop{sid}", "province": state})
            sid += 1

    order_items = pd.DataFrame(items)
    sellers = pd.DataFrame(sellers).drop_duplicates("outlet_id")
    products = pd.DataFrame(
        {"product_sku": [pid[c] for c in cat_pool], "department": list(cat_pool)}
    )
    cats = pd.DataFrame(
        {
            "department": list(cat_pool),
            "category_en": [f"en-{c}" for c in cat_pool],
        }
    )

    orders.to_csv(out / "transacciones.csv", index=False)
    order_items.to_csv(out / "lineas_pedido.csv", index=False)
    customers.to_csv(out / "clientes.csv", index=False)
    products.to_csv(out / "catalogo.csv", index=False)
    cats.to_csv(out / "categorias.csv", index=False)
    sellers.to_csv(out / "tiendas.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".eval/retail")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--orders", type=int, default=700)
    ap.add_argument("--customers", type=int, default=3500)
    flags = ap.add_mutually_exclusive_group()
    flags.add_argument("--anomalies", action="store_true")
    flags.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    build_retail(
        args.out,
        seed=args.seed,
        orders_per_quarter=args.orders,
        n_customers=args.customers,
        anomalies=args.anomalies,
    )
    print(f"Retail dataset written to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()