"""Graph discovery: market-basket co-purchase and category X geography over-indexing."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .models import FactSchema


@dataclass
class AffinityEdge:
    item_a: str
    item_b: str
    lift: float
    support: float  # P(both) as fraction of orders
    n_orders: int


@dataclass
class GeoOverIndex:
    state: str
    category: str
    lift: float  # category share within state / overall category share
    share_state: float  # fraction of state revenue this category accounts for
    revenue: float
    n_orders: int


def _order_item_sets(fact: pd.DataFrame, schema: FactSchema) -> pd.DataFrame:
    """Map each order to the set of categories it contains (order grain)."""
    sub = fact[~fact["product_category_name_english"].isna()].copy()
    sub["_cat"] = sub["product_category_name_english"].astype(str)
    return sub.groupby(schema.order_col)["_cat"].agg(lambda s: frozenset(s)).dropna()


def co_occurrence_lift(
    fact: pd.DataFrame,
    schema: FactSchema | None = None,
    min_support: float = 0.001,
    min_count: int = 20,
) -> list[AffinityEdge]:
    """Category pairs that co-occur in orders more than independence predicts (lift)."""
    schema = schema or FactSchema()
    sets = _order_item_sets(fact, schema)
    n_orders = len(sets)
    item_count: dict[str, int] = {}
    pair_count: dict[tuple[str, str], int] = {}
    for s in sets:
        items = [c for c in s if c]
        for it in items:
            item_count[it] = item_count.get(it, 0) + 1
        uniq = list(set(items))
        for i in range(len(uniq)):
            for a in range(i + 1, len(uniq)):
                key = tuple(sorted((uniq[i], uniq[a])))
                pair_count[key] = pair_count.get(key, 0) + 1
    out: list[AffinityEdge] = []
    for (a, b), cnt in pair_count.items():
        if cnt < min_count:
            continue
        pab = cnt / n_orders
        if pab < min_support:
            continue
        pa = item_count[a] / n_orders
        pb = item_count[b] / n_orders
        if pa == 0 or pb == 0:
            continue
        lift = pab / (pa * pb)
        out.append(AffinityEdge(item_a=a, item_b=b, lift=float(lift), support=float(pab), n_orders=cnt))
    return sorted(out, key=lambda e: e.lift, reverse=True)


def category_geo_lift(
    fact: pd.DataFrame,
    schema: FactSchema | None = None,
    min_orders: int = 30,
) -> list[GeoOverIndex]:
    """States where a category over-indexes relative to its national share."""
    schema = schema or FactSchema()
    sub = fact.dropna(subset=[schema.revenue_col, "customer_state", "product_category_name_english"]).copy()
    if not len(sub):
        return []
    overall = sub.groupby("product_category_name_english")[schema.revenue_col].sum()
    overall_share = overall / overall.sum()
    by_state = sub.groupby(["customer_state", "product_category_name_english"]).agg(
        revenue=(schema.revenue_col, "sum"),
        orders=(schema.order_col, "nunique"),
    )
    state_total = sub.groupby("customer_state")[schema.revenue_col].sum()
    out: list[GeoOverIndex] = []
    for (state, cat), row in by_state.iterrows():
        if row["orders"] < min_orders:
            continue
        if state_total.get(state, 0) == 0:
            continue
        share_state = row["revenue"] / state_total[state]
        base = float(overall_share.get(cat, np.nan))
        if not base or np.isnan(base):
            continue
        lift = share_state / base
        out.append(
            GeoOverIndex(
                state=str(state),
                category=str(cat),
                lift=float(lift),
                share_state=float(share_state),
                revenue=float(row["revenue"]),
                n_orders=int(row["orders"]),
            )
        )
    return sorted(out, key=lambda e: e.lift, reverse=True)


def community_overview(
    affinity: list[AffinityEdge],
    top_edges: int = 800,
) -> tuple[list[list[str]], dict[str, int]]:
    """Louvain communities over the co-purchase graph (undirected, lift-weighted)."""
    g = nx.Graph()
    for e in affinity[:top_edges]:
        w = max(0.0, e.lift - 1.0)
        if w > 0:
            g.add_edge(e.item_a, e.item_b, weight=w)
    communities = list(nx.algorithms.community.louvain_communities(g, seed=42))
    memberships: dict[str, int] = {}
    for i, comm in enumerate(communities):
        for node in comm:
            memberships[node] = i
    return [sorted(c) for c in communities], memberships