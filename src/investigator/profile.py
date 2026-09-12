"""Data understanding: per-table profiles and foreign-key join graph discovery."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from .models import JoinEdge, Profiles, TableProfile

_ID_HINT_SUFFIXES = ("_id", "_code", "_prefix")
_TIME_HINT = {
    "timestamp",
    "date",
    "time",
    "approved",
    "delivered",
    "estimated",
    "limit",
    "created",
    "answer",
    "purchase",
}


def _is_id_col(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(s) for s in _ID_HINT_SUFFIXES) or low.endswith("id")


def _looks_temporal(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in _TIME_HINT)


def infer_type(col: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(col):
        return "bool"
    if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_datetime64_any_dtype(col):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime"
    try:
        pd.to_datetime(col, errors="raise")
        if _looks_temporal(col.name):
            return "datetime"
    except (ValueError, TypeError):
        pass
    return "categorical" if col.nunique() / max(len(col), 1) < 0.5 else "text"


def _profile_table(name: str, df: pd.DataFrame) -> TableProfile:
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    missing = {c: float(df[c].isna().mean()) for c in df.columns}
    cardinality = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    numeric_range: dict[str, tuple[float, float]] = {}
    time_cols: list[str] = []
    time_coverage: dict[str, tuple[object, object]] = {}
    for c in df.columns:
        col = df[c]
        if col.dtype.kind in "ifu" and not pd.api.types.is_bool_dtype(col):
            if col.dropna().size:
                numeric_range[c] = (float(col.min()), float(col.max()))
        if pd.api.types.is_datetime64_any_dtype(col):
            time_cols.append(c)
            if col.dropna().size:
                time_coverage[c] = (col.min(), col.max())
    return TableProfile(
        name=name,
        n_rows=len(df),
        n_cols=len(df.columns),
        dtypes=dtypes,
        missing=missing,
        cardinality=cardinality,
        numeric_range=numeric_range,
        time_cols=time_cols,
        time_coverage=time_coverage,
    )


def profile_tables(raw: dict[str, pd.DataFrame]) -> Profiles:
    return Profiles(tables={name: _profile_table(name, df) for name, df in raw.items()})


def discover_joins(raw: dict[str, pd.DataFrame]) -> list[JoinEdge]:
    """Find plausible foreign-key links between tables by name pattern + value overlap."""
    names = list(raw)
    edges: list[JoinEdge] = []
    seen: set[tuple[str, str]] = set()

    def overlap(a: pd.Series, b: pd.Series) -> float:
        va, vb = set(a.dropna().unique()), set(b.dropna().unique())
        if not va:
            return 0.0
        return len(va & vb) / len(va)

    colmap = {n: set(raw[n].columns) for n in names}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if (left, right) in seen:
                continue
            lcols, rcols = colmap[left], colmap[right]
            common = {c for c in lcols & rcols if _is_id_col(c)}
            # cross-lookup: table-prefixed id (e.g. olist_order_items has owner_table = left stem)
            for c in lcols:
                if _is_id_col(c) and c not in common:
                    sans_prefix = c
                    for stem in (right.replace("olist_", ""), right, right.replace("olist_", "").rstrip("s") + "_id"):
                        if c in rcols and stem == c:
                            common.add(c)
            for col in sorted(common):
                a, b = raw[left][col], raw[right][col]
                ov = overlap(a, b)
                if ov >= 0.05:
                    edges.append(
                        JoinEdge(
                            left_table=left,
                            left_col=col,
                            right_table=right,
                            right_col=col,
                            overlap_frac=round(ov, 3),
                        )
                    )
                    seen.add((left, right))
                    seen.add((right, left))
            # category name relationships (product_category_name <-> translation)
            if {left, right} >= {
                "product_category_name_translation",
                "olist_products_dataset",
            }:
                edges.append(
                    JoinEdge(
                        left_table="olist_products_dataset",
                        left_col="product_category_name",
                        right_table="product_category_name_translation",
                        right_col="product_category_name",
                        overlap_frac=1.0,
                    )
                )
                seen.add(("olist_products_dataset", "product_category_name_translation"))
    return edges


def join_graph_text(edges: list[JoinEdge]) -> str:
    if not edges:
        return "no joins discovered"
    lines = []
    for e in edges:
        lines.append(
            f"{e.left_table}.{e.left_col} = {e.right_table}.{e.right_col} (overlap {e.overlap_frac:.1%})"
        )
    return "\n".join(lines)