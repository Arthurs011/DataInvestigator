"""Shared data structures for the investigator pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class JoinEdge:
    """A discovered foreign-key-style join between two tables."""

    left_table: str
    left_col: str
    right_table: str
    right_col: str
    overlap_frac: float  # fraction of left values present in right (or global match)


@dataclass
class TableProfile:
    name: str
    n_rows: int
    n_cols: int
    dtypes: dict[str, str]
    missing: dict[str, float]  # col -> fraction missing
    cardinality: dict[str, int]
    numeric_range: dict[str, tuple[float, float]]
    time_cols: list[str] = field(default_factory=list)
    time_coverage: dict[str, tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class Profiles:
    tables: dict[str, TableProfile] = field(default_factory=dict)

    def table(self, name: str) -> TableProfile:
        return self.tables[name]


@dataclass
class DrillDownStep:
    dimension: str
    entity: str
    base_value: float
    comp_value: float
    delta_pct: float
    contribution_pct: float  # share of the headline change this segment explains
    n: int


@dataclass
class StatTest:
    name: str
    statistic: float | None
    p_value: float | None
    effect_size: float | None
    effect_name: str = ""


@dataclass
class CausalVerdict:
    verdict: str  # "supported" | "confounded" | "spurious" | "insufficient_data"
    method: str
    effect: float | None = None
    p_value: float | None = None
    ci: tuple[float, float] | None = None
    details: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    kind: str  # "sql" | "python"
    code: str
    result_head: str = ""  # ascii/plain preview of what it produced
    verified: bool = False


@dataclass
class Finding:
    id: str
    title: str
    kind: str  # "time_anomaly" | "trend" | "concentration" | "association" | "graph" | ...
    target_metric: str
    headline: dict[str, Any] = field(default_factory=dict)  # base/comp/delta numbers
    drill_down: list[DrillDownStep] = field(default_factory=list)
    stats: list[StatTest] = field(default_factory=list)
    causal: CausalVerdict | None = None
    evidence: list[Evidence] = field(default_factory=list)
    coverage: float = 0.0  # fraction of rows involved
    narrative: str = ""
    confidence: float | None = None
    score: float = 0.0
    rank: int = 0
    caveats: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)  # which stages produced this


@dataclass
class InvestigationResult:
    data_dir: str
    profiles: Profiles
    join_graph: list[JoinEdge]
    findings: list[Finding]
    ran_llm: bool
    duration_s: float = 0.0


@dataclass
class FactSchema:
    """Column names in the order-item-grain fact table used by analyses."""

    time_col: str = "order_purchase_timestamp"
    revenue_col: str = "revenue"
    order_col: str = "order_id"
    status_col: str = "order_status"
    dimensions: tuple[str, ...] = (
        "customer_state",
        "customer_city",
        "product_category_name_english",
        "seller_state",
    )
    numeric_cols: tuple[str, ...] = ("price", "freight_value", "revenue")


@dataclass
class Dataset:
    """A collection of loaded tables plus the derived fact table."""

    raw: dict[str, pd.DataFrame]
    fact: pd.DataFrame
    schema: FactSchema = field(default_factory=FactSchema)