"""Hypothesis engine: fuse statistical + graph signals into candidate findings
with reproducible DuckDB SQL evidence and pandas Python evidence."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import Config
from .graph_engine import co_occurrence_lift, category_geo_lift
from .models import (
    Dataset,
    DrillDownStep,
    Evidence,
    Finding,
    Profiles,
    StatTest,
)
from .stats_engine import (
    correlation_hits,
    daily_period_tests,
    detect_period_changes,
    drilldown_change,
    monthly_series,
    run_tests,
    segment_movers,
)

_CANCEL_FILTER = "AND o.order_status <> 'canceled'"


def _fmt_pct(x: float) -> str:
    return f"{x:+.1%}" if abs(x) < 10 else f"{x:+.2f}x"


def _period_parts(period: str) -> tuple[int, int]:
    """'2018Q3' -> (2018, 3)."""
    m = re.match(r"(\d{4})Q(\d)", str(period))
    if not m:
        return (0, 0)
    return int(m.group(1)), int(m.group(2))


class DBEV:
    """DuckDB handle exposing clean SQL views over the loaded canonical tables.

    Accepts either a directory (CSVs are loaded + canonicalized) or an already
    loaded ``raw`` map of canonical tables, so evidence SQL always runs against
    exactly the same data as the pandas fact table (even under row sampling).
    """

    _VIEW_COLS: dict[str, tuple[str, tuple[str, ...]]] = {
        "orders": ("olist_orders_dataset", ("order_id", "customer_id", "order_status", "order_purchase_timestamp")),
        "order_items": ("olist_order_items_dataset", ("order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value")),
        "customers": ("olist_customers_dataset", ("customer_id", "customer_state", "customer_city")),
        "products": ("olist_products_dataset", ("product_id", "product_category_name")),
        "categories": ("product_category_name_translation", ("product_category_name", "product_category_name_english")),
        "payments": ("olist_order_payments_dataset", ("order_id", "payment_type", "payment_value")),
        "reviews": ("olist_order_reviews_dataset", ("order_id", "review_score", "review_creation_date")),
    }

    def __init__(self, data_dir_or_raw: str | Path | dict[str, pd.DataFrame]):
        self.con = duckdb.connect()
        if isinstance(data_dir_or_raw, (str, Path)):
            from .loader import load_raw

            raw = load_raw(data_dir_or_raw)
        else:
            raw = data_dir_or_raw
        self._register(raw)

    def _register(self, raw: dict[str, pd.DataFrame]) -> None:
        for view, (table, cols) in self._VIEW_COLS.items():
            df = raw.get(table)
            if df is None:
                continue
            sub = df[[c for c in cols if c in df.columns]].copy()
            if view == "order_items" and "freight_value" not in sub.columns:
                sub["freight_value"] = 0.0
            self.con.register(view, sub)

    def exec(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def close(self) -> None:
        self.con.close()


class HypothesisEngine:
    def __init__(
        self,
        dataset: Dataset,
        profiles: Profiles,
        data_dir: str | Path | dict[str, pd.DataFrame] | None = None,
        config: Config | None = None,
    ):
        self.dataset = dataset
        self.profiles = profiles
        self.schema = dataset.schema
        self.fact = dataset.fact
        self.config = config or Config()
        self.db = DBEV(data_dir if data_dir is not None else dataset.raw)
        self._seq = 0

    def _id(self, kind: str) -> str:
        self._seq += 1
        return f"{kind}-{self._seq:02d}"

    def _evidence(
        self,
        sql: str,
        expected: float | None = None,
        tol: float = 0.02,
        expected_col: str | None = None,
    ) -> list[Evidence]:
        python_code = (
            "fact = pd.read_csv(...)  # joined order-item grain\n"
            "mask = (fact.order_purchase_timestamp < t1) & (fact.order_purchase_timestamp >= t0)\n"
            f"result = fact.loc[mask, '{self.schema.revenue_col}'].sum()"
        )
        verified = False
        preview = ""
        try:
            df = self.db.exec(sql)
            preview = _preview(df)
            if expected is not None and len(df):
                if expected_col and expected_col in df.columns:
                    actual = float(df.iloc[0][expected_col])
                else:
                    actual = float(df.iloc[0, 0])
                if expected != 0:
                    verified = abs(actual - expected) / abs(expected) <= tol
        except Exception as e:  # duckdb parse errors shouldn't kill the pipeline
            preview = f"# SQL failed: {e}"
        return [
            Evidence(kind="sql", code=sql, result_head=preview, verified=verified),
            Evidence(kind="python", code=python_code),
        ]

    def run(self) -> list[Finding]:
        findings: list[Finding] = []
        findings += self._time_change_findings()
        findings += self._segment_mover_findings()
        findings += self._correlation_findings()
        findings += self._graph_findings()
        return findings

    # ------------------------------------------------------------------ time
    def _time_change_findings(self) -> list[Finding]:
        fact = self.fact
        schema = self.schema
        changes = detect_period_changes(fact, schema=schema, min_abs_delta=0.15, min_rows=150)
        out: list[Finding] = []
        seen: set[str] = set()
        for ch in changes:
            key = (ch.period, ch.prev_period, ch.metric)
            if key in seen:
                continue
            seen.add(key)
            y, q = _period_parts(ch.period)
            py, pq = _period_parts(ch.prev_period)
            if not (y and q and py and pq):
                continue
            # skip early ramp (startup growth) and periods that look partially observed
            periods = sorted({str(p) for p in pd.PeriodIndex(fact[schema.time_col], freq="Q")})
            if len(periods) < 3:
                break
            if str(ch.prev_period) in periods[:2]:
                continue

            def period_days(p: str) -> int:
                sel = fact[pd.PeriodIndex(fact[schema.time_col], freq="Q").astype(str) == p]
                return int(sel[schema.time_col].dt.date.nunique())

            if period_days(str(ch.period)) < 55 or period_days(str(ch.prev_period)) < 55:
                continue  # incomplete period -> comparison is biased
            steps, _n = drilldown_change(fact, ch.period, ch.prev_period, ch.metric, schema=schema)
            if not steps:
                continue
            # significance: daily volume/revenue across the two periods
            tests = daily_period_tests(fact, ch.period, ch.prev_period, schema=schema)
            direction = "increase" if ch.delta_pct > 0 else "decrease"
            title = (
                f"{ch.metric.title()} {direction} of {abs(ch.delta_pct):.1%} "
                f"in {ch.period} vs {ch.prev_period}"
            )
            sql = _headline_sql(ch, metric=ch.metric)
            expected = ch.comp_value
            evidence = self._evidence(sql, expected=expected, expected_col="comp_value")
            top = steps[0]
            coverage = (ch.n_base + ch.n_comp) / max(len(fact), 1)
            finding = Finding(
                id=self._id("time"),
                title=title,
                kind="time_change",
                target_metric=ch.metric,
                headline={
                    "period": ch.period,
                    "prev_period": ch.prev_period,
                    "base_value": ch.base_value,
                    "comp_value": ch.comp_value,
                    "delta_pct": ch.delta_pct,
                },
                drill_down=steps,
                stats=tests,
                evidence=evidence,
                coverage=coverage,
                agents=["stats_engine", "hypothesis_engine"],
            )
            if top.contribution_pct >= 0.15:
                finding.caveats.append(
                    f"{top.entity} ({top.dimension}) alone accounts for "
                    f"{top.contribution_pct:.0%} of the change"
                )
            out.append(finding)

        # monthly anomalies (spikes/dips vs rolling median)
        m = monthly_series(fact, schema=schema)
        last_complete = m.index[-1] if len(m) else None
        med = m["revenue"].rolling(6, center=True, min_periods=3).median()
        mad = (m["revenue"] - med).abs().rolling(6, center=True, min_periods=3).median()
        z = (m["revenue"] - med) / mad.replace(0, np.nan)
        flags = z[z.abs() >= 2.0]
        for month in flags.index:
            if str(month) == str(last_complete):
                continue  # partial final month
            if m.loc[month, "orders"] < 200:
                continue
            val, zs = float(m.loc[month, "revenue"]), float(flags[month])
            if med[month] == 0:
                continue
            sql = _monthly_sql(month)
            evidence = self._evidence(sql, expected=None)
            direction = "spike" if zs > 0 else "dip"
            out.append(
                Finding(
                    id=self._id("anomaly"),
                    title=f"Monthly revenue {direction} in {month} (z={zs:+.1f})",
                    kind="time_anomaly",
                    target_metric="revenue",
                    headline={"period": month, "value": val, "z_score": zs},
                    stats=[
                        StatTest(
                            name="rolling median/MAD z",
                            statistic=zs,
                            p_value=None,
                            effect_size=float(val / med[month] - 1),
                            effect_name="excess-over-median (x)",
                        )
                    ],
                    evidence=evidence,
                    coverage=float(m.loc[month, "orders"]) / max(len(fact), 1),
                    agents=["stats_engine", "hypothesis_engine"],
                )
            )
        return out

    # ------------------------------------------------------------- segments
    def _segment_mover_findings(self) -> list[Finding]:
        fact = self.fact
        schema = self.schema
        # compare the two most recent complete quarters
        periods = sorted({str(p) for p in pd.PeriodIndex(fact[schema.time_col], freq="Q")})
        if len(periods) < 2:
            return []
        base_p, comp_p = periods[-2], periods[-1]
        out: list[Finding] = []
        for dim in ("product_category_name_english", "customer_state"):
            sm = segment_movers(fact, dim, base_p, comp_p, schema=schema)
            if len(sm) == 0:
                continue
            sm = sm[sm["significant"]]
            # keep the most extreme two movers per dimension
            shell = pd.concat([sm.head(2), sm.tail(2)]).drop_duplicates(subset=[dim]) if len(sm) >= 4 else sm
            for _, row in shell.iterrows():
                cat = row[dim]
                delta = float(row["delta_pct"])
                rev = float(row["comp_revenue"])
                if abs(delta) < 0.25:
                    continue
                direction = "grew" if delta > 0 else "fell"
                title = f"Segment {cat} {direction} {abs(delta):.1%} in {comp_p} vs {base_p}"
                sql = _segment_sql(dim, str(cat), base_p, comp_p)
                evidence = self._evidence(sql, expected=None)
                out.append(
                    Finding(
                        id=self._id("segment"),
                        title=title,
                        kind="segment_mover",
                        target_metric="revenue",
                        headline={
                            "dimension": dim,
                            "entity": cat,
                            "period": comp_p,
                            "prev_period": base_p,
                            "base_value": float(row["base_revenue"]),
                            "comp_value": rev,
                            "delta_pct": delta,
                        },
                        drill_down=[
                            DrillDownStep(
                                dimension=dim,
                                entity=str(cat),
                                base_value=float(row["base_revenue"]),
                                comp_value=rev,
                                delta_pct=delta,
                                contribution_pct=float("nan"),
                                n=int(row["comp_orders"]),
                            )
                        ],
                        stats=[
                            StatTest(
                                name="min p (Welch/Mann-Whitney)",
                                statistic=None,
                                p_value=float(row["min_p"]),
                                effect_size=None,
                            )
                        ],
                        evidence=evidence,
                        coverage=float(row["base_orders"] + row["comp_orders"]) / max(len(fact), 1),
                        agents=["stats_engine", "hypothesis_engine"],
                    )
                )
        return out

    # ------------------------------------------------------- correlations
    def _correlation_findings(self) -> list[Finding]:
        hits = correlation_hits(self.fact, schema=self.schema, min_abs=0.4)
        out: list[Finding] = []
        for a, b, r, p in hits:
            if a == self.schema.revenue_col or b == self.schema.revenue_col:
                pass
            title = f"Correlation: {a} vs {b} (r={r:+.2f})"
            if b == self.schema.revenue_col or a == self.schema.revenue_col:
                expr = f"corr({self.schema.numeric_cols[0]}, {self.schema.numeric_cols[0]} + {self.schema.numeric_cols[1]})"
                sql = (
                    f"SELECT {expr} AS r FROM order_items "
                    f"WHERE {self.schema.numeric_cols[0]} IS NOT NULL AND {self.schema.numeric_cols[1]} IS NOT NULL"
                )
            else:
                sql = f"SELECT corr({a}, {b}) AS r FROM order_items WHERE {a} IS NOT NULL AND {b} IS NOT NULL"
            evidence = self._evidence(sql, expected=float(r), expected_col="r", tol=0.15)
            out.append(
                Finding(
                    id=self._id("corr"),
                    title=title,
                    kind="association",
                    target_metric="pairwise",
                    headline={"a": a, "b": b, "r": r, "p": p},
                    stats=[
                        StatTest(
                            name="Spearman/Pearson",
                            statistic=r,
                            p_value=p,
                            effect_size=r,
                            effect_name="|r|",
                        )
                    ],
                    evidence=evidence,
                    coverage=1.0,
                    agents=["stats_engine", "hypothesis_engine"],
                )
            )
        return out

    # ------------------------------------------------------------ graph
    def _graph_findings(self) -> list[Finding]:
        out: list[Finding] = []
        geo = category_geo_lift(self.fact, schema=self.schema)[:6]
        for e in geo:
            if e.lift < 1.4:
                continue
            title = f"{e.category} over-indexes in {e.state} (lift {e.lift:.2f})"
            sql = (
                "SELECT s.customer_state, p.product_category_name_english, "
                "SUM(i.price + i.freight_value) AS revenue "
                "FROM order_items i JOIN orders o USING(order_id) "
                "JOIN customers c USING(customer_id) JOIN products pr USING(product_id) "
                "JOIN categories p USING(product_category_name) "
                f'WHERE c.customer_state = \'{e.state.replace("'", "''")}\' GROUP BY 1, 2 '
                "ORDER BY 3 DESC"
            )
            evidence = self._evidence(sql, expected=None)
            out.append(
                Finding(
                    id=self._id("graph"),
                    title=title,
                    kind="graph",
                    target_metric="revenue_share",
                    headline={
                        "state": e.state,
                        "category": e.category,
                        "lift": e.lift,
                        "share_state": e.share_state,
                        "revenue": e.revenue,
                    },
                    stats=[
                        StatTest(
                            name="regional index (lift)",
                            statistic=e.lift,
                            p_value=None,
                            effect_size=e.lift,
                            effect_name="lift",
                        )
                    ],
                    evidence=evidence,
                    coverage=float(e.n_orders) / max(len(self.fact), 1),
                    agents=["graph_engine", "hypothesis_engine"],
                )
            )
        aff = co_occurrence_lift(self.fact, schema=self.schema, min_count=10)[:5]
        for e in aff:
            if e.lift < 3.0:
                continue
            title = f"Co-purchase affinity: {e.item_a} + {e.item_b} (lift {e.lift:.1f})"
            sql = (
                "SELECT count(*) AS n FROM (\n"
                "  SELECT order_id FROM order_items JOIN orders USING(order_id) "
                "    JOIN products USING(product_id) JOIN categories USING(product_category_name)\n"
                f"  WHERE product_category_name_english = '{e.item_a.replace("'", "''")}' AND order_status <> 'canceled'\n"
                "  INTERSECT\n"
                "  SELECT order_id FROM order_items JOIN orders USING(order_id) "
                "    JOIN products USING(product_id) JOIN categories USING(product_category_name)\n"
                f"  WHERE product_category_name_english = '{e.item_b.replace("'", "''")}' AND order_status <> 'canceled'\n"
                ") t"
            )
            evidence = self._evidence(sql, expected=None)
            out.append(
                Finding(
                    id=self._id("affinity"),
                    title=title,
                    kind="graph",
                    target_metric="co_purchase",
                    headline={"a": e.item_a, "b": e.item_b, "lift": e.lift, "n_orders": e.n_orders},
                    stats=[
                        StatTest(
                            name="lift",
                            statistic=e.lift,
                            p_value=None,
                            effect_size=e.lift,
                            effect_name="lift",
                        )
                    ],
                    evidence=evidence,
                    coverage=float(e.n_orders) / max(len(self.fact), 1),
                    agents=["graph_engine", "hypothesis_engine"],
                )
            )
        return out


# ---------------------------------------------------------------- SQL builds
def _headline_sql(ch, metric: str) -> str:
    y, q = _period_parts(ch.period)
    py, pq = _period_parts(ch.prev_period)
    if metric == "orders":
        b_expr = "(SELECT COUNT(DISTINCT order_id) FROM base)"
        c_expr = "(SELECT COUNT(DISTINCT order_id) FROM comp)"
    else:
        b_expr = "(SELECT SUM(revenue) FROM base)"
        c_expr = "(SELECT SUM(revenue) FROM comp)"
    return (
        "WITH base AS (\n"
        "  SELECT i.price + i.freight_value AS revenue, o.order_id\n"
        "  FROM order_items i\n"
        "  JOIN orders o USING (order_id)\n"
        f"  WHERE YEAR(o.order_purchase_timestamp) = {py} AND QUARTER(o.order_purchase_timestamp) = {pq} {_CANCEL_FILTER}\n"
        "),\ncomp AS (\n"
        "  SELECT i.price + i.freight_value AS revenue, o.order_id\n"
        "  FROM order_items i\n"
        "  JOIN orders o USING (order_id)\n"
        f"  WHERE YEAR(o.order_purchase_timestamp) = {y} AND QUARTER(o.order_purchase_timestamp) = {q} {_CANCEL_FILTER}\n"
        ")\nSELECT\n"
        f"  {b_expr} AS base_value,\n"
        f"  {c_expr} AS comp_value,\n"
        f"  CASE WHEN {b_expr} = 0 THEN NULL ELSE ({c_expr} - {b_expr}) / {b_expr} END AS delta_pct,\n"
        "  (SELECT COUNT(DISTINCT order_id) FROM base) AS base_orders,\n"
        "  (SELECT COUNT(DISTINCT order_id) FROM comp) AS comp_orders"
    )


def _monthly_sql(month: str) -> str:
    y = int(month[:4])
    m = int(month[5:7])
    return (
        "SELECT YEAR(o.order_purchase_timestamp) AS y, MONTH(o.order_purchase_timestamp) AS m, "
        "SUM(i.price + i.freight_value) AS revenue, COUNT(DISTINCT o.order_id) AS orders\n"
        "FROM order_items i JOIN orders o USING (order_id)\n"
        f"WHERE YEAR(o.order_purchase_timestamp) = {y} AND MONTH(o.order_purchase_timestamp) = {m} {_CANCEL_FILTER}\n"
        "GROUP BY 1, 2"
    )


def _segment_sql(dim: str, entity: str, base_p: str, comp_p: str) -> str:
    by, bq = _period_parts(base_p)
    cy, cq = _period_parts(comp_p)
    join_sql = (
        "SELECT i.price + i.freight_value AS revenue, o.order_purchase_timestamp AS t\n"
        "FROM order_items i JOIN orders o USING(order_id)\n"
        "JOIN customers c USING(customer_id) JOIN products pr USING(product_id)\n"
        "JOIN categories p USING(product_category_name)\n"
        "WHERE o.order_status <> 'canceled'"
    )
    return (
        f"WITH base AS (\n  {join_sql}\n"
        f"  AND YEAR(t)={by} AND QUARTER(t)={bq} AND {dim} = '{entity}'\n"
        f"),\ncomp AS (\n  {join_sql} AND YEAR(t)={cy} AND QUARTER(t)={cq} AND {dim} = '{entity}'\n)\n"
        "SELECT\n"
        "  (SELECT SUM(revenue) FROM base) AS base_value,\n"
        "  (SELECT SUM(revenue) FROM comp) AS comp_value"
    )


def _preview(df: pd.DataFrame, n: int = 5) -> str:
    if df is None or len(df) == 0:
        return "(no rows)"
    head = df.head(n).to_string(index=False)
    if len(df) > n:
        head += f"\n... {len(df) - n} more rows"
    return head