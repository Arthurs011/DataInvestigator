"""Statistical investigation: time trends, period changes, segment drill-down and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from .models import DrillDownStep, FactSchema, StatTest

MIN_ALPHA = 0.05


@dataclass
class PeriodChange:
    period: object
    prev_period: object
    metric: str
    base_value: float
    comp_value: float
    delta_pct: float
    n_base: int
    n_comp: int


def quarterly_series(fact: pd.DataFrame, schema: FactSchema | None = None) -> pd.DataFrame:
    schema = schema or FactSchema()
    s = fact.copy()
    s["_period"] = pd.PeriodIndex(s[schema.time_col], freq="Q").astype(str)
    g = s.groupby("_period").agg(
        revenue=(schema.revenue_col, "sum"),
        orders=(schema.order_col, "nunique"),
        items=(schema.order_col, "size"),
    )
    g["avg_order_value"] = g["revenue"] / g["orders"]
    g["items_per_order"] = g["items"] / g["orders"]
    return g


def monthly_series(fact: pd.DataFrame, schema: FactSchema | None = None) -> pd.DataFrame:
    schema = schema or FactSchema()
    s = fact.copy()
    s["_period"] = pd.PeriodIndex(s[schema.time_col], freq="M").astype(str)
    return (
        s.groupby("_period")
        .agg(
            revenue=(schema.revenue_col, "sum"),
            orders=(schema.order_col, "nunique"),
        )
        .reindex(pd.period_range(s["order_purchase_timestamp"].min(), s["order_purchase_timestamp"].max(), freq="M").astype(str), fill_value=0)
    )


def detect_period_changes(
    fact: pd.DataFrame,
    schema: FactSchema | None = None,
    min_abs_delta: float = 0.12,
    min_rows: int = 50,
    freq: str = "Q",
) -> list[PeriodChange]:
    """Period-over-period changes on the headline metrics (revenue, orders)."""
    schema = schema or FactSchema()
    s = fact.copy()
    s["_period"] = pd.PeriodIndex(s[schema.time_col], freq=freq).astype(str)
    s["_count"] = 1
    out: list[PeriodChange] = []
    for metric, how in (("revenue", "sum"), ("orders", "nunique")):
        grouped = s.groupby("_period").agg(
            value=(schema.revenue_col, how) if metric == "revenue" else (schema.order_col, "nunique"),
            n=(schema.order_col, "size"),
        )
        periods = sorted(grouped.index)
        # remove partial-edge periods (first/last months incomplete)
        for i in range(1, len(periods)):
            prev, cur = periods[i - 1], periods[i]
            pv, cv = grouped.loc[prev, "value"], grouped.loc[cur, "value"]
            nb, nc = int(grouped.loc[prev, "n"]), int(grouped.loc[cur, "n"])
            if nb < min_rows or nc < min_rows:
                continue
            base = float(pv)
            comp = float(cv)
            if base == 0:
                continue
            delta = (comp - base) / base
            if abs(delta) >= min_abs_delta:
                out.append(
                    PeriodChange(
                        period=cur,
                        prev_period=prev,
                        metric=metric,
                        base_value=base,
                        comp_value=comp,
                        delta_pct=delta,
                        n_base=nb,
                        n_comp=nc,
                    )
                )
    return out


def _scalar(a: float, b: float) -> tuple[float, float]:
    a = float(a)
    b = float(b)
    return a, b


def drilldown_change(
    fact: pd.DataFrame,
    period: object,
    prev_period: object,
    metric: str,
    schema: FactSchema | None = None,
    max_depth: int = 3,
) -> tuple[list[DrillDownStep], int]:
    """Recursively find the dimension chain that best explains a period change."""
    schema = schema or FactSchema()
    s = fact.copy()
    s["_period"] = pd.PeriodIndex(s[schema.time_col], freq="Q").astype(str)
    base = s[s["_period"] == str(prev_period)].copy()
    comp = s[s["_period"] == str(period)].copy()

    def col_values(df: pd.DataFrame, col: str) -> pd.Series:
        if metric == "orders":
            return df.groupby(col)[schema.order_col].nunique()
        return df.groupby(col)[schema.revenue_col].sum()

    total_base = float(col_values(base, "_period").iloc[0]) if len(base) else 0.0
    total_comp = float(col_values(comp, "_period").iloc[0]) if len(comp) else 0.0
    delta_total = (total_comp - total_base) / total_base if total_base else 0.0
    change_units = delta_total * total_base  # revenue (or order) units of the headline change

    steps: list[DrillDownStep] = []
    base_view = base
    comp_view = comp
    remaining_dims = list(schema.dimensions)

    for depth in range(max_depth):
        if not remaining_dims:
            break
        dim = remaining_dims[0]
        b = col_values(base_view, dim)
        c = col_values(comp_view, dim)
        idx = b.index.union(c.index)
        b = b.reindex(idx, fill_value=0.0)
        c = c.reindex(idx, fill_value=0.0)
        if b.sum() == 0:
            break
        contrib = c - b
        if change_units != 0:
            share = contrib / change_units
        else:
            share = contrib / contrib.abs().sum() if contrib.abs().sum() else contrib
        best = contrib.abs().idxmax()
        aval, cval = _scalar(b[best], c[best])
        step_delta = (cval - aval) / aval if aval else 0.0
        step_n = int(comp_view[comp_view[dim] == best].groupby("_period")[schema.order_col].nunique().sum())
        steps.append(
            DrillDownStep(
                dimension=dim,
                entity=str(best),
                base_value=aval,
                comp_value=cval,
                delta_pct=step_delta,
                contribution_pct=float(contrib[best] / change_units) if change_units else 0.0,
                n=step_n,
            )
        )
        base_view = base_view[base_view[dim] == best]
        comp_view = comp_view[comp_view[dim] == best]
        remaining_dims = remaining_dims[1:]
        if not len(base_view) or not len(comp_view):
            break

    return steps, int(len(comp))


def _bootstrap_ci(values_a: np.ndarray, values_b: np.ndarray, n_boot: int = 500, seed: int = 42) -> tuple[float, float]:
    """Bootstrap CI on the difference of means (a - b)."""
    rng = np.random.default_rng(seed)
    if len(values_a) == 0 or len(values_b) == 0:
        return (np.nan, np.nan)
    diffs = []
    for _ in range(n_boot):
        ma = rng.choice(values_a, size=len(values_a), replace=True).mean()
        mb = rng.choice(values_b, size=len(values_b), replace=True).mean()
        diffs.append(ma - mb)
    return (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))


def run_tests(
    base_values: pd.Series,
    comp_values: pd.Series,
    alpha: float = MIN_ALPHA,
) -> tuple[list[StatTest], tuple[float, float] | None]:
    """Two-group significance battery: Welch t, Mann-Whitney, effect sizes, bootstrap CI."""
    tests: list[StatTest] = []
    a = np.asarray(pd.to_numeric(base_values, errors="coerce").dropna(), dtype=float)
    b = np.asarray(pd.to_numeric(comp_values, errors="coerce").dropna(), dtype=float)
    if len(a) < 8 or len(b) < 8:
        return tests, None

    t, p_t = sps.ttest_ind(a, b, equal_var=False)
    tests.append(
        StatTest(
            name="Welch t-test",
            statistic=float(t),
            p_value=float(p_t),
            effect_size=float(_cohens_d(a, b)),
            effect_name="Cohen's d",
        )
    )
    u, p_u = sps.mannwhitneyu(a, b, alternative="two-sided")
    tests.append(
        StatTest(
            name="Mann-Whitney U",
            statistic=float(u),
            p_value=float(p_u),
            effect_size=float(_rank_biserial(a, b)),
            effect_name="rank-biserial r",
        )
    )
    if p_t > alpha and p_u > alpha:
        return tests, None
    return tests, _bootstrap_ci(a, b)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2)
    if pooled <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def _rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    combined = np.concatenate([a, b])
    ranks = sps.rankdata(combined)
    ra, rb = ranks[: len(a)], ranks[len(a):]
    u = ra.sum() - len(a) * (len(a) + 1) / 2
    return float(1 - 2 * u / (len(a) * len(b)))


def rolling_anomalies(
    series: pd.Series,
    window: int = 6,
    z_threshold: float = 2.5,
) -> pd.DataFrame:
    """Flag periods whose metric deviates strongly from a rolling median/MAD baseline."""
    s = series.astype(float)
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    z = (s - med) / mad.replace(0, np.nan)
    out = pd.DataFrame({"value": s, "median": med, "z_score": z})
    out["anomalous"] = z.abs() >= z_threshold
    return out


def segment_movers(
    fact: pd.DataFrame,
    dim: str,
    period_base: object,
    period_comp: object,
    schema: FactSchema | None = None,
    return_top: int = 8,
    min_cat_rows: int = 30,
) -> pd.DataFrame:
    """Per-category revenue change between two periods, with significance flags."""
    schema = schema or FactSchema()
    s = fact.copy()
    s["_period"] = pd.PeriodIndex(s[schema.time_col], freq="Q").astype(str)
    b = s[s["_period"] == str(period_base)]
    c = s[s["_period"] == str(period_comp)]
    rows = []
    for cat, group in c.groupby(dim):
        base_rev = float(b.loc[b[dim] == cat, schema.revenue_col].sum())
        base_n = int(b.loc[b[dim] == cat, schema.order_col].nunique())
        comp_rev = float(group[schema.revenue_col].sum())
        comp_n = int(group[schema.order_col].nunique())
        if base_n < min_cat_rows or comp_n < min_cat_rows:
            continue
        delta = (comp_rev - base_rev) / base_rev if base_rev else np.nan
        tests, ci = run_tests(
            b.loc[b[dim] == cat, schema.revenue_col],
            c.loc[c[dim] == cat, schema.revenue_col],
        )
        p_valid = min((t.p_value for t in tests if t.p_value is not None), default=1.0)
        rows.append(
            {
                dim: cat,
                "base_revenue": base_rev,
                "comp_revenue": comp_rev,
                "delta_pct": delta,
                "base_orders": base_n,
                "comp_orders": comp_n,
                "min_p": p_valid,
                "significant": p_valid < MIN_ALPHA,
            }
        )
    df = pd.DataFrame(rows).sort_values("delta_pct", ascending=False)
    return df.dropna().head(return_top) if len(df) else df


def daily_period_tests(
    fact: pd.DataFrame,
    period: object,
    prev_period: object,
    schema: FactSchema | None = None,
) -> list[StatTest]:
    """Compare daily order volume and daily revenue between two periods (Mann-Whitney)."""
    schema = schema or FactSchema()
    s = fact.copy()
    s["_period"] = pd.PeriodIndex(s[schema.time_col], freq="Q").astype(str)
    s["_day"] = s[schema.time_col].dt.date
    base = s[s["_period"] == str(prev_period)]
    comp = s[s["_period"] == str(period)]
    out: list[StatTest] = []

    def daily_series(df: pd.DataFrame, col: str) -> pd.Series:
        if col == "orders":
            return df.groupby("_day")[schema.order_col].nunique()
        return df.groupby("_day")[schema.revenue_col].sum()

    for label, col in (("orders", "orders"), ("revenue", schema.revenue_col)):
        a = daily_series(base, col)
        b = daily_series(comp, col)
        if len(a) < 10 or len(b) < 10:
            continue
        tests, _ = run_tests(a, b)
        for t in tests:
            t.name = f"{label} | {t.name}"
        out.extend(tests)
    # volume share: is the split of totals between the two periods balanced?
    k_comp = int(comp.groupby("_day")[schema.order_col].nunique().sum())
    k_base = int(base.groupby("_day")[schema.order_col].nunique().sum())
    n = k_comp + k_base
    if n > 0:
        share_comp = k_comp / n
        p_share = float(sps.binomtest(k_comp, n, 0.5, alternative="two-sided").pvalue)
        out.append(
            StatTest(
                name="volume share test (binomial)",
                statistic=float(share_comp),
                p_value=float(p_share),
                effect_size=(k_comp - k_base) / k_base if k_base else None,
                effect_name="relative volume change",
            )
        )
    return out


def correlation_hits(
    fact: pd.DataFrame,
    schema: FactSchema | None = None,
    min_abs: float = 0.2,
) -> list[tuple[str, str, float, float]]:
    """Pearson + Spearman for numeric fact columns; return pairs passing the bar."""
    schema = schema or FactSchema()
    cols = [c for c in schema.numeric_cols if c in fact.columns]
    hits: list[tuple[str, str, float, float]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            sub = fact[[a, b]].dropna()
            if len(sub) < 100:
                continue
            r, p = sps.pearsonr(sub[a], sub[b])
            rho, p_s = sps.spearmanr(sub[a], sub[b])
            best_r = rho if abs(rho) > abs(r) else r
            best_p = p_s if abs(rho) > abs(r) else p
            if abs(best_r) >= min_abs:
                hits.append((a, b, float(best_r), float(best_p)))
    return hits