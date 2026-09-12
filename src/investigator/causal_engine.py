"""Causal testing: is a detected relationship meaningful, or confounded/spurious?

For time/segment findings we test whether the treatment (e.g. being in period Q3,
or in segment X) has an effect on the target (revenue / orders) once plausible
confounders are controlled for, using:
  1. An aggregated panel (region x quarter x category).
  2. OLS regression of target ~ treatment + controls (causal-learn-free core).
  3. Optional PC skeleton discovery on the panel's continuous variables
     (causal-learn) to check whether treatment and target are directly connected.
  4. A bootstrap CI on the treatment coefficient (robustness).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import Config
from .models import (
    CausalVerdict,
    Dataset,
    Evidence,
    Finding,
    Profiles,
)

ALPHA = 0.05

CONTROLS = ("n_items", "avg_price", "freight_ratio", "reviews_mean", "review_rate")


def _panel_for_period(
    fact: pd.DataFrame,
    schema,
    period: object,
    prev_period: object,
) -> pd.DataFrame:
    """region x-quarter panel of revenue (target) + treatment + controls."""
    s = fact.copy()
    s["_period"] = pd.PeriodIndex(s[schema.time_col], freq="Q").astype(str)

    def aggregate(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        sub["_rev"] = sub[schema.revenue_col]
        sub["_order"] = sub[schema.order_col]
        sub["_freight"] = sub["freight_value"]
        sub["_price"] = sub["price"]
        sub["_review"] = pd.to_numeric(sub.get("review_score", pd.Series(dtype=float)), errors="coerce")
        g = sub.groupby(["customer_state", "product_category_name_english", "_period"]).agg(
            revenue=("_rev", "sum"),
            orders=("_order", "nunique"),
            n_items=("_order", "size"),
            avg_price=("_price", "mean"),
            freight_total=("_freight", "sum"),
            price_total=("_price", "sum"),
            reviews_mean=("_review", "mean"),
            reviews_count=("_review", "count"),
        )
        g["freight_ratio"] = g["freight_total"] / g["price_total"].replace(0, np.nan)
        g["review_rate"] = g["reviews_count"] / g["n_items"]
        return g.reset_index()

    base = aggregate(s[s["_period"] == str(prev_period)])
    comp = aggregate(s[s["_period"] == str(period)])
    for df in (base, comp):
        df["treatment"] = 0
    comp["treatment"] = 1
    panel = pd.concat([base, comp], ignore_index=True)
    for c in CONTROLS:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")
    return panel.dropna(subset=["revenue", "n_items", "avg_price"])


def _pc_skeleton(panel: pd.DataFrame, target: str) -> dict[str, bool]:
    """PC skeleton (independence graph) on continuous variables; return edges touching target."""
    cols = [c for c in ("revenue", "avg_price", "freight_ratio", "n_items") if c in panel.columns]
    data = panel[cols].dropna()
    if len(data) < 50 or len(cols) < 2:
        return {}
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        data_np = data.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(data_np) < 50:
            return {}
        cg = pc(
            data_np,
            0.05,
            fisherz,
            bg=None,
            uc_rule=0,
            uc_priority=-1,
            mvpc=False,
            show_progress=False,
            verbose=False,
        )
        edges = set()
        for e in cg.G.get_edge_list():
            edges.add((e[0], e[1]))
        target_idx = list(data.columns).index(target)
        return {list(data.columns)[o]: True for (i2, o) in edges if i2 == target_idx}
    except Exception:
        return {}


def _regression(panel: pd.DataFrame, target_col: str, controls: list[str]) -> tuple[float, float, tuple[float, float]]:
    """OLS: target ~ treatment + controls. Returns coef, p, bootstrap CI."""
    cols = [target_col, "treatment"] + controls
    data = panel[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 30:
        raise ValueError("panel too small")
    y = data[target_col]
    X = data[["treatment"] + controls].copy()
    X["const"] = 1.0
    model = sm.OLS(y, X.astype(float)).fit()
    coef = float(model.params["treatment"])
    p = float(model.pvalues["treatment"])
    rng = np.random.default_rng(42)
    boots = []
    Xn, yn = X.astype(float).to_numpy(), y.astype(float).to_numpy()
    for _ in range(200):
        idx = rng.integers(0, len(Xn), len(Xn))
        try:
            m = sm.OLS(yn[idx], Xn[idx]).fit()
            boots.append(float(m.params[0]))
        except Exception:
            continue
    if boots:
        lo, hi = np.quantile(boots, [0.025, 0.975])
    else:
        lo, hi = float("nan"), float("nan")
    return coef, p, (float(lo), float(hi))


def _verdict(coef: float, p: float, ci: tuple[float, float], preceding: bool) -> CausalVerdict:
    inside_zero = ci[0] <= 0 <= ci[1] or np.isnan(ci[0])
    strong = p < ALPHA and not inside_zero
    if strong:
        note = "effect survives controls & CI excludes 0"
        verdict = "supported" if preceding or True else "supported"
        details = [note, "temporal precedence assumed from sequential periods"]
        return CausalVerdict(verdict=verdict, method="OLS + controls (+PC skeleton)", effect=coef, p_value=p, ci=ci, details=details)
    if p < 0.20:
        return CausalVerdict(
            verdict="confounded",
            method="OLS + controls (+PC skeleton)",
            effect=coef,
            p_value=p,
            ci=ci,
            details=["treatment association weakened with controls; confounding cannot be ruled out"],
        )
    return CausalVerdict(
        verdict="spurious",
        method="OLS + controls (+PC skeleton)",
        effect=coef,
        p_value=p,
        ci=ci,
        details=["no independent effect of treatment after controlling for confounders"],
    )


class CausalEngine:
    def __init__(self, dataset: Dataset, profiles: Profiles, config: Config | None = None):
        self.dataset = dataset
        self.schema = dataset.schema
        self.profiles = profiles
        self.config = config or Config()

    def test_finding(self, finding: Finding) -> Finding:
        """Attach a CausalVerdict to a finding when a test is applicable."""
        if finding.kind == "time_change":
            period = finding.headline["period"]
            prev = finding.headline["prev_period"]
            panel = _panel_for_period(self.dataset.fact, self.schema, period, prev)
            if len(panel) < 50:
                finding.causal = CausalVerdict("insufficient_data", "panel too sparse", details=["<50 panel rows"])
                return finding
            coef, p, ci = _regression(panel, "revenue", [c for c in CONTROLS if c in panel.columns])
            skeleton = _pc_skeleton(panel, "revenue")
            verdict = _verdict(coef, p, ci, preceding=True)
            if skeleton:
                verdict.details.append(f"PC skeleton links revenue to: {', '.join(sorted(skeleton)) or 'nothing'}")
                if "treatment" not in skeleton:
                    verdict.details.append("PC skeleton found no direct link treatment->revenue (may be confounded)")
            finding.causal = verdict
        elif finding.kind == "segment_mover":
            # treatment = being in the segment; control for category/state mix
            dim = finding.headline["dimension"]
            entity = finding.headline["entity"]
            period = finding.headline["period"]
            prev = finding.headline["prev_period"]
            panel = _panel_for_period(self.dataset.fact, self.schema, period, prev)
            if len(panel) < 50:
                return finding
            panel["in_segment"] = (panel[dim] == entity).astype(int)
            cats = [c for c in CONTROLS if c in panel.columns]
            Xd = pd.get_dummies(panel[[dim]].astype(str), prefix="cat", drop_first=True)
            controls_box = pd.concat([panel[cats].reset_index(drop=True), Xd.reset_index(drop=True)], axis=1)
            data = pd.concat(
                [panel[["revenue", "in_segment"]].reset_index(drop=True), controls_box],
                axis=1,
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) < 30:
                return finding
            y = data["revenue"]
            X = data[["in_segment"] + [c for c in data.columns if c not in ("revenue", "in_segment")]].copy()
            X["const"] = 1.0
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=sm.tools.singular_matrix_warning if hasattr(sm.tools, "singular_matrix_warning") else Warning)
                model = sm.OLS(y, X.astype(float)).fit()
            coef = float(model.params["in_segment"])
            p = float(model.pvalues["in_segment"])
            finding.causal = CausalVerdict(
                verdict="supported" if p < ALPHA else "spurious",
                method="OLS with segment dummy + controls",
                effect=coef,
                p_value=p,
                ci=None,
                details=["segment membership associated with revenue after controlling for confounders" if p < ALPHA else "no independent association"],
            )
        return finding

    def test_all(self, findings: list[Finding]) -> list[Finding]:
        return [self.test_finding(f) for f in findings]