"""Evidence ranking: combine significance, effect size, causal support, robustness,
verifiability and coverage into a single score, then pick the top findings."""

from __future__ import annotations

import math

import numpy as np

from .config import Config
from .models import CausalVerdict, Finding

_CAUSAL_SCORE = {
    "supported": 1.0,
    "confounded": 0.6,
    "spurious": 0.3,
    "insufficient_data": 0.45,
}


def _significance(f: Finding) -> float:
    ps = [t.p_value for t in f.stats if t.p_value is not None]
    if not ps:
        return 0.5
    p = min(ps)
    return float((1.0 - p) ** 0.5)  # sqrt keeps it smooth


def _effect(f: Finding) -> float:
    effects = [
        abs(t.effect_size)
        for t in f.stats
        if t.effect_size is not None and np.isfinite(t.effect_size)
    ]
    if not effects:
        return 0.4
    best = max(effects)
    return float(math.tanh(best / max(best, 1e-9)) if best >= 1 else best)


def _causal(f: Finding) -> float:
    if f.causal is None:
        return 0.5
    return float(_CAUSAL_SCORE.get(f.causal.verdict, 0.5))


def _robustness(f: Finding) -> float:
    if f.causal is None or f.causal.ci is None:
        return 0.6
    lo, hi = f.causal.ci
    if np.isnan(lo):
        return 0.4
    if lo <= 0 <= hi:
        return 0.4
    return 1.0


def _verified(f: Finding) -> float:
    return 1.0 if any(e.verified for e in f.evidence) else 0.4


def score_finding(f: Finding, weights: tuple[float, float, float, float, float, float] = (0.25, 0.2, 0.25, 0.1, 0.1, 0.1)) -> float:
    wsig, weff, wcau, wrob, wver, wcov = weights
    s = (
        wsig * _significance(f)
        + weff * _effect(f)
        + wcau * _causal(f)
        + wrob * _robustness(f)
        + wver * _verified(f)
        + wcov * min(f.coverage * 5.0, 1.0)  # scale coverage to [0,1] quickly
    )
    return float(s)


def rank_findings(findings: list[Finding], config: Config | None = None, diversity: bool = True) -> list[Finding]:
    config = config or Config()
    for f in findings:
        f.score = score_finding(f)
    ranked = sorted(findings, key=lambda f: f.score, reverse=True)
    for i, f in enumerate(ranked, start=1):
        f.rank = i
    if not diversity:
        return ranked
    if len(ranked) <= config.top_findings:
        return ranked
    top = ranked[: config.top_findings]
    # diversity: for each missing kind, swap in its best unshown finding,
    # evicting the lowest-scoring member of an over-represented kind (never a
    # finding that was itself added for diversity).
    have = {f.kind for f in top}
    for kind in ("graph", "segment_mover", "association", "time_anomaly"):
        if kind in have:
            continue
        have.add(kind)
        top_ids = {id(f) for f in top}
        extra = next((f for f in ranked if f.kind == kind and id(f) not in top_ids), None)
        if extra is None:
            continue
        top.append(extra)
    if len(top) > config.top_findings:
        kinds = {}
        for f in top:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
        victims = [i for i, f in enumerate(top) if kinds[f.kind] > 1]
        while len(top) > config.top_findings:
            drop = min(victims, key=lambda i: top[i].score)
            top.pop(drop)
            victims.remove(drop)
    top = sorted(top, key=lambda f: f.score, reverse=True)
    for i, f in enumerate(top, start=1):
        f.rank = i
    return top