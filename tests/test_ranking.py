from __future__ import annotations

import pytest

from investigator.config import Config
from investigator.models import Evidence, Finding, StatTest
from investigator.ranking import rank_findings, score_finding

pytestmark = pytest.mark.unit


def _finding(kind: str, p_value: float = 0.001, rank: int = 0) -> Finding:
    t = StatTest(name="t", statistic=1.0, p_value=p_value, effect_size=1.0)
    return Finding(
        id=f"{kind}-{p_value}",
        title=f"{kind} ({p_value})",
        kind=kind,
        target_metric="revenue",
        stats=[t],
        rank=rank,
    )


def test_score_finding_baseline():
    f = Finding(
        id="x",
        title="x",
        kind="time_change",
        target_metric="revenue",
    )
    s = score_finding(f)
    assert 0.0 <= s <= 1.0


def test_score_finding_discriminates():
    strong = _finding("time_change", p_value=1e-12)
    weak = _finding("time_change", p_value=0.5)
    assert score_finding(strong) > score_finding(weak)


def test_rank_findings_sorts_by_score():
    findings = [_finding("time_change", 0.1), _finding("time_change", 1e-12), _finding("time_change", 0.3)]
    out = rank_findings(findings, Config(top_findings=10))
    assert [f.rank for f in out] == [1, 2, 3]
    assert [f.score for f in out] == sorted((score_finding(f) for f in findings), reverse=True)


def test_rank_findings_diversity():
    """A single low-scoring graph finding must still make the cut when the
    top-N are all time_changes."""
    findings = []
    for i in range(12):
        findings.append(_finding("time_change", p_value=1e-10 + i * 1e-11))
    graph = _finding("graph", p_value=0.5)
    findings.append(graph)
    out = rank_findings(findings, Config(top_findings=10))
    kinds = {f.kind for f in out}
    assert "graph" in kinds
    assert len(out) == 10


def test_rank_findings_respects_top_findings():
    findings = [_finding("time_change", p_value=0.2 * (i + 1) if i < 5 else 0.01) for i in range(20)]
    out = rank_findings(findings, Config(top_findings=5))
    assert len(out) == 5
    assert [f.rank for f in out] == [1, 2, 3, 4, 5]


def test_verified_evidence_preferred():
    base = _finding("time_change")
    base.evidence = [Evidence(kind="sql", code="SELECT 1", result_head="1", verified=True)]
    no_verify = _finding("time_change")
    no_verify.evidence = [Evidence(kind="sql", code="SELECT 1", result_head="1", verified=False)]
    assert score_finding(base) > score_finding(no_verify)