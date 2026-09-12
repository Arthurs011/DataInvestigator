"""LLM decision layer: selection, prioritization and templated fallbacks."""

from __future__ import annotations

import json

import pytest

from investigator.config import Config
from investigator.llm_reporter import LLMReporter, _templated_summary
from investigator.models import Finding, Profiles

pytestmark = pytest.mark.unit


def _candidate(i: int, kind: str = "graph") -> Finding:
    headline = (
        {"period": "2020Q3", "prev_period": "2020Q2", "delta_pct": -0.2}
        if kind == "time_change"
        else {"state": "TS", "category": "furniture", "lift": 2.1}
    )
    return Finding(id=f"f{i}", title=f"Finding {i}", kind=kind, target_metric="revenue", headline=headline)


def test_templated_summary_is_nonempty():
    s = _templated_summary([_candidate(1), _candidate(2, "time_change")])
    assert isinstance(s, str) and len(s) > 20


def test_decide_without_llm_uses_deterministic_ranking():
    reporter = LLMReporter(Config(openrouter_api_key=""))
    cands = [_candidate(1, "graph"), _candidate(2, "time_change"), _candidate(3, "association")]
    ran, summary, selected = reporter.decide(cands, Profiles(), top_findings=2)
    assert ran is False
    assert summary
    assert len(selected) <= 2
    assert all(f.rank >= 1 for f in selected)
    assert all(f.narrative for f in selected)


def test_apply_decision_respects_llm_order_and_cap():
    reporter = LLMReporter(Config(openrouter_api_key=""))
    cands = [_candidate(1), _candidate(2), _candidate(3)]
    raw = json.dumps(
        {
            "summary": "All good.",
            "selected_ids": ["f3", "f1"],
            "findings": [
                {"id": "f3", "title": "F3", "narrative": "third", "confidence": 0.9, "caveats": []},
                {"id": "f1", "title": "F1", "narrative": "first", "confidence": 0.8, "caveats": []},
            ],
        }
    )
    summary, selected = reporter._apply_decision(cands, raw, top_findings=2)
    assert summary == "All good."
    assert [f.id for f in selected] == ["f3", "f1"]
    assert selected[0].narrative == "third"


def test_apply_decision_fills_gaps_with_ranked_rest():
    reporter = LLMReporter(Config(openrouter_api_key=""))
    cands = [_candidate(1), _candidate(2)]
    raw = json.dumps({"summary": "", "selected_ids": [], "findings": []})
    _, selected = reporter._apply_decision(cands, raw, top_findings=1)
    assert len(selected) == 1
    assert all(f.narrative for f in selected)