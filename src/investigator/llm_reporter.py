"""LLM report generation via OpenRouter (OpenAI-compatible API).

Turns ranked, evidence-backed candidates into narrative Investigation prose.
Degrades gracefully to a templated narrative when no API key is available.
"""

from __future__ import annotations

import json

from .config import Config
from .models import Finding, Profiles

SYSTEM_PROMPT = (
    "You are an autonomous data-science investigator. A deterministic engine has already "
    "discovered, statistically tested and verified findings in a real messy dataset. "
    "Your job is to write crisp, honest narrative prose for each one, in plain language. "
    "Do not invent numbers. Do not claim causality the evidence doesn't support. "
    "Where the causal verdict is 'confounded' or 'spurious', say so plainly and note "
    "what control was applied. Keep each narrative under 90 words. "
    "Return STRICT JSON: an object with key 'findings', a list of objects with keys "
    '{"id": str, "title": str, "narrative": str, "confidence": float (0..1), '
    '"caveats": [str]}.'
)

DECISION_SYSTEM_PROMPT = (
    "You are a data-science lead reviewing candidates produced by an autonomous "
    "investigation engine. Every candidate is evidence-backed (numbers were "
    "cross-verified with SQL). Your job:\n"
    "1. Select the findings that matter most to a busy executive (not every "
    "candidate is worth reporting).\n"
    "2. Order your selection from most to least important.\n"
    "3. For each selected finding write an honest narrative under 90 words — plain "
    "language, no invented numbers, flag confounded/spurious causal verdicts.\n"
    "4. Write a 2-4 sentence executive summary that a non-technical reader can act on.\n"
    "Return STRICT JSON with exactly these keys:\n"
    '{"summary": str, "selected_ids": [str], "findings": [{"id": str, "title": str, '
    '"narrative": str, "confidence": float (0..1), "caveats": [str]}]}\n'
    "selected_ids must use the candidate ids you were given, in priority order."
)


def _finding_payload(f: Finding) -> dict:
    return {
        "id": f.id,
        "title": f.title,
        "kind": f.kind,
        "key_numbers": f.headline,
        "drill_down": [
            {"dimension": d.dimension, "entity": d.entity, "delta_pct": d.delta_pct, "contribution_pct": d.contribution_pct}
            for d in f.drill_down
        ],
        "stats": [
            {"test": t.name, "p_value": t.p_value, "effect_size": t.effect_size, "effect_name": t.effect_name}
            for t in f.stats
        ],
        "causal": None
        if f.causal is None
        else {"verdict": f.causal.verdict, "effect": f.causal.effect, "p_value": f.causal.p_value, "ci": f.causal.ci},
        "evidence_verified": any(e.verified for e in f.evidence),
    }


def _templated_narrative(f: Finding) -> tuple[str, float, list[str]]:
    h = f.headline
    if f.kind == "time_change":
        actor = "The metric"
        if h.get("delta_pct", 0) < 0:
            direction = f"decreased {abs(h['delta_pct']):.1%}"
        else:
            direction = f"increased {h.get('delta_pct', 0):.1%}"
        chain = " -> ".join(
            f"{d.entity} ({d.dimension})" for d in f.drill_down[:3]
        )
        conf = 0.8 if any(e.verified for e in f.evidence) else 0.6
        caveats = [
            c for c in f.caveats
        ]
        if f.causal and f.causal.verdict != "supported":
            caveats.append(f"causal check: {f.causal.verdict}")
        return (
            f"{f.target_metric.title()} {direction} in {h['period']} vs {h['prev_period']}. "
            f"The concentration chain is: {chain}.",
            conf,
            caveats,
        )
    if f.kind == "time_anomaly":
        z = h.get("z_score", 0)
        return (
            f"Monthly revenue was an outlier in {h['period']} (z-score {z:+.1f} vs rolling "
            f"median), suggesting a non-routine event relative to the surrounding trend.",
            0.7,
            ["based on a rolling median/MAD z-score, not a model"],
        )
    if f.kind == "segment_mover":
        delta = h.get("delta_pct", 0)
        return (
            f"Segment '{h.get('entity')}' ({h.get('dimension')}) split to "
            f"{delta:+.0%} revenue in {h['period']} vs {h['prev_period']}.",
            0.7,
            [],
        )
    if f.kind == "association":
        return (
            f"Correlation between '{h.get('a')}' and '{h.get('b')}' is r={h.get('r', 0):+.2f}. "
            "Correlation does not imply causation.",
            0.5,
            ["observational correlation only"],
        )
    if f.kind == "graph":
        if "state" in h:
            return (
                f"Category '{h['category']}' over-indexes in {h['state']} "
                f"(index {h.get('lift', 0):.1f}), well above its national share "
                f"of {h.get('share_state', 0):.0%} of state revenue.",
                0.7,
                [],
            )
        return (
            f"Co-purchase affinity between '{h.get('a')}' and '{h.get('b')}' "
            f"(lift {h.get('lift', 0):.1f}).",
            0.7,
            [],
        )
    return (f"{f.title}.", 0.5, [])


def _templated_summary(findings: list[Finding]) -> str:
    """Deterministic executive summary from the ranked findings' headlines."""
    bits: list[str] = []
    for f in findings:
        h = f.headline
        if f.kind == "time_change":
            bits.append(
                f"{f.target_metric} {h.get('period')} vs {h.get('prev_period')} changed "
                f"{h.get('delta_pct', 0):+.1%}"
            )
        elif f.kind == "time_anomaly":
            bits.append(f"an outlier month {h.get('period')} (z={h.get('z_score', 0):+.1f})")
        elif f.kind == "graph" and "state" in h:
            bits.append(f"{h.get('category')} over-indexing in {h.get('state')}")
        elif f.kind == "segment_mover":
            bits.append(f"segment {h.get('entity')} moving {h.get('delta_pct', 0):+.0%}")
        elif f.kind == "association":
            bits.append(f"correlation between {h.get('a')} and {h.get('b')}")
    if not bits:
        return (
            f"The investigation surfaced {len(findings)} evidence-backed findings; "
            "no single anomaly dominates the dataset's behaviour."
        )
    return (
        f"The investigation surfaced {len(findings)} evidence-backed findings. The most "
        f"material are: {'; '.join(bits[:6])}."
    )


class LLMReporter:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.client = None
        if self.config.openrouter_api_key:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.config.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
            )

    @property
    def available(self) -> bool:
        return self.client is not None

    def narrate(self, findings: list[Finding], profiles: Profiles) -> tuple[bool, str]:
        """Return (used_llm, raw_json_or_empty)."""
        if not self.available:
            for f in findings:
                f.narrative, f.confidence, f.caveats = _templated_narrative(f)
            return False, ""
        try:
            payload = {
                "tables": [
                    {"name": p.name, "rows": p.n_rows, "cols": p.n_cols, "missing_frac": p.missing}
                    for p in profiles.tables.values()
                ],
                "findings": [_finding_payload(f) for f in findings],
            }
            resp = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            raw = resp.choices[0].message.content or ""
            self._apply(findings, raw)
            return True, raw
        except Exception as e:
            print(f"[llm_reporter] LLM call failed ({e}); using templated narratives")
            for f in findings:
                f.narrative, f.confidence, f.caveats = _templated_narrative(f)
            return False, ""

    def decide(self, candidates: list[Finding], profiles: Profiles, top_findings: int) -> tuple[bool, str, list[Finding]]:
        """Select, prioritize and narrate findings.

        Returns (used_llm, executive_summary, selected_findings). Selection is the
        LLM's job when a key is configured; otherwise the deterministic ranking is
        kept and narratives are templated.
        """
        if not self.available:
            return self._decide_deterministically(candidates, top_findings)
        try:
            payload = {
                "tables": [
                    {"name": p.name, "rows": p.n_rows, "cols": p.n_cols, "missing_frac": p.missing}
                    for p in profiles.tables.values()
                ],
                "candidates": [
                    {
                        "id": f.id,
                        "title": f.title,
                        "kind": f.kind,
                        "score": round(f.score, 3),
                        **_finding_payload(f),
                    }
                    for f in candidates
                ],
            }
            resp = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            raw = resp.choices[0].message.content or ""
            summary, selected = self._apply_decision(candidates, raw, top_findings)
            return True, summary, selected
        except Exception as e:
            print(f"[llm_reporter] LLM decision call failed ({e}); falling back to deterministic ranking")
            return self._decide_deterministically(candidates, top_findings)

    def _decide_deterministically(self, candidates: list[Finding], top_findings: int) -> tuple[bool, str, list[Finding]]:
        from .ranking import rank_findings

        selected = rank_findings(candidates, self.config)[:top_findings]
        for f in selected:
            f.narrative, f.confidence, f.caveats = _templated_narrative(f)
        return False, _templated_summary(selected), selected

    def _apply_decision(self, candidates: list[Finding], raw: str, top_findings: int) -> tuple[str, list[Finding]]:
        try:
            doc = json.loads(raw)
            summary = str(doc.get("summary") or "").strip()
            ordered_ids = [str(i) for i in doc.get("selected_ids") or []]
            by_id = {i["id"]: i for i in doc.get("findings") or []}
        except (json.JSONDecodeError, TypeError):
            return "", []
        by_cand = {f.id: f for f in candidates}
        ordered = [by_cand[i] for i in ordered_ids if i in by_cand]
        if len(ordered) < len({i for i in ordered_ids if i in by_cand}):
            ordered = list(dict.fromkeys(ordered))
        extra = [f for f in candidates if f.id not in {f.id for f in ordered}]
        extra.sort(key=lambda f: (-f.score, f.id))
        ordered += extra
        selected = ordered[:top_findings]
        for f in selected:
            item = by_id.get(f.id)
            if item:
                f.narrative = str(item.get("narrative") or f.narrative or "")
                conf = item.get("confidence")
                if isinstance(conf, (int, float)):
                    f.confidence = float(min(max(conf, 0.0), 1.0))
                else:
                    f.confidence = f.confidence if f.confidence is not None else 0.6
                extra_caveats = item.get("caveats") or []
                if isinstance(extra_caveats, list):
                    f.caveats = [str(c) for c in extra_caveats if str(c).strip()]
            else:
                f.narrative, f.confidence, f.caveats = _templated_narrative(f)
        return summary or _templated_summary(selected), selected

    def _apply(self, findings: list[Finding], raw: str) -> None:
        try:
            doc = json.loads(raw)
            by_id = {i["id"]: i for i in doc.get("findings", [])}
        except (json.JSONDecodeError, TypeError):
            for f in findings:
                f.narrative, f.confidence, f.caveats = _templated_narrative(f)
            return
        for f in findings:
            item = by_id.get(f.id)
            if not item:
                f.narrative, f.confidence, f.caveats = _templated_narrative(f)
                continue
            f.narrative = str(item.get("narrative") or f.narrative or "")
            conf = item.get("confidence")
            if isinstance(conf, (int, float)):
                f.confidence = float(min(max(conf, 0.0), 1.0))
            else:
                f.confidence = f.confidence if f.confidence is not None else 0.6
            extra = item.get("caveats") or []
            if isinstance(extra, list):
                f.caveats = [str(c) for c in extra if str(c).strip()]