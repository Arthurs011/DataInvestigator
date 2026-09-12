"""End-to-end orchestration of the investigator pipeline."""

from __future__ import annotations

import time
from pathlib import Path

from .causal_engine import CausalEngine
from .config import Config, load_config
from .graph_engine import co_occurrence_lift, category_geo_lift
from .hypothesis_engine import HypothesisEngine
from .llm_reporter import LLMReporter
from .loader import Dataset, load_dataset
from .models import InvestigationResult, JoinEdge
from .profile import discover_joins, join_graph_text, profile_tables
from .ranking import rank_findings
from .report import generate_report


def investigate(
    data_dir: str | Path,
    outdir: str | Path,
    config: Config | None = None,
    no_llm: bool = False,
) -> dict[str, Path]:
    """Run the full pipeline and write the Investigation Report."""
    config = config or load_config()
    if no_llm:
        config.openrouter_api_key = ""
    t0 = time.time()

    print(f"[loader] loading CSVs from {data_dir} ...")
    ds: Dataset = load_dataset(data_dir, row_limit=config.row_limit)

    print("[profile] profiling tables and discovering joins ...")
    profiles = profile_tables(ds.raw)
    joins = discover_joins(ds.raw)
    print("  " + join_graph_text(joins).replace("\n", "\n  "))

    print("[hypothesis] generating candidate findings with DuckDB evidence ...")
    he = HypothesisEngine(ds, profiles, data_dir, config)
    candidates = he.run()
    he.db.close()

    print("[causal] testing causal support for headline findings ...")
    ce = CausalEngine(ds, profiles, config)
    candidates = ce.test_all(candidates)

    print("[ranking] scoring and ranking findings ...")
    top = rank_findings(candidates, config)
    print(f"  top {len(top)} findings selected")

    print("[report] narrating findings and generating report ...")
    reporter = LLMReporter(config)
    ran_llm, llm_raw = reporter.narrate(top, profiles)

    result = InvestigationResult(
        data_dir=str(data_dir),
        profiles=profiles,
        join_graph=joins,
        findings=top,
        ran_llm=ran_llm,
        duration_s=time.time() - t0,
    )
    print(f"[report] writing to {outdir} ...")
    paths = generate_report(result, Path(outdir))
    print(f"\nDone — total time {result.duration_s:.1f}s")
    for label, p in paths.items():
        print(f"  {label}: {p}")
    return paths