"""Render the Investigation Report: charts + Markdown + self-contained HTML."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .models import InvestigationResult

_FINDING_ICON = {
    "time_change": "▸",
    "time_anomaly": "◬",
    "segment_mover": "◂",
    "association": "⌁",
    "graph": "◇",
}


def chart_monthly_revenue(fact: pd.DataFrame, schema, outdir: Path) -> str:
    s = fact.copy()
    s["_ym"] = pd.PeriodIndex(s[schema.time_col], freq="M").astype(str)
    g = s.groupby("_ym").agg(revenue=(schema.revenue_col, "sum"), orders=(schema.order_col, "nunique"))
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=110)
    ax.plot(range(len(g)), g["revenue"] / 1e6, marker="o", ms=3, lw=1.4, color="#2563eb")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, rotation=90, fontsize=7)
    ax.set_ylabel("revenue (M)")
    ax.set_title("Monthly revenue")
    ax.grid(alpha=0.25)
    path = outdir / "chart_monthly_revenue.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path.name


def chart_quarterly_revenue(fact: pd.DataFrame, schema, outdir: Path) -> str:
    s = fact.copy()
    s["_q"] = pd.PeriodIndex(s[schema.time_col], freq="Q").astype(str)
    g = s.groupby("_q").agg(revenue=(schema.revenue_col, "sum"))
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=110)
    ax.bar(range(len(g)), g["revenue"] / 1e6, color="#94a3b8")
    if len(g) >= 2:
        ax.bar(len(g) - 1, g["revenue"].iloc[-1] / 1e6, color="#f43f5e")
        ax.bar(len(g) - 2, g["revenue"].iloc[-2] / 1e6, color="#f43f5e")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, rotation=45, fontsize=8)
    ax.set_ylabel("revenue (M)")
    ax.set_title("Quarterly revenue (last two quarters highlighted)")
    ax.grid(axis="y", alpha=0.25)
    path = outdir / "chart_quarterly_revenue.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path.name


def _to_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _markdown_finding(f, figures: dict) -> str:
    icon = _FINDING_ICON.get(f.kind, "•")
    conf = f"{(f.confidence or 0) * 100:.0f}%" if f.confidence is not None else "n/a"
    lines = [f"## {f.rank}. {icon} {f.title}", ""]
    lines.append(f"> **Confidence:** {conf} · **Source agents:** {', '.join(f.agents)}")
    lines.append("")
    if f.narrative:
        lines.append(f"{f.narrative}")
        lines.append("")
    h = f.headline
    if h:
        rows = []
        for k, v in h.items():
            if isinstance(v, float):
                v = f"{v:,.2f}" if abs(v) >= 100 else f"{v:.3f}"
            elif k == "delta_pct":
                v = f"{float(v):+.1%}"
            rows.append(f"| {k} | {v} |")
        lines.append("Key numbers:")
        lines.append("")
        lines.append("| field | value |")
        lines.append("|---|---|")
        lines += rows
        lines.append("")
    if f.drill_down:
        lines.append("**Drill-down (concentration chain):**")
        lines.append("")
        lines.append("| dimension | entity | Δ% | share of change | orders |")
        lines.append("|---|---|---|---|---|")
        for d in f.drill_down:
            cp = "n/a" if pd.isna(d.contribution_pct) else f"{d.contribution_pct:.0%}"
            lines.append(f"| {d.dimension} | {d.entity} | {d.delta_pct:+.1%} | {cp} | {d.n} |")
        lines.append("")
    if f.stats:
        lines.append("**Statistical tests:**")
        lines.append("")
        lines.append("| test | statistic | p-value | effect |")
        lines.append("|---|---|---|---|")
        for t in f.stats:
            st = "—" if t.statistic is None else f"{t.statistic:.3g}"
            pv = "—" if t.p_value is None else f"{t.p_value:.2e}"
            ef = "—" if t.effect_size is None else f"{t.effect_size:.3f}"
            lines.append(f"| {t.name} | {st} | {pv} | {ef} |")
        lines.append("")
    if f.causal:
        c = f.causal
        lines.append(f"**Causal testing:** `{c.verdict}` — {c.method}")
        lines.append("")
        if c.effect is not None:
            lines.append(f"- effect: {c.effect:,.1f}, p = {c.p_value:.3g}")
        if c.ci:
            lines.append(f"- bootstrap 95% CI: [{c.ci[0]:,.1f}, {c.ci[1]:,.1f}]")
        for d_ in c.details:
            lines.append(f"- {d_}")
        lines.append("")
    if f.caveats:
        lines.append("**Caveats:** " + "; ".join(f.caveats))
        lines.append("")
    sql_ev = [e for e in f.evidence if e.kind == "sql"]
    py_ev = [e for e in f.evidence if e.kind == "python"]
    for e in sql_ev:
        ver = "✓ verified against engine numbers" if e.verified else "not cross-verified"
        lines.append(f"**SQL evidence** ({ver}):")
        lines.append("")
        lines.append("```sql")
        lines.append(e.code)
        lines.append("```")
        lines.append("")
        if e.result_head:
            lines.append("Query result:")
            lines.append("")
            lines.append("```text")
            lines.append(e.result_head)
            lines.append("```")
            lines.append("")
    for e in py_ev:
        lines.append("**Python evidence:**")
        lines.append("")
        lines.append("```python")
        lines.append(e.code)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def generate_report(result: InvestigationResult, outdir: Path) -> dict[str, Path]:
    """Write reports/report.md and reports/report.html; return map name->path."""
    outdir.mkdir(parents=True, exist_ok=True)

    # build charts against the freshly loaded fact table
    from .loader import load_dataset

    ds = load_dataset(result.data_dir)
    fig_files = {
        "chart_monthly_revenue.png": chart_monthly_revenue(ds.fact, ds.schema, outdir),
        "chart_quarterly_revenue.png": chart_quarterly_revenue(ds.fact, ds.schema, outdir),
    }

    md: list[str] = [
        "# Investigation Report",
        "",
        f"*Dataset: `{result.data_dir}` · {len(result.findings)} ranked findings · "
        f"generated in {result.duration_s:.1f}s · LLM narrative: {'yes' if result.ran_llm else 'no (templated)'}*",
        "",
        "## Overview — data understanding",
        "",
        "The system discovered the following tables and joins:",
        "",
        "| table | rows | cols | peak missing % |",
        "|---|---|---|---|",
    ]
    for p in result.profiles.tables.values():
        peak = max(p.missing.values(), default=0.0)
        md.append(f"| {p.name} | {p.n_rows:,} | {p.n_cols} | {peak:.0%} |")
    md.append("")
    md.append("**Join graph:**")
    md.append("")
    md.append("```")
    for e in result.join_graph:
        md.append(f"{e.left_table}.{e.left_col} == {e.right_table}.{e.right_col}  "
                  f"(matched {e.overlap_frac:.0%})")
    md.append("```")
    md.append("")
    if result.executive_summary:
        md.append("## Executive summary")
        md.append("")
        md.append(result.executive_summary)
        md.append("")
    md.append("![Monthly revenue](chart_monthly_revenue.png)")
    md.append("")
    md.append("![Quarterly revenue](chart_quarterly_revenue.png)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("# Findings")
    md.append("")

    for f in result.findings:
        md.append(_markdown_finding(f, fig_files))
        md.append("---")
        md.append("")
    md_path = outdir / "report.md"
    md_path.write_text("\n".join(md))

    # HTML (self-contained, images inlined)
    html_body = _as_html(md, outdir, fig_files)
    html_path = outdir / "report.html"
    html_path.write_text(html_body)
    return {"markdown": md_path, "html": html_path}


def _as_html(md_lines: list[str], outdir: Path, fig_files: dict[str, str]) -> str:
    src = "\n".join(md_lines)
    for name, fname in fig_files.items():
        path = outdir / fname
        if path.exists():
            src = src.replace(name, _to_data_url(path))
    import markdown  # type: ignore

    body = markdown.markdown(src, extensions=["tables", "fenced_code"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Investigation Report</title>
<style>
  body {{ max-width: 900px; margin: 2rem auto; padding: 0 1rem;
         font: 15px/1.6 -apple-system, Segoe UI, Roboto, sans-serif; color: #1f2937; }}
  h1 {{ border-bottom: 3px solid #2563eb; padding-bottom: .3rem; }}
  h2 {{ margin-top: 2.2rem; border-bottom: 1px solid #e5e7eb; padding-bottom: .2rem; }}
  code {{ background: #f3f4f6; padding: .1rem .3rem; border-radius: 4px; }}
  pre {{ background: #0f172a; color: #e2e8f0; padding: .8rem; border-radius: 8px; overflow-x: auto; }}
  pre code {{ background: none; color: inherit; }}
  table {{ border-collapse: collapse; width: 100%; margin: .6rem 0; }}
  th, td {{ border: 1px solid #e5e7eb; padding: .3rem .5rem; text-align: left; font-size: 13px; }}
  th {{ background: #f8fafc; }}
  blockquote {{ border-left: 4px solid #2563eb; margin: 0; padding: 0 .8rem; color: #334155; }}
  img {{ max-width: 100%; }}
</style></head><body>{body}</body></html>"""