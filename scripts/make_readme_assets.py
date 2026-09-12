"""Generate the README images (pipeline diagram + example charts).

Outputs:

  docs/img/pipeline.png              pipeline flow diagram
  docs/img/example_monthly_revenue.png    monthly revenue chart (real Olist data)
  docs/img/example_quarterly_revenue.png  quarterly revenue chart (real Olist data)

The charts reuse the report's own chart functions, so what you see in the README
is exactly what a report run produces.

Usage:
    uv run python scripts/make_readme_assets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from investigator.loader import load_dataset
from investigator.report import chart_monthly_revenue, chart_quarterly_revenue

_BLUE = "#2563eb"
_SLATE = "#0f172a"
_GREY = "#64748b"


def draw_pipeline(path: Path) -> None:
    """Stack the pipeline stages as boxes + arrows, honest and compact."""
    stages = [
        ("CSVs (any schema)", "orders, items, customers, ..."),
        ("Loader + profiles", "schema-agnostic load, joins"),
        ("Signals", "stats engine + graph engine"),
        ("Hypothesis engine", "findings + SQL/pandas evidence"),
        ("Causal testing", "OLS + controls, bootstrap CI"),
        ("Ranking", "score + diversity"),
        ("LLM decision layer", "select, order, narrate (optional)"),
        ("Investigation Report", "report.md / report.html"),
    ]
    n = len(stages)
    fig_h = max(6.4, n * 0.82)
    fig, ax = plt.subplots(figsize=(8.0, fig_h), dpi=150)
    ax.axis("off")

    x = 0.5
    box_w, box_h = 0.72, 0.115
    top = 1.6
    for i, (title, sub) in enumerate(stages):
        cy = top - i * (box_h + 0.09)
        box = FancyBboxPatch(
            (x - box_w / 2, cy - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=_SLATE,
            facecolor=_BLUE if i == 0 or i == n - 1 else "white",
            zorder=3,
        )
        ax.add_patch(box)
        tcol = "white" if i == 0 or i == n - 1 else _SLATE
        ax.text(x, cy + 0.028, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color=tcol, zorder=4)
        ax.text(x, cy - 0.032, sub, ha="center", va="center", fontsize=9,
                color=_GREY, zorder=4)
        if i > 0:
            y0 = cy + box_h / 2 + 0.085
            y1 = cy - box_h / 2 - 0.085
            ax.annotate(
                "",
                xy=(x, y0),
                xytext=(x, y1),
                arrowprops=dict(arrowstyle="-", color=_GREY, lw=1.2),
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.75)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/raw/olist", help="real dataset directory")
    ap.add_argument("--out", default=Path("docs/img"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    draw_pipeline(out / "pipeline.png")

    ds = load_dataset(args.data)
    chart_monthly_revenue(ds.fact, ds.schema, out)
    chart_quarterly_revenue(ds.fact, ds.schema, out)
    (out / "chart_monthly_revenue.png").rename(out / "example_monthly_revenue.png")
    (out / "chart_quarterly_revenue.png").rename(out / "example_quarterly_revenue.png")
    print(f"charts written to {out.resolve()}")


if __name__ == "__main__":
    main()