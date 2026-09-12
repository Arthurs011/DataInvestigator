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
_DARK = "#0f172a"
_GREY = "#64748b"
_BAND = "#eff6ff"


def draw_pipeline(path: Path) -> None:
    """Stack the pipeline stages as boxes + arrows, honest and compact."""
    stages = [
        ("Raw data", "directory of CSVs"),
        ("Loader + profiles", "schema-agnostic intake, join discovery"),
        ("Statistical + graph signals", "period changes, segments, geo lift, outliers"),
        ("Hypothesis engine", "findings + SQL and pandas evidence"),
        ("Causal testing", "OLS + controls, bootstrap CI"),
        ("Ranking", "score + kind diversity"),
        ("LLM decision layer", "select, order, narrate  (optional)"),
        ("Investigation Report", "report.md  /  report.html"),
    ]
    # rows 1..5 (loader..ranking) run deterministically; everything before the LLM
    n = len(stages)
    box_h, gap = 0.9, 0.9
    y0, y1 = 0.8, 0.8 + n * (box_h + gap) - gap
    fig, ax = plt.subplots(figsize=(7.6, y1 + 1.0), dpi=150)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, y1 + 1.0)

    top = y1 + box_h / 2
    cx = 5.0

    # deterministic band behind rows 1..5
    band_bottom = top - 6 * (box_h + gap) + box_h / 2 + 0.55
    band_top = top - 1 * (box_h + gap) + box_h / 2 + 0.55
    ax.add_patch(
        FancyBboxPatch(
            (0.9, band_bottom),
            8.2,
            band_top - band_bottom,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor=_BAND, edgecolor="#bfdbfe", linewidth=1.0, zorder=1,
        )
    )
    ax.text(
        1.15, (band_top + band_bottom) / 2, "deterministic",
        rotation=90, va="center", ha="center", fontsize=10, color=_BLUE, zorder=2,
    )

    for i, (title, sub) in enumerate(stages):
        cy = top - i * (box_h + gap)
        first = i == 0
        last = i == n - 1
        llm = i == n - 2
        face = _DARK if last else (_BLUE if first else "#ffffff")
        edge = _BLUE if (first or llm) else _DARK
        text_col = "white" if (first or last) else _DARK
        box = FancyBboxPatch(
            (cx - 3.4, cy - box_h / 2),
            6.8, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=face,
            linestyle="--" if llm else "-",
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(cx, cy + 0.22 if sub else cy, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color=text_col, zorder=4)
        ax.text(cx, cy - 0.26, sub, ha="center", va="center", fontsize=9.5,
                color=_GREY, zorder=4)
        if i > 0:
            cy_prev = top - (i - 1) * (box_h + gap)
            ax.annotate(
                "",
                xy=(cx, cy + box_h / 2),
                xytext=(cx, cy_prev - box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=_GREY, lw=1.6, mutation_scale=22),
                zorder=2,
            )
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