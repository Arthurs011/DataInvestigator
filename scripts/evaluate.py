"""Ground-truth evaluation harness.

Builds two synthetic datasets — one clean, one with three known injected
anomalies (see scripts/synth_data.py) — runs the full pipeline on each, and
checks that every injected anomaly is recovered by the injected run and is
absent from the clean run.

Usage:
    uv run python scripts/evaluate.py [--orders 700] [--customers 4000] [--seed 42] [--top 10]

Exit code 0 iff all anomalies are recovered (recall = 3/3) and none ghosted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.synth_data import ANOMALIES, GEO_CAT, GEO_STATE, SPIKE_MONTH, build_dataset
from investigator.config import Config
from investigator.models import Finding
from investigator.pipeline import run_pipeline

DROP_Q = "2020Q3"


def match(f: Finding, key: str) -> bool:
    """True when finding f is the pipeline's way of reporting anomaly `key`."""
    h = f.headline
    if key == "A_quarterly_revenue_drop":
        return (
            f.kind == "time_change"
            and f.target_metric == "revenue"
            and h.get("period") == DROP_Q
            and h.get("prev_period") == "2020Q2"
            and h.get("delta_pct", 0) < -0.15
        )
    if key == "B_state_category_overindex":
        return f.kind == "graph" and h.get("state") == GEO_STATE and h.get("category") == GEO_CAT
    if key == "C_monthly_spike":
        return (
            f.kind == "time_anomaly"
            and h.get("period") == SPIKE_MONTH
            and h.get("z_score", 0) >= 2.0
        )
    return False


def evaluate(seed: int, orders: int, customers: int, top: int, out: Path) -> int:
    clean_dir = build_dataset(out / "clean", seed=seed, orders_per_quarter=orders, n_customers=customers, anomalies=False)
    inj_dir = build_dataset(out / "injected", seed=seed, orders_per_quarter=orders, n_customers=customers, anomalies=True)
    config = Config(top_findings=top, seed=seed)

    print(f"\n=== pipeline on CLEAN data ===")
    clean = run_pipeline(clean_dir, config, no_llm=True)
    print(f"\n=== pipeline on INJECTED data ===")
    injected = run_pipeline(inj_dir, config, no_llm=True)

    ok = True
    rows = []
    for key, meta in ANOMALIES.items():
        hit_inj = next((f for f in injected.findings if match(f, key)), None)
        hit_clean = next((f for f in clean.findings if match(f, key)), None)
        rec = hit_inj is not None
        gho = hit_clean is not None
        if not rec or gho:
            ok = False
        rows.append((key, meta["describe"], rec, gho, hit_inj))

    print("\n┌─────────────────────────────────────┬────────────┬────────────┬──────┬─────────┐")
    print("│ anomaly                              │ recovered  │ ghosted    │ rank │ score    │")
    print("├─────────────────────────────────────┼────────────┼────────────┼──────┼─────────┤")
    for key, desc, rec, gho, hit in rows:
        rank = hit.rank if hit else 0
        score = f"{hit.score:.3f}" if hit else "-"
        print(f"│ {desc:<35} │ {'YES' if rec else 'NO ':<10} │ {'NO ' if not gho else 'YES':<10} │ {rank:>4} │ {score:>7} │")
    print("└─────────────────────────────────────┴────────────┴────────────┴──────┴─────────┘")

    with (out / "eval.md").open("w") as fh:
        fh.write("# Ground-truth evaluation\n\n")
        fh.write(f"seed={seed} orders/quarter={orders} customers={customers} top={top}\n\n")
        for key, desc, rec, gho, hit in rows:
            fh.write(f"## {key} — {desc}\n")
            fh.write(f"- recovered: {'YES' if rec else 'NO'} (rank {hit.rank if hit else '-'})\n")
            fh.write(f"- ghosted on clean data: {'YES' if gho else 'NO'}\n")
            if hit:
                fh.write(f"- title: {hit.title}\n")
                fh.write(f"- headline: {hit.headline}\n")
                fh.write(f"- evidence verified: {any(e.verified for e in hit.evidence)}\n")
            fh.write("\n")
    print(f"\nDetails written to {out / 'eval.md'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--orders", type=int, default=700)
    ap.add_argument("--customers", type=int, default=4000)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=".eval")
    args = ap.parse_args()
    rc = evaluate(args.seed, args.orders, args.customers, args.top, Path(args.out))
    print(f"\nEvaluation {'PASSED' if rc == 0 else 'FAILED'}: recall must be 3/3 with zero ghosting")
    return rc


if __name__ == "__main__":
    sys.exit(main())