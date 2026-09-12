"""CLI entrypoint: `investigator --data data/raw/olist --out reports/`."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="investigator",
        description="Autonomous AI investigator for messy real-world datasets.",
    )
    p.add_argument("--data", required=True, help="directory containing the CSV files")
    p.add_argument("--out", default="reports", help="output directory for the report")
    p.add_argument("--no-llm", action="store_true", help="skip LLM narrative (templated)")
    p.add_argument("--limit", type=int, default=0, help="random-sample row limit for fast iteration (0=all)")
    p.add_argument("--top", type=int, default=10, help="how many findings to keep in the report")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .config import Config

    config = Config(row_limit=args.limit, top_findings=args.top)
    from .pipeline import investigate

    paths = investigate(args.data, args.out, config=config, no_llm=args.no_llm)
    print("\nOpen the report:")
    for label, p in paths.items():
        print(f"  open {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())