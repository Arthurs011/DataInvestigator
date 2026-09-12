"""Download the Olist Brazilian E-Commerce dataset (9 CSVs) from Hugging Face.

Serves as the real, heterogeneous multi-table dataset for the investigator.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

HF_REPO = "bulutttt/olist-raw-data"
FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def download(outdir: Path, force: bool = False) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fname in FILES:
        dest = outdir / fname
        if dest.exists() and not force:
            paths.append(dest)
            continue
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{fname}"
        print(f"  downloading {fname} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "data-investigator/0.1"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        paths.append(dest)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the Olist e-commerce dataset.")
    parser.add_argument("--outdir", type=Path, default=Path("data/raw/olist"))
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)
    paths = download(args.outdir, force=args.force)
    print(f"\n'{len(paths)}' files ready in {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())