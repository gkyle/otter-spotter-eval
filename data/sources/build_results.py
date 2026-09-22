#!/usr/bin/env python3
"""Offline script to consolidate all data sources into data/results.csv."""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.sources import COMMON_COLUMNS, RESULTS_CSV, save_sources_results

SOURCES_DIR = REPO_ROOT / "data" / "sources"
for s in ("inat", "flickr", "wikimedia", "upload"):
    p = SOURCES_DIR / s
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import inat
import flickr
import wikimedia
import upload


def build_results(repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    """Collect translated records from all data sources into a single DataFrame."""
    rows: list[dict[str, str]] = []
    rows.extend(inat.translate_to_results(repo_root))
    rows.extend(flickr.translate_to_results(repo_root))
    rows.extend(wikimedia.translate_to_results(repo_root))
    rows.extend(upload.translate_to_results(repo_root))
    return pd.DataFrame(rows, columns=COMMON_COLUMNS)


def main() -> int:
    print(f"Rebuilding consolidated {RESULTS_CSV}...")
    df = build_results()
    saved = save_sources_results(df, RESULTS_CSV)
    print(f"Successfully saved {saved} with {len(df)} rows.")
    print("Breakdown by source:")
    print(df["source"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
