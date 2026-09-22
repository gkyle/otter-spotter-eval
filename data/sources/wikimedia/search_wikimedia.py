#!/usr/bin/env python3
"""Fetch Wikimedia Commons search results and save snapshot JSON."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from wikimedia import (
    DEFAULT_OUTPUT,
    DEFAULT_QUERY,
    atomic_json,
    query_slug,
    search_wikimedia,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Search query string")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max results to fetch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Searching Wikimedia Commons for query: {args.query!r}...")
    try:
        photos = search_wikimedia(args.query, limit=args.limit)
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"search_results_{query_slug(args.query)}.json"
        atomic_json(
            path,
            {
                "query": args.query,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "photos": photos,
            },
        )
        print(f"Saved {len(photos)} result(s) to {path}.")
        return 0
    except Exception as exc:
        print(f"Error searching Wikimedia: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
