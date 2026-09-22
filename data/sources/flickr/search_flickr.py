#!/usr/bin/env python3
"""Fetch a Flickr search and save its complete result snapshot."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

from flickr import (
    DEFAULT_ENV_FILE,
    DEFAULT_OUTPUT,
    DEFAULT_QUERY,
    OPEN_LICENSES,
    FlickrAPIError,
    FlickrClient,
    RateLimiter,
    atomic_json,
    env_value,
    query_slug,
    search_photos,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--licenses",
        default=OPEN_LICENSES,
        help="comma-separated Flickr license IDs to search (default: open licenses 1..16)",
    )
    parser.add_argument(
        "--all-licenses",
        action="store_true",
        help="search photos with any license including All Rights Reserved",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.25)
    parser.add_argument("--rate-limit-cooldown", type=float, default=3600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval < 1.0:
        print("error: --interval must be at least 1.0 second", file=sys.stderr)
        return 2
    if args.rate_limit_cooldown < 60:
        print("error: --rate-limit-cooldown must be at least 60 seconds", file=sys.stderr)
        return 2
    api_key = os.environ.get("FLICKR_API_KEY", "").strip() or env_value(
        args.env_file, "FLICKR_API_KEY"
    ).strip()
    if not api_key:
        print(f"error: set FLICKR_API_KEY or add it to {args.env_file}", file=sys.stderr)
        return 2

    client = FlickrClient(
        api_key,
        args.timeout,
        RateLimiter(
            args.interval,
            clock=time.time,
            state_path=DEFAULT_OUTPUT / ".request_rate_limit",
        ),
        args.rate_limit_cooldown,
    )
    license_filter = None if args.all_licenses else args.licenses
    try:
        photos = search_photos(client, args.query, license=license_filter)
        path = args.output / f"search_results_{query_slug(args.query)}.json"
        atomic_json(
            path,
            {
                "query": args.query,
                "licenses": "all" if args.all_licenses else args.licenses,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "photos": photos,
            },
        )
    except (HTTPError, URLError, FlickrAPIError, OSError, ValueError) as exc:
        print(f"error searching Flickr: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {len(photos)} result(s) to {path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
