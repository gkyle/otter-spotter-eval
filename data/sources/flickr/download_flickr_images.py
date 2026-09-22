#!/usr/bin/env python3
"""Download images and metadata from a saved Flickr search snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

from flickr import (
    DEFAULT_ENV_FILE,
    DEFAULT_OUTPUT,
    FlickrClient,
    RateLimiter,
    atomic_photo_csv,
    compact_search_photo,
    download_image,
    env_value,
    existing_image,
    is_open_license,
    largest_size,
    read_csv_rows,
)
from flickr_encounters import flickr_encounter

DEFAULT_SEARCH_RESULTS = DEFAULT_OUTPUT / "search_results_giant-river-otter.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "search_results",
        type=Path,
        nargs="?",
        default=DEFAULT_SEARCH_RESULTS,
        help="search snapshot created by search_flickr.py",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--limit", type=int, help="download at most N search results")
    parser.add_argument(
        "--all-licenses",
        action="store_true",
        help="download photos with any license including All Rights Reserved (default: open licenses only)",
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
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be positive", file=sys.stderr)
        return 2
    if args.rate_limit_cooldown < 60:
        print("error: --rate-limit-cooldown must be at least 60 seconds", file=sys.stderr)
        return 2
    try:
        snapshot = json.loads(args.search_results.read_text(encoding="utf-8"))
        query = str(snapshot["query"])
        photos = snapshot["photos"]
        if not isinstance(photos, list):
            raise ValueError("photos must be a list")
        photos = [compact_search_photo(photo) for photo in photos]
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"error reading {args.search_results}: {exc}", file=sys.stderr)
        return 2

    if not args.all_licenses:
        open_photos = [photo for photo in photos if is_open_license(photo.get("license"))]
        excluded = len(photos) - len(open_photos)
        if excluded:
            print(
                f"Filtered out {excluded} photo(s) with non-open licenses "
                f"(e.g. All Rights Reserved). {len(open_photos)} photo(s) remaining.",
                flush=True,
            )
        photos = open_photos

    args.output.mkdir(parents=True, exist_ok=True)
    selected = photos[: args.limit] if args.limit is not None else photos
    image_dir = args.output / "images"
    needs_download = any(
        existing_image(image_dir, str(search["id"])) is None for search in selected
    )
    client: FlickrClient | None = None
    if needs_download:
        api_key = os.environ.get("FLICKR_API_KEY", "").strip() or env_value(
            args.env_file, "FLICKR_API_KEY"
        ).strip()
        if not api_key:
            print(
                f"error: set FLICKR_API_KEY or add it to {args.env_file}",
                file=sys.stderr,
            )
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
    csv_path = args.output / "flickr_results.csv"
    rows_by_photo = {row.get("id"): row for row in read_csv_rows(csv_path)}
    rows: list[dict[str, str]] = []
    failures = downloaded = existing = 0
    for position, search in enumerate(selected, start=1):
        photo_id = str(search["id"])
        image_path = existing_image(image_dir, photo_id)
        try:
            row = compact_search_photo(search)
            row["encounter_id"] = flickr_encounter(
                str(row.get("owner", "")),
                str(row.get("datetaken", "")),
                photo_id,
            )
            if image_path is not None:
                row["image_url"] = rows_by_photo.get(photo_id, {}).get("image_url", "")
                existing += 1
                print(
                    f"[{position}/{len(selected)}] {photo_id}: exists {image_path.name}",
                    flush=True,
                )
                rows.append(row)
                atomic_photo_csv(csv_path, [row])
                continue
            assert client is not None
            sizes = client.call("flickr.photos.getSizes", photo_id=photo_id)
            size = largest_size(sizes)
            image_path, created = download_image(
                client, photo_id, str(size["source"]), image_dir
            )
            downloaded += int(created)
            existing += int(not created)
            row["image_url"] = str(size["source"])
            print(
                f"[{position}/{len(selected)}] {photo_id}: "
                f"{'downloaded' if created else 'exists'} {image_path.name}",
                flush=True,
            )
        except (
            HTTPError,
            URLError,
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            failures += 1
            print(f"[{position}/{len(selected)}] {photo_id}: {exc}", file=sys.stderr)
            continue
        rows.append(row)
        atomic_photo_csv(csv_path, [row])

    print(
        f"Done: {downloaded} downloaded, {existing} existing, "
        f"{failures} failed, {len(rows)} CSV row(s)."
    )
    if downloaded > 0:
        try:
            source_dir = Path(__file__).resolve().parent
            if str(source_dir) not in sys.path:
                sys.path.insert(0, str(source_dir))
            from flickr import update_results_csv

            saved = update_results_csv()
            print(f"Updated results at {saved}.")
        except Exception as exc:
            print(f"Warning: could not update unified results.csv: {exc}", file=sys.stderr)
    return 1 if failures else 0



if __name__ == "__main__":
    raise SystemExit(main())
