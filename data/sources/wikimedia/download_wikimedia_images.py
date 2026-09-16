#!/usr/bin/env python3
"""Download images and metadata from saved Wikimedia search snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

from wikimedia import (
    DEFAULT_OUTPUT,
    atomic_csv,
    compact_wikimedia_photo,
    download_image,
    existing_image,
    read_csv_rows,
)

DEFAULT_SNAPSHOT = DEFAULT_OUTPUT / "search_results_giant-river-otter.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "search_results",
        type=Path,
        nargs="*",
        default=[DEFAULT_SNAPSHOT],
        help="search snapshot JSON files created by search_wikimedia.py",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output directory")
    parser.add_argument("--limit", type=int, help="download at most N search results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be positive", file=sys.stderr)
        return 2

    # Load and combine all specified snapshots
    all_pages: dict[str, dict] = {}
    for snapshot_path in args.search_results:
        if not snapshot_path.is_file():
            print(f"warning: snapshot file not found: {snapshot_path}", file=sys.stderr)
            continue
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            photos = snapshot.get("photos", [])
            for page in photos:
                pageid = str(page.get("pageid") or "").strip()
                if pageid and pageid not in all_pages:
                    all_pages[pageid] = page
        except Exception as exc:
            print(f"error reading {snapshot_path}: {exc}", file=sys.stderr)
            return 2

    if not all_pages:
        print("No search results found to download.", file=sys.stderr)
        return 0

    selected_pages = list(all_pages.values())
    if args.limit is not None:
        selected_pages = selected_pages[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output / "wikimedia_results.csv"
    existing_rows = {row.get("id"): row for row in read_csv_rows(csv_path)}

    rows: list[dict[str, str]] = []
    downloaded = existing = failures = 0

    print(f"Processing {len(selected_pages)} Wikimedia image(s)...")

    for position, page in enumerate(selected_pages, start=1):
        row = compact_wikimedia_photo(page)
        photo_id = row["id"]
        image_url = row["image_url"]

        if not image_url:
            print(f"[{position}/{len(selected_pages)}] {photo_id}: missing image URL", file=sys.stderr)
            failures += 1
            continue

        img_path = existing_image(image_dir, photo_id)
        if img_path is not None:
            existing += 1
            print(f"[{position}/{len(selected_pages)}] {photo_id}: exists {img_path.name}")
            rows.append(row)
            atomic_csv(csv_path, rows)
            continue

        try:
            img_path, created = download_image(image_url, photo_id, image_dir)
            if created:
                downloaded += 1
                print(f"[{position}/{len(selected_pages)}] {photo_id}: downloaded {img_path.name}")
            else:
                existing += 1
                print(f"[{position}/{len(selected_pages)}] {photo_id}: exists {img_path.name}")
        except (HTTPError, URLError, OSError, ValueError) as exc:
            failures += 1
            print(f"[{position}/{len(selected_pages)}] {photo_id}: download error {exc}", file=sys.stderr)
            continue

        rows.append(row)
        atomic_csv(csv_path, rows)

    print(f"\nDone: {downloaded} downloaded, {existing} existing, {failures} failed, {len(rows)} CSV row(s).")
    print(f"Saved results to {csv_path}")
    if downloaded > 0:
        try:
            source_dir = Path(__file__).resolve().parent
            if str(source_dir) not in sys.path:
                sys.path.insert(0, str(source_dir))
            from wikimedia import update_results_csv

            saved = update_results_csv()
            print(f"Updated results at {saved}.")
        except Exception as exc:
            print(f"Warning: could not update unified results.csv: {exc}", file=sys.stderr)
    return 1 if failures else 0



if __name__ == "__main__":
    raise SystemExit(main())
