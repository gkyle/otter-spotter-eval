#!/usr/bin/env python3
"""Download original images referenced by an iNaturalist occurrence CSV."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

USER_AGENT = "otter-spotter/1.0 (iNaturalist image downloader)"
OBSERVATION_PATH = re.compile(r"^/observations/(\d+)/?$")
PHOTO_SIZE = re.compile(r"/(?:square|thumb|small|medium|large)\.([^/?]+)(?=$|[?])")


def is_open_license(license_code: Any) -> bool:
    """Return True if license_code is an open/Creative Commons/Public Domain license."""
    if license_code is None:
        return False
    val = str(license_code).strip().lower()
    return val not in ("", "nan", "none", "null", "0", "unlicensed")


def request(url: str, timeout: float, attempts: int = 3):
    """Open a URL, retrying temporary network/server failures."""
    for attempt in range(1, attempts + 1):
        try:
            return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)
        except HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise
            error: Exception = exc
        except URLError as exc:
            error = exc
        if attempt < attempts:
            time.sleep(2 ** (attempt - 1))
    raise error


def observation_id(value: str) -> str | None:
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    if not (host == "inaturalist.org" or host.endswith(".inaturalist.org")):
        return None
    match = OBSERVATION_PATH.match(parsed.path)
    return match.group(1) if match else None


def sniff_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return "\t"


def iter_observations(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=sniff_delimiter(path))
        if not reader.fieldnames or "occurrenceID" not in reader.fieldnames:
            raise ValueError(f"{path} has no occurrenceID column")
        seen: set[str] = set()
        for row_number, row in enumerate(reader, 2):
            obs_id = observation_id(row.get("occurrenceID") or "")
            if obs_id and obs_id not in seen:
                seen.add(obs_id)
                yield row_number, obs_id


def load_photo_metadata(
    path: Path, open_only: bool = True
) -> dict[str, list[tuple[str, str]]]:
    """Load observation -> (photo ID, original URL) from a normalized photo CSV."""
    photos: dict[str, list[tuple[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=sniff_delimiter(path))
        required = {"observation_id", "photo_id", "original_url"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        has_license = "license_code" in (reader.fieldnames or [])
        seen: set[str] = set()
        for row in reader:
            if open_only and has_license and not is_open_license(row.get("license_code")):
                continue
            observation = str(row.get("observation_id") or "").strip()
            photo_id = str(row.get("photo_id") or "").strip()
            image_url = str(row.get("original_url") or "").strip()
            if not observation or not photo_id or not image_url or photo_id in seen:
                continue
            seen.add(photo_id)
            photos.setdefault(observation, []).append((photo_id, image_url))
    return photos


def find_photos(
    obs_id: str, timeout: float, open_only: bool = True
) -> list[tuple[str, str]]:
    """Return (photo ID, original image URL) pairs from the public API."""
    url = f"https://api.inaturalist.org/v1/observations/{obs_id}"
    with request(url, timeout) as response:
        payload = json.load(response)
    results = payload.get("results") or []
    if not results:
        return []
    photos = []
    for photo in results[0].get("photos") or []:
        if open_only and not is_open_license(photo.get("license_code")):
            continue
        photo_id = str(photo["id"])
        image_url = photo.get("url")
        if not image_url:
            continue
        original_url, replacements = PHOTO_SIZE.subn(r"/original.\1", image_url, count=1)
        if replacements != 1:
            raise ValueError(f"unrecognized image URL for photo {photo_id}: {image_url}")
        photos.append((photo_id, original_url))
    return photos


def image_filename(response, photo_id: str) -> str:
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r"""filename\*?=(?:UTF-8''|)["']?([^"';]+)""", disposition, re.I)
    name = Path(unquote(match.group(1))).name if match else Path(urlparse(response.url).path).name
    if not name or "." not in name:
        extension = mimetypes.guess_extension(response.headers.get_content_type()) or ".jpg"
        name = f"{photo_id}{extension}"
    return name


def download_photo(
    photo_id: str, image_url: str, destination: Path, timeout: float
) -> tuple[Path, bool]:
    existing = sorted(destination.glob(f"{photo_id}.*")) if destination.is_dir() else []
    if existing:
        return existing[0], False
    with request(image_url, timeout) as response:
        source_name = image_filename(response, photo_id)
        suffix = Path(source_name).suffix or ".jpg"
        # iNaturalist's static URLs are all named "original", so use the photo
        # ID to prevent multiple photos in one observation overwriting each other.
        target = destination / f"{photo_id}{suffix}"
        destination.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.part")
        try:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_file", type=Path, help="CSV/TSV containing an occurrenceID column"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "images",
        help="output directory (default: sources/inat/images)",
    )
    parser.add_argument("--timeout", type=float, default=30, help="request timeout in seconds")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="delay between observations (default: 1 second, per iNaturalist guidance)",
    )
    parser.add_argument(
        "--photo-metadata",
        type=Path,
        help=(
            "photo CSV containing observation_id, photo_id, and original_url; "
            "auto-detected beside inaturalist_results.csv"
        ),
    )
    parser.add_argument(
        "--all-licenses",
        action="store_true",
        help="download photos with any license including unlicensed/all-rights-reserved (default: open licenses only)",
    )
    parser.add_argument("--limit", type=int, help="process at most this many observations")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    downloaded = skipped = failed = 0
    open_only = not args.all_licenses
    try:
        photo_metadata_path = args.photo_metadata
        if photo_metadata_path is None:
            candidate = args.csv_file.with_name("inaturalist_photos.csv")
            if candidate.is_file():
                photo_metadata_path = candidate
        photo_metadata = (
            load_photo_metadata(photo_metadata_path, open_only=open_only)
            if photo_metadata_path is not None
            else None
        )
        if photo_metadata_path is not None:
            print(f"Using saved photo metadata from {photo_metadata_path}.")
        observations = iter_observations(args.csv_file)
        for index, (row_number, obs_id) in enumerate(observations, 1):
            if args.limit is not None and index > args.limit:
                break
            used_observation_api = photo_metadata is None
            created_for_observation = False
            try:
                photos = (
                    find_photos(obs_id, args.timeout, open_only=open_only)
                    if photo_metadata is None
                    else photo_metadata.get(obs_id, [])
                )
                if not photos:
                    print(f"[{index}] observation {obs_id}: no photos found")
                for photo_id, image_url in photos:
                    target, created = download_photo(
                        photo_id, image_url, args.output / obs_id, args.timeout
                    )
                    downloaded += created
                    skipped += not created
                    created_for_observation = created_for_observation or created
                    print(f"[{index}] observation {obs_id}: {'saved' if created else 'exists'} {target}")
            except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
                failed += 1
                print(f"[{index}] row {row_number}, observation {obs_id}: {exc}", file=sys.stderr)
            if used_observation_api or created_for_observation:
                time.sleep(args.delay)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Done: {downloaded} downloaded, {skipped} already present, {failed} observations failed")
    if downloaded > 0:
        try:
            source_dir = Path(__file__).resolve().parent
            if str(source_dir) not in sys.path:
                sys.path.insert(0, str(source_dir))
            from inat import update_results_csv

            saved = update_results_csv()
            print(f"Updated image paths in {saved}.")
        except Exception as exc:
            print(f"Warning: could not update unified results.csv: {exc}", file=sys.stderr)
    return 1 if failed else 0



if __name__ == "__main__":
    raise SystemExit(main())
