#!/usr/bin/env python3
"""Fetch all public iNaturalist Giant Otter observations without authentication.

The fetch writes a normalized observation CSV for the existing image downloader,
a photo-level CSV containing license and attribution data, a compressed JSONL
archive of the complete API objects, and a small snapshot summary.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_URL = "https://api.inaturalist.org/v1/observations"
DEFAULT_TAXON_ID = 41845  # Pteronura brasiliensis
DEFAULT_OUTPUT = Path(__file__).resolve().parent
PER_PAGE = 200
USER_AGENT = "otter-spotter/1.0 (public iNaturalist metadata fetcher)"

OBSERVATION_COLUMNS = [
    "observation_id",
    "occurrenceID",
    "uri",
    "quality_grade",
    "observation_license_code",
    "taxon_id",
    "scientific_name",
    "preferred_common_name",
    "observed_on",
    "time_observed_at",
    "observed_on_string",
    "observed_time_zone",
    "created_at",
    "updated_at",
    "observer_id",
    "observer_login",
    "observer_name",
    "latitude",
    "longitude",
    "public_positional_accuracy",
    "coordinates_obscured",
    "geoprivacy",
    "taxon_geoprivacy",
    "place_guess",
    "place_ids",
    "captive",
    "mappable",
    "description",
    "identifications_count",
    "comments_count",
    "photos_count",
    "photo_ids",
    "photo_license_codes",
    "retrieved_at",
]

PHOTO_COLUMNS = [
    "observation_id",
    "photo_id",
    "position",
    "license_code",
    "attribution",
    "url",
    "original_url",
    "width",
    "height",
    "retrieved_at",
]


class INaturalistClient:
    """Rate-limited client for public, unauthenticated API requests."""

    def __init__(self, timeout: float, interval: float, attempts: int = 3):
        self.timeout = timeout
        self.interval = interval
        self.attempts = attempts
        self.last_request: float | None = None
        self.total_results: int | None = None

    def _wait(self) -> None:
        now = time.monotonic()
        if self.last_request is not None:
            remaining = self.interval - (now - self.last_request)
            if remaining > 0:
                time.sleep(remaining)
        self.last_request = time.monotonic()

    def get(self, parameters: dict[str, object]) -> dict[str, Any]:
        url = f"{API_URL}?{urlencode(parameters)}"
        error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            self._wait()
            try:
                with urlopen(
                    Request(url, headers={"User-Agent": USER_AGENT}),
                    timeout=self.timeout,
                ) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict):
                    raise ValueError("iNaturalist returned a non-object response")
                return payload
            except HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    raise
                error = exc
                if attempt < self.attempts:
                    retry_after = exc.headers.get("Retry-After")
                    try:
                        delay = max(self.interval, float(retry_after))
                    except (TypeError, ValueError):
                        delay = 2 ** (attempt - 1)
                    time.sleep(delay)
            except (URLError, json.JSONDecodeError, TimeoutError) as exc:
                error = exc
                if attempt < self.attempts:
                    time.sleep(2 ** (attempt - 1))
        assert error is not None
        raise error


def iter_observations(
    client: INaturalistClient,
    taxon_id: int,
    limit: int | None = None,
    photo_licensed: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield every matching observation using stable ID cursor pagination."""
    seen: set[int] = set()
    id_above: int | None = None
    while limit is None or len(seen) < limit:
        parameters: dict[str, object] = {
            "taxon_id": taxon_id,
            "photos": "true",
            "per_page": PER_PAGE,
            "order_by": "id",
            "order": "asc",
        }
        if photo_licensed:
            parameters["photo_licensed"] = "true"
        if id_above is not None:
            parameters["id_above"] = id_above
        payload = client.get(parameters)
        if client.total_results is None:
            client.total_results = int(payload.get("total_results") or 0)
        results = payload.get("results") or []
        if not isinstance(results, list):
            raise ValueError("iNaturalist response has no results list")
        if not results:
            break

        page_ids: list[int] = []
        for observation in results:
            observation_id = int(observation["id"])
            page_ids.append(observation_id)
            if observation_id in seen:
                continue
            seen.add(observation_id)
            yield observation
            if limit is not None and len(seen) >= limit:
                break
        if not page_ids:
            break
        next_id = max(page_ids)
        if id_above is not None and next_id <= id_above:
            raise ValueError("iNaturalist ID pagination did not advance")
        id_above = next_id
        expected = client.total_results or len(seen)
        print(f"Fetched {len(seen)}/{expected} observation(s).", flush=True)
        if len(results) < PER_PAGE:
            break


def public_coordinates(observation: dict[str, Any]) -> tuple[object, object]:
    """Return public latitude/longitude, excluding invalid and 0,0 values."""
    coordinates = (observation.get("geojson") or {}).get("coordinates") or []
    if len(coordinates) < 2:
        return "", ""
    try:
        longitude = float(coordinates[0])
        latitude = float(coordinates[1])
    except (TypeError, ValueError):
        return "", ""
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return "", ""
    if latitude == 0 and longitude == 0:
        return "", ""
    return latitude, longitude


def original_photo_url(url: object) -> str:
    """Convert an iNaturalist resized-photo URL to its original-size URL."""
    value = str(url or "")
    for size in ("square", "thumb", "small", "medium", "large"):
        marker = f"/{size}."
        if marker in value:
            return value.replace(marker, "/original.", 1)
    return value


def observation_row(
    observation: dict[str, Any], retrieved_at: str, open_only: bool = True
) -> dict[str, object]:
    taxon = observation.get("taxon") or {}
    user = observation.get("user") or {}
    photos = observation.get("photos") or []
    if open_only:
        photos = [p for p in photos if is_open_license(p.get("license_code"))]
    latitude, longitude = public_coordinates(observation)
    observation_id = str(observation["id"])
    uri = str(
        observation.get("uri")
        or f"https://www.inaturalist.org/observations/{observation_id}"
    )
    return {
        "observation_id": observation_id,
        "occurrenceID": uri,
        "uri": uri,
        "quality_grade": observation.get("quality_grade"),
        "observation_license_code": observation.get("license_code"),
        "taxon_id": taxon.get("id"),
        "scientific_name": taxon.get("name"),
        "preferred_common_name": taxon.get("preferred_common_name"),
        "observed_on": observation.get("observed_on"),
        "time_observed_at": observation.get("time_observed_at"),
        "observed_on_string": observation.get("observed_on_string"),
        "observed_time_zone": observation.get("observed_time_zone"),
        "created_at": observation.get("created_at"),
        "updated_at": observation.get("updated_at"),
        "observer_id": user.get("id"),
        "observer_login": user.get("login"),
        "observer_name": user.get("name"),
        "latitude": latitude,
        "longitude": longitude,
        "public_positional_accuracy": observation.get("public_positional_accuracy"),
        "coordinates_obscured": observation.get("obscured"),
        "geoprivacy": observation.get("geoprivacy"),
        "taxon_geoprivacy": observation.get("taxon_geoprivacy"),
        "place_guess": observation.get("place_guess"),
        "place_ids": json.dumps(observation.get("place_ids") or []),
        "captive": observation.get("captive"),
        "mappable": observation.get("mappable"),
        "description": observation.get("description"),
        "identifications_count": observation.get("identifications_count"),
        "comments_count": observation.get("comments_count"),
        "photos_count": len(photos),
        "photo_ids": json.dumps([photo.get("id") for photo in photos]),
        "photo_license_codes": json.dumps(
            [photo.get("license_code") for photo in photos]
        ),
        "retrieved_at": retrieved_at,
    }


def photo_rows(
    observation: dict[str, Any], retrieved_at: str, open_only: bool = True
) -> list[dict[str, object]]:
    rows = []
    for position, photo in enumerate(observation.get("photos") or []):
        license_code = photo.get("license_code")
        if open_only and not is_open_license(license_code):
            continue
        dimensions = photo.get("original_dimensions") or {}
        rows.append(
            {
                "observation_id": observation["id"],
                "photo_id": photo.get("id"),
                "position": position,
                "license_code": license_code,
                "attribution": photo.get("attribution"),
                "url": photo.get("url"),
                "original_url": original_photo_url(photo.get("url")),
                "width": dimensions.get("width"),
                "height": dimensions.get("height"),
                "retrieved_at": retrieved_at,
            }
        )
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_open_license(license_code: Any) -> bool:
    """Return True if license_code is an open/Creative Commons/Public Domain license."""
    if license_code is None:
        return False
    val = str(license_code).strip().lower()
    return val not in ("", "nan", "none", "null", "0", "unlicensed")


def license_label(value: object) -> str:
    if value is None:
        return "unlicensed"
    return str(value).strip() or "unlicensed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxon-id", type=int, default=DEFAULT_TAXON_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--all-licenses",
        action="store_true",
        help="fetch observations with any license including unlicensed/all-rights-reserved (default: open licenses only)",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="minimum seconds between API requests (default: 1)",
    )
    parser.add_argument(
        "--limit", type=int, help="fetch at most N observations (for testing)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.taxon_id < 1:
        print("error: --taxon-id must be positive", file=sys.stderr)
        return 2
    if args.interval < 1.0:
        print("error: --interval must be at least 1 second", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be positive", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    observation_path = args.output / "inaturalist_results.csv"
    photo_path = args.output / "inaturalist_photos.csv"
    raw_path = args.output / "inaturalist_observations.jsonl.gz"
    snapshot_path = args.output / "snapshot.json"
    paths = [observation_path, photo_path, raw_path, snapshot_path]
    temporary = {path: path.with_name(f".{path.name}.tmp") for path in paths}

    retrieved_at = datetime.now(timezone.utc).isoformat()
    client = INaturalistClient(args.timeout, args.interval)
    observations: list[dict[str, object]] = []
    photos: list[dict[str, object]] = []
    quality_grades: Counter[str] = Counter()
    observation_licenses: Counter[str] = Counter()
    photo_licenses: Counter[str] = Counter()

    open_only = not args.all_licenses
    try:
        with gzip.open(temporary[raw_path], "wt", encoding="utf-8") as raw:
            for observation in iter_observations(
                client, args.taxon_id, args.limit, photo_licensed=open_only
            ):
                new_photos = photo_rows(observation, retrieved_at, open_only=open_only)
                if open_only and not new_photos:
                    continue
                row = observation_row(observation, retrieved_at, open_only=open_only)
                raw.write(
                    json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                observations.append(row)
                photos.extend(new_photos)
                quality_grades[license_label(row["quality_grade"])] += 1
                observation_licenses[
                    license_label(row["observation_license_code"])
                ] += 1
                photo_licenses.update(
                    license_label(photo["license_code"]) for photo in new_photos
                )

        write_csv(temporary[observation_path], OBSERVATION_COLUMNS, observations)
        write_csv(temporary[photo_path], PHOTO_COLUMNS, photos)
        summary = {
            "source": "iNaturalist public API",
            "endpoint": API_URL,
            "retrieved_at": retrieved_at,
            "query": {
                "taxon_id": args.taxon_id,
                "photos": True,
                "quality_grades": "all",
                "licenses": "all" if args.all_licenses else "open",
            },
            "api_total_results_at_start": client.total_results,
            "observation_count": len(observations),
            "photo_count": len(photos),
            "quality_grades": dict(sorted(quality_grades.items())),
            "observation_licenses": dict(sorted(observation_licenses.items())),
            "photo_licenses": dict(sorted(photo_licenses.items())),
            "files": {
                "observations": observation_path.name,
                "photos": photo_path.name,
                "raw_observations": raw_path.name,
            },
        }
        temporary[snapshot_path].write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for path in paths:
            temporary[path].replace(path)
    except (
        HTTPError,
        URLError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        print(f"error fetching iNaturalist observations: {exc}", file=sys.stderr)
        return 1

    print(
        f"Saved {len(observations)} observations and {len(photos)} photos "
        f"under {args.output}."
    )
    try:
        source_dir = Path(__file__).resolve().parent
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from inat import update_results_csv

        saved = update_results_csv()
        print(f"Updated unified dataset at {saved}.")
    except Exception as exc:
        print(f"Warning: could not update unified results.csv: {exc}", file=sys.stderr)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
