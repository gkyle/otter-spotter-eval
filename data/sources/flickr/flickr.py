#!/usr/bin/env python3
"""Shared Flickr API, image, and metadata helpers."""

from __future__ import annotations

import csv
import fcntl
import json
import mimetypes
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from PIL import ExifTags, Image

from flickr_encounters import flickr_encounter

API_URL = "https://www.flickr.com/services/rest/"
DEFAULT_QUERY = "giant river otter"
DEFAULT_OUTPUT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"
USER_AGENT = "otter-spotter/1.0 (Flickr research image downloader)"
PER_PAGE = 500
SEARCH_RESULT_FIELDS = (
    "id",
    "owner",
    "secret",
    "ispublic",
    "license",
    "dateupload",
    "datetaken",
    "ownername",
    "latitude",
    "longitude",
    "accuracy",
)

CSV_COLUMNS = [*SEARCH_RESULT_FIELDS, "encounter_id", "image_url"]

# Flickr license IDs 1..16 are open/Creative Commons/Public Domain licenses; 0 is All Rights Reserved.
OPEN_LICENSE_IDS = set(range(1, 17))
OPEN_LICENSES = ",".join(str(i) for i in range(1, 17))


def is_open_license(license_id: Any) -> bool:
    """Return True if license_id is an open/Creative Commons/Public Domain Flickr license (1..16)."""
    try:
        return int(license_id) in OPEN_LICENSE_IDS
    except (ValueError, TypeError):
        return False


class FlickrAPIError(RuntimeError):
    """An error returned in a Flickr JSON response."""


@dataclass
class RateLimiter:
    """Enforce a minimum interval between starts of all HTTP requests."""

    interval: float = 1.0
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep
    last_request: float | None = None
    state_path: Path | None = None

    def wait(self) -> None:
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with self.state_path.open("a+", encoding="utf-8") as state:
                fcntl.flock(state, fcntl.LOCK_EX)
                state.seek(0)
                try:
                    previous = float(state.read().strip())
                except ValueError:
                    previous = None
                now = self.clock()
                if previous is not None:
                    remaining = self.interval - (now - previous)
                    if remaining > 0:
                        self.sleeper(remaining)
                requested_at = self.clock()
                state.seek(0)
                state.truncate()
                state.write(f"{requested_at:.9f}\n")
                state.flush()
                self.last_request = requested_at
                fcntl.flock(state, fcntl.LOCK_UN)
            return
        now = self.clock()
        if self.last_request is not None:
            remaining = self.interval - (now - self.last_request)
            if remaining > 0:
                self.sleeper(remaining)
        self.last_request = self.clock()


class FlickrClient:
    def __init__(
        self,
        api_key: str,
        timeout: float,
        limiter: RateLimiter,
        rate_limit_cooldown: float = 3600.0,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.limiter = limiter
        self.rate_limit_cooldown = rate_limit_cooldown

    def open(self, url: str, attempts: int = 3):
        """Open one URL, rate-limiting and retrying temporary failures."""
        error: Exception | None = None
        for attempt in range(1, attempts + 1):
            self.limiter.wait()
            try:
                return urlopen(
                    Request(url, headers={"User-Agent": USER_AGENT}),
                    timeout=self.timeout,
                )
            except HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    raise
                error = exc
                if exc.code == 429 and attempt < attempts:
                    retry_after = exc.headers.get("Retry-After")
                    try:
                        cooldown = max(self.rate_limit_cooldown, float(retry_after))
                    except (TypeError, ValueError):
                        cooldown = self.rate_limit_cooldown
                    print(
                        f"Flickr rate limit reached; cooling down for "
                        f"{cooldown:.0f} seconds.",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(cooldown)
                    continue
            except URLError as exc:
                error = exc
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
        assert error is not None
        raise error

    def call(self, method: str, **parameters: Any) -> dict[str, Any]:
        query = {
            "method": method,
            "api_key": self.api_key,
            "format": "json",
            "nojsoncallback": 1,
            **parameters,
        }
        with self.open(f"{API_URL}?{urlencode(query)}") as response:
            payload = json.load(response)
        if payload.get("stat") != "ok":
            raise FlickrAPIError(
                f"{method}: Flickr error {payload.get('code')}: {payload.get('message')}"
            )
        return payload


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("_content", ""))
    return "" if value is None else str(value)


def compact_search_photo(photo: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields needed for provenance and subsequent API requests."""
    return {field: photo[field] for field in SEARCH_RESULT_FIELDS if field in photo}


def search_photos(
    client: FlickrClient,
    query: str,
    license: str | None = OPEN_LICENSES,
) -> list[dict[str, Any]]:
    """Fetch all Flickr-visible pages for a text query (API maximum: 4,000)."""
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    page = 1
    total_pages = 1
    while True:
        params: dict[str, Any] = {
            "text": query,
            "media": "photos",
            "content_types": "0",
            "safe_search": 1,
            "sort": "date-posted-asc",
            "extras": "license,date_upload,date_taken,owner_name,geo",
            "per_page": PER_PAGE,
            "page": page,
        }
        if license:
            params["license"] = license
        payload = client.call("flickr.photos.search", **params)
        photos = payload["photos"]
        for photo in photos.get("photo", []):
            photo_id = str(photo["id"])
            if photo_id not in seen:
                seen.add(photo_id)
                results.append(compact_search_photo(photo))
        # Search totals can fluctuate between requests. Never reduce the number
        # of pages after Flickr has advertised a higher one during this run.
        total_pages = max(
            total_pages,
            min(int(photos.get("pages", 1)), 8),  # Flickr search cap: 4,000.
        )
        print(
            f"search page {page}/{total_pages}: {len(results)} unique result(s)",
            flush=True,
        )
        if page >= total_pages or not photos.get("photo"):
            break
        page += 1
    return results


def largest_size(payload: dict[str, Any]) -> dict[str, Any]:
    sizes = [
        size
        for size in payload.get("sizes", {}).get("size", [])
        if size.get("source") and size.get("media", "photo") == "photo"
    ]
    if not sizes:
        raise ValueError("Flickr returned no downloadable photo sizes")
    return max(
        sizes,
        key=lambda size: (
            int(size.get("width") or 0) * int(size.get("height") or 0),
            int(size.get("width") or 0),
            int(size.get("height") or 0),
        ),
    )


def image_suffix(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix and len(suffix) <= 6:
        return suffix
    return mimetypes.guess_extension(content_type.split(";", 1)[0]) or ".jpg"


def download_image(
    client: FlickrClient, photo_id: str, url: str, image_dir: Path
) -> tuple[Path, bool]:
    """Download a largest-size image atomically; return path and created flag."""
    existing = sorted(image_dir.glob(f"{photo_id}.*")) if image_dir.exists() else []
    if existing:
        return existing[0], False
    with client.open(url) as response:
        suffix = image_suffix(response.url, response.headers.get_content_type())
        target = image_dir / f"{photo_id}{suffix}"
        image_dir.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.part")
        try:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target, True


def existing_image(image_dir: Path, photo_id: str) -> Path | None:
    """Return an already-downloaded image for a Flickr photo ID."""
    matches = sorted(image_dir.glob(f"{photo_id}.*")) if image_dir.exists() else []
    return matches[0] if matches else None


def rational(value: Any) -> float:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]) / float(value[1])
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def coordinate(value: Any, reference: str) -> float | None:
    try:
        degrees, minutes, seconds = (rational(component) for component in value)
        result = degrees + minutes / 60 + seconds / 3600
        return -result if reference.upper() in {"S", "W"} else result
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def embedded_exif(path: Path) -> dict[str, Any]:
    """Extract useful EXIF fields, particularly embedded GPS coordinates."""
    result: dict[str, Any] = {}
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            result["camera_make"] = text(exif.get(271)).strip()
            result["camera_model"] = text(exif.get(272)).strip()
            result["datetime_original"] = text(exif.get(36867)).strip()
            try:
                gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
            except (AttributeError, KeyError, TypeError):
                gps = {}
    except (OSError, ValueError):
        return result

    latitude = coordinate(gps.get(2), text(gps.get(1))) if gps else None
    longitude = coordinate(gps.get(4), text(gps.get(3))) if gps else None
    altitude = None
    if gps and gps.get(6) is not None:
        try:
            altitude = rational(gps[6]) * (-1 if gps.get(5) == 1 else 1)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    result.update(latitude=latitude, longitude=longitude, altitude=altitude)
    return result


def api_exif_summary(payload: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Return normalized label values and all Flickr API GPS tags."""
    values: dict[str, str] = {}
    gps: dict[str, str] = {}
    for item in payload.get("photo", {}).get("exif", []):
        label = str(item.get("label") or item.get("tag") or "")
        value = text(item.get("clean")) or text(item.get("raw"))
        values[label.lower()] = value
        if str(item.get("tagspace", "")).upper() == "GPS":
            gps[label] = value
    return values, gps


def api_exif_coordinates(payload: dict[str, Any]) -> dict[str, float | None]:
    """Convert Flickr's raw EXIF GPS tag values to decimal coordinates."""
    tags: dict[str, str] = {}
    for item in payload.get("photo", {}).get("exif", []):
        if str(item.get("tagspace", "")).upper() != "GPS":
            continue
        tags[str(item.get("tag", ""))] = text(item.get("raw"))

    def components(value: str) -> list[float]:
        return [rational(component.strip()) for component in value.split(",")]

    latitude = longitude = altitude = None
    try:
        latitude = coordinate(components(tags["2"]), tags.get("1", "N"))
        longitude = coordinate(components(tags["4"]), tags.get("3", "E"))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    if "6" in tags:
        try:
            altitude = rational(tags["6"]) * (-1 if tags.get("5") == "1" else 1)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return {"latitude": latitude, "longitude": longitude, "altitude": altitude}


def location(info: dict[str, Any], search: dict[str, Any]) -> dict[str, str]:
    value = info.get("photo", {}).get("location") or {}
    return {
        "latitude": text(value.get("latitude") or search.get("latitude")),
        "longitude": text(value.get("longitude") or search.get("longitude")),
        "accuracy": text(value.get("accuracy") or search.get("accuracy")),
        "context": text(value.get("context") or search.get("context")),
        "place_id": text(value.get("place_id") or search.get("place_id")),
    }


def photo_row(
    query: str,
    search: dict[str, Any],
    metadata: dict[str, Any],
    image_path: Path | None,
    output_dir: Path,
    error: str = "",
) -> dict[str, Any]:
    info = metadata.get("info", {}).get("photo", {})
    size = metadata.get("selected_size", {})
    exif_values, api_gps = api_exif_summary(metadata.get("exif", {}))
    api_coordinates = api_exif_coordinates(metadata.get("exif", {}))
    file_exif = embedded_exif(image_path) if image_path and image_path.exists() else {}
    embedded_has_gps = (
        file_exif.get("latitude") is not None
        and file_exif.get("longitude") is not None
    )
    api_has_gps = (
        api_coordinates.get("latitude") is not None
        and api_coordinates.get("longitude") is not None
    )
    exif_coordinates = file_exif if embedded_has_gps else api_coordinates
    geo = location(metadata.get("info", {}), search)
    owner = info.get("owner", {})
    dates = info.get("dates", {})
    tags = info.get("tags", {}).get("tag", [])
    photo_id = str(search.get("id", info.get("id", "")))
    owner_id = text(owner.get("nsid") or search.get("owner"))
    date_taken = text(dates.get("taken") or search.get("datetaken"))
    return {
        "query": query,
        "photo_id": photo_id,
        "encounter_id": flickr_encounter(owner_id, date_taken, photo_id),
        "owner_id": owner_id,
        "owner_username": text(owner.get("username") or search.get("ownername")),
        "owner_realname": text(owner.get("realname")),
        "title": text(info.get("title") or search.get("title")),
        "description": text(info.get("description") or search.get("description")),
        "tags": " ".join(text(tag.get("raw") or tag.get("_content")) for tag in tags),
        "machine_tags": text(search.get("machine_tags")),
        "license_id": text(info.get("license") or search.get("license")),
        "date_uploaded": text(dates.get("posted") or search.get("dateupload")),
        "date_taken": date_taken,
        "last_update": text(dates.get("lastupdate") or search.get("lastupdate")),
        "views": text(info.get("views") or search.get("views")),
        "media": text(info.get("media") or search.get("media")),
        "photopage_url": text(info.get("urls", {}).get("url", [{}])[0]),
        "image_path": str(image_path.relative_to(output_dir)) if image_path else "",
        "image_url": text(size.get("source")),
        "image_size_label": text(size.get("label")),
        "image_width": text(size.get("width")),
        "image_height": text(size.get("height")),
        "image_bytes": image_path.stat().st_size if image_path and image_path.exists() else "",
        "flickr_latitude": geo["latitude"],
        "flickr_longitude": geo["longitude"],
        "flickr_accuracy": geo["accuracy"],
        "flickr_context": geo["context"],
        "flickr_place_id": geo["place_id"],
        "exif_latitude": exif_coordinates.get("latitude", ""),
        "exif_longitude": exif_coordinates.get("longitude", ""),
        "exif_altitude": exif_coordinates.get("altitude", ""),
        "exif_gps_source": "embedded" if embedded_has_gps else "flickr_api" if api_has_gps else "",
        "exif_datetime_original": file_exif.get("datetime_original")
        or exif_values.get("date and time (original)", ""),
        "camera_make": file_exif.get("camera_make") or exif_values.get("make", ""),
        "camera_model": file_exif.get("camera_model") or exif_values.get("model", ""),
        "api_gps_tags_json": json.dumps(api_gps, ensure_ascii=False, sort_keys=True),
        "downloaded_at": metadata.get("downloaded_at", ""),
        "error": error,
    }


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def query_slug(query: str) -> str:
    """Return a stable filename component for a Flickr search query."""
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    return slug or "query"


def existing_csv_rows(path: Path, excluding_query: str) -> list[dict[str, str]]:
    """Preserve rows belonging to other searches when refreshing one query."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("query") != excluding_query
        ]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read the flattened Flickr metadata cache if it exists."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_photo_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Upsert Flickr rows by photo ID under a lock shared by downloaders."""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = {row.get("id", ""): row for row in read_csv_rows(path) if row.get("id")}
        merged.update({str(row["id"]): row for row in rows})
        atomic_csv(path, list(merged.values()))
        fcntl.flock(lock, fcntl.LOCK_UN)


def atomic_query_csv(
    path: Path, query: str, query_rows: list[dict[str, Any]]
) -> None:
    """Merge one query's rows under a lock shared by concurrent downloaders."""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        preserved = existing_csv_rows(path, query)
        atomic_csv(path, preserved + query_rows)
        fcntl.flock(lock, fcntl.LOCK_UN)


def env_value(path: Path, name: str) -> str:
    """Read one variable from a simple dotenv file without changing the process."""
    if not path.exists():
        return ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value
    return ""


SOURCE_NAME = "flickr"


def translate_to_results(repo_root: Path | None = None) -> list[dict[str, str]]:
    """Translate Flickr metadata into unified results rows."""
    import pandas as pd

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    source_dir = repo_root / "data" / "sources" / "flickr"
    flickr_results_path = source_dir / "flickr_results.csv"
    flickr_images_dir = source_dir / "images"

    def _normalize_val(val: object) -> str:
        if pd.isna(val) or val is None:
            return ""
        s = str(val).strip()
        return "" if s.lower() in ("nan", "none", "null") else s

    flickr_filtered_path = source_dir / "flickr_results_filtered.csv"
    flickr_duplicates_path = source_dir / "flickr_duplicates.csv"
    labels_path = repo_root / "data" / "labels.csv"

    excluded_pids: set[str] = set()
    if flickr_duplicates_path.is_file():
        try:
            df_dup = pd.read_csv(flickr_duplicates_path, dtype=str)
            excluded_pids = {
                Path(str(r.get("flickr_path") or "")).stem
                for _, r in df_dup.iterrows()
            }
        except Exception:
            pass

    qualified_pids: set[str] = set()
    if flickr_filtered_path.is_file():
        try:
            df_filt = pd.read_csv(flickr_filtered_path, dtype=str)
            for _, fr in df_filt.iterrows():
                try:
                    tp_ok = int(float(fr.get("throat_portrait_count") or 0)) > 0
                except (ValueError, TypeError):
                    tp_ok = False
                try:
                    reg_ok = int(float(fr.get("is_target_region") or 0)) > 0
                except (ValueError, TypeError):
                    reg_ok = False
                if tp_ok or reg_ok:
                    pid = _normalize_val(fr.get("id"))
                    if pid and pid not in excluded_pids:
                        qualified_pids.add(pid)
        except Exception:
            pass

    # Ensure all Flickr photos that have annotations in labels.csv are always retained
    if labels_path.is_file():
        try:
            df_lbl = pd.read_csv(labels_path, dtype=str, usecols=["image_path"])
            fl_lbl = df_lbl[df_lbl["image_path"].str.startswith("data/sources/flickr", na=False)]
            for p in fl_lbl["image_path"]:
                qualified_pids.add(Path(p).stem)
        except Exception:
            pass

    rows: list[dict[str, str]] = []
    if flickr_results_path.is_file():
        df_flickr = pd.read_csv(flickr_results_path, dtype=str)
        for _, fr in df_flickr.iterrows():
            pid = _normalize_val(fr.get("id"))
            if not pid or (qualified_pids and pid not in qualified_pids):
                continue
            img_p = flickr_images_dir / f"{pid}.jpg"
            if not img_p.is_file():
                continue
            enc = _normalize_val(fr.get("encounter_id"))
            if enc.startswith("fl-"):
                encounter_id = f"flickr_{enc[3:]}"
            elif enc.startswith("flickr_"):
                encounter_id = enc
            elif enc:
                encounter_id = f"flickr_{enc}"
            else:
                encounter_id = f"flickr_{pid}"

            img_p = flickr_images_dir / f"{pid}.jpg"
            img_rel = str(img_p.relative_to(repo_root)) if img_p.is_file() else ""
            owner = _normalize_val(fr.get("owner"))

            lat_str = _normalize_val(fr.get("latitude"))
            lon_str = _normalize_val(fr.get("longitude"))
            try:
                if float(lat_str) == 0 and float(lon_str) == 0:
                    lat_str = lon_str = ""
            except (ValueError, TypeError):
                pass
            loc_src = SOURCE_NAME if lat_str and lon_str else ""

            rows.append({
                "source": SOURCE_NAME,
                "encounter_id": encounter_id,
                "photo_id": pid,
                "image_path": img_rel,
                "image_url": _normalize_val(fr.get("image_url")),
                "page_url": f"https://www.flickr.com/photos/{owner}/{pid}" if owner else "",
                "observed_at": _normalize_val(fr.get("datetaken")),
                "observer": _normalize_val(fr.get("ownername")),
                "latitude": lat_str,
                "longitude": lon_str,
                "location_source": loc_src,
                "locality": "",
                "license": _normalize_val(fr.get("license")),
                "quality_grade": "",
            })

    return rows


def update_results_csv(repo_root: Path | None = None) -> Path:
    """Update flickr records in data/results.csv while preserving other sources."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    records = translate_to_results(repo_root)
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from common.sources import update_source_records
    return update_source_records(SOURCE_NAME, records, target_path=repo_root / "data" / "results.csv")


