#!/usr/bin/env python3
"""Process uploaded otter photos into identities and timed encounters."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import ExifTags, Image

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT / "src"))
from common.manifest import REPO_ROOT

UPLOAD_DIR = REPO_ROOT / "data/sources/upload"
UPLOAD_OTTERS_DIR = UPLOAD_DIR / "otters"
UPLOAD_RESULTS_PATH = UPLOAD_DIR / "upload_results.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ENCOUNTER_GAP = timedelta(hours=1)
UPLOAD_OWNER = ""
RESULT_COLUMNS = [
    "image_path",
    "owner",
    "individual_id",
    "source_folder",
    "captured_at",
    "encounter_id",
    "latitude",
    "longitude",
    "location_source",
]
_FILENAME_DATETIME = re.compile(r"(?:^|_)(\d{8})[_-]?(\d{6})(?:\d{3})?(?:\D|$)")


def _rational(value: Any) -> float:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def _coordinate(value: Any, reference: object) -> float | None:
    try:
        degrees, minutes, seconds = (_rational(component) for component in value)
        coordinate = degrees + minutes / 60 + seconds / 3600
        ref = (
            reference.decode("ascii", errors="ignore")
            if isinstance(reference, bytes)
            else str(reference)
        ).upper()
        return -coordinate if ref in {"S", "W"} else coordinate
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _valid_coordinates(latitude: float | None, longitude: float | None) -> bool:
    return (
        latitude is not None
        and longitude is not None
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
        and (latitude != 0 or longitude != 0)
    )


def _parse_exif_datetime(value: object) -> datetime | None:
    text = str(value or "").strip().rstrip("\x00")
    for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            pass
    return None


def _filename_datetime(path: Path) -> datetime | None:
    match = _FILENAME_DATETIME.search(path.stem)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def image_metadata(path: Path) -> tuple[datetime | None, float | None, float | None]:
    """Return capture time and valid decimal EXIF coordinates for an image."""
    captured_at = None
    latitude = longitude = None
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            for tag in (36867, 36868, 306):
                captured_at = _parse_exif_datetime(exif.get(tag))
                if captured_at is not None:
                    break
            try:
                gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
            except (AttributeError, KeyError, TypeError):
                gps = {}
            if gps:
                latitude = _coordinate(gps.get(2), gps.get(1))
                longitude = _coordinate(gps.get(4), gps.get(3))
    except (OSError, ValueError):
        pass
    captured_at = captured_at or _filename_datetime(path)
    if not _valid_coordinates(latitude, longitude):
        latitude = longitude = None
    return captured_at, latitude, longitude


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "otter"


def _encounter_id(individual_id: str, folder: str, start: datetime | None) -> str:
    key = f"{individual_id}\0{folder}\0{start.isoformat() if start else 'undated'}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    timestamp = start.strftime("%Y%m%dT%H%M%S") if start else "undated"
    return f"upload-{_slug(individual_id)}-{timestamp}-{digest}"


def build_upload_index(root: Path = UPLOAD_OTTERS_DIR) -> pd.DataFrame:
    """Scan uploads and group same-folder photos separated by at most one hour."""
    records: list[dict[str, object]] = []
    if not root.is_dir():
        return pd.DataFrame(columns=RESULT_COLUMNS)

    existing_owners: dict[str, str] = {}
    if UPLOAD_RESULTS_PATH.is_file():
        try:
            df_res = pd.read_csv(UPLOAD_RESULTS_PATH, dtype=str)
            for _, r in df_res.iterrows():
                p = str(r.get("image_path") or "").strip()
                o = str(r.get("owner") or "").strip()
                if p and o:
                    existing_owners[p] = o
        except Exception:
            pass

    pending_csv = UPLOAD_DIR / "pending_uploads.csv"
    if pending_csv.is_file():
        try:
            df_pending = pd.read_csv(pending_csv, dtype=str)
            for _, r in df_pending.iterrows():
                p = str(r.get("image_path") or "").strip()
                o = str(r.get("owner") or "").strip()
                if p and o:
                    existing_owners[p] = o
        except Exception:
            pass

    for individual_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        individual_id = individual_dir.name
        images = sorted(
            path
            for path in individual_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        by_folder: dict[Path, list[dict[str, object]]] = {}
        for image_path in images:
            captured_at, latitude, longitude = image_metadata(image_path)
            by_folder.setdefault(image_path.parent, []).append(
                {
                    "path": image_path,
                    "captured_at_value": captured_at,
                    "latitude_value": latitude,
                    "longitude_value": longitude,
                }
            )

        for folder, folder_records in sorted(by_folder.items(), key=lambda item: str(item[0])):
            dated = sorted(
                (row for row in folder_records if row["captured_at_value"] is not None),
                key=lambda row: (row["captured_at_value"], str(row["path"])),
            )
            undated = sorted(
                (row for row in folder_records if row["captured_at_value"] is None),
                key=lambda row: str(row["path"]),
            )
            encounters: list[list[dict[str, object]]] = []
            for row in dated:
                if (
                    not encounters
                    or row["captured_at_value"]
                    - encounters[-1][-1]["captured_at_value"]
                    > ENCOUNTER_GAP
                ):
                    encounters.append([])
                encounters[-1].append(row)
            encounters.extend([[row] for row in undated])

            folder_name = str(folder.relative_to(individual_dir)) or "."
            for encounter in encounters:
                start = encounter[0]["captured_at_value"]
                encounter_id = _encounter_id(individual_id, folder_name, start)
                located = [
                    row
                    for row in encounter
                    if row["latitude_value"] is not None
                    and row["longitude_value"] is not None
                ]
                encounter_latitude = (
                    float(pd.Series([row["latitude_value"] for row in located]).median())
                    if located
                    else None
                )
                encounter_longitude = (
                    float(pd.Series([row["longitude_value"] for row in located]).median())
                    if located
                    else None
                )
                for row in encounter:
                    image_path = row["path"]
                    rel_img_path = str(image_path.relative_to(REPO_ROOT))
                    has_image_gps = row["latitude_value"] is not None
                    records.append(
                        {
                            "image_path": rel_img_path,
                            "owner": existing_owners.get(rel_img_path, UPLOAD_OWNER),
                            "individual_id": individual_id,
                            "source_folder": folder_name,
                            "captured_at": (
                                row["captured_at_value"].isoformat(timespec="seconds")
                                if row["captured_at_value"]
                                else ""
                            ),
                            "encounter_id": encounter_id,
                            "latitude": encounter_latitude if located else "",
                            "longitude": encounter_longitude if located else "",
                            "location_source": (
                                "upload_exif"
                                if has_image_gps
                                else "upload_encounter_exif" if located else ""
                            ),
                        }
                    )
    return pd.DataFrame(records, columns=RESULT_COLUMNS).sort_values(
        ["individual_id", "captured_at", "image_path"], kind="stable"
    )


def save_upload_index(frame: pd.DataFrame, path: Path = UPLOAD_RESULTS_PATH) -> None:
    """Atomically save a generated upload index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=UPLOAD_OTTERS_DIR)
    parser.add_argument("--output", type=Path, default=UPLOAD_RESULTS_PATH)
    args = parser.parse_args()
    frame = build_upload_index(args.root)
    save_upload_index(frame, args.output)
    mapped = pd.to_numeric(frame["latitude"], errors="coerce").notna().sum()
    print(
        f"Indexed {len(frame)} images for {frame['individual_id'].nunique()} otters "
        f"across {frame['encounter_id'].nunique()} encounters; "
        f"{mapped} images mapped"
    )


if __name__ == "__main__":
    main()
