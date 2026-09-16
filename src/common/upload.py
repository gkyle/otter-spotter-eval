"""Upload directory indexing and EXIF helper functions."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import ExifTags, Image, ImageOps

from common.manifest import LABELS_PATH, REPO_ROOT

UPLOAD_DIR = REPO_ROOT / "data" / "sources" / "upload"
UPLOAD_IMAGES_DIR = UPLOAD_DIR / "images"
UPLOAD_SESSIONS_DIR = UPLOAD_DIR / "sessions"
UPLOAD_RESULTS_PATH = UPLOAD_DIR / "upload_results.csv"
PENDING_CSV = UPLOAD_DIR / "upload_pending.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}

_DATE_TIME_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")
_FILENAME_DATETIME = re.compile(
    r"(20\d{2})[-_]?([0-1]\d)[-_]?([0-3]\d)[-_ T]?([0-2]\d)[-_]?([0-5]\d)[-_]?([0-5]\d)"
)


def _parse_exif_datetime(value: object) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    for fmt in _DATE_TIME_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def _coordinate(rational_tuple: object, ref: object) -> float | None:
    if not rational_tuple:
        return None
    try:
        degrees, minutes, seconds = rational_tuple
        decimal = float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0
        return -decimal if str(ref).upper() in {"S", "W"} else decimal
    except (TypeError, ValueError, ZeroDivisionError):
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
    if captured_at is None:
        match = _FILENAME_DATETIME.search(path.stem)
        if match:
            try:
                captured_at = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
            except ValueError:
                pass
    if latitude is not None and longitude is not None:
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180 and not (latitude == 0 and longitude == 0)):
            latitude = longitude = None
    return captured_at, latitude, longitude


def transpose_image_file(path: Path) -> bool:
    """If an image file has non-default EXIF orientation, transpose its pixels to upright,
    clear the EXIF orientation tag, and overwrite the file in place. Returns True if transposed."""
    try:
        with Image.open(path) as im:
            orient = im.getexif().get(274)
            if not orient or orient == 1:
                return False
            transposed = ImageOps.exif_transpose(im)
            exif = transposed.getexif()
            format_name = im.format or ("JPEG" if path.suffix.lower() in (".jpg", ".jpeg") else None)
            save_kwargs: dict[str, object] = {}
            if exif:
                save_kwargs["exif"] = exif
            if format_name == "JPEG":
                save_kwargs["quality"] = 95
            temp_file = path.with_name(f".{path.name}.tmp")
            transposed.convert("RGB" if format_name == "JPEG" else transposed.mode).save(
                temp_file, format=format_name, **save_kwargs
            )
            temp_file.replace(path)
            return True
    except Exception:
        return False


def copy_and_transpose_image(src: Path, dest: Path) -> None:
    """Copy an image from src to dest, transposing to upright orientation if EXIF orientation is present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            orient = im.getexif().get(274)
            if orient and orient != 1:
                transposed = ImageOps.exif_transpose(im)
                exif = transposed.getexif()
                format_name = im.format or ("JPEG" if dest.suffix.lower() in (".jpg", ".jpeg") else None)
                save_kwargs: dict[str, object] = {}
                if exif:
                    save_kwargs["exif"] = exif
                if format_name == "JPEG":
                    save_kwargs["quality"] = 95
                temp_dest = dest.with_name(f".{dest.name}.tmp")
                transposed.convert("RGB" if format_name == "JPEG" else transposed.mode).save(
                    temp_dest, format=format_name, **save_kwargs
                )
                temp_dest.replace(dest)
                return
    except Exception:
        pass
    shutil.copy2(src, dest)


def build_upload_index() -> pd.DataFrame:
    """Index all committed upload images in UPLOAD_IMAGES_DIR, merging cached metadata from upload_results.csv and labels.csv."""
    existing_map: dict[str, dict[str, str]] = {}
    if UPLOAD_RESULTS_PATH.is_file():
        try:
            df_old = pd.read_csv(UPLOAD_RESULTS_PATH, dtype=str)
            for _, row in df_old.iterrows():
                p = str(row.get("image_path") or "")
                if p:
                    existing_map[p] = {k: ("" if pd.isna(v) else str(v)) for k, v in row.items()}
        except Exception:
            pass

    manifest_map: dict[str, dict[str, str]] = {}
    manifest_path = LABELS_PATH if LABELS_PATH.is_file() else (REPO_ROOT / "data" / "manifest.csv")
    if manifest_path.is_file():
        try:
            df_m = pd.read_csv(manifest_path, dtype=str)
            for _, mrow in df_m.iterrows():
                p = str(mrow.get("image_path") or "")
                if p and str(p).startswith("data/sources/upload/images/"):
                    if p not in manifest_map:
                        manifest_map[p] = {
                            "latitude": str(mrow.get("latitude") or "") if not pd.isna(mrow.get("latitude")) else "",
                            "longitude": str(mrow.get("longitude") or "") if not pd.isna(mrow.get("longitude")) else "",
                            "location_source": str(mrow.get("location_source") or "") if not pd.isna(mrow.get("location_source")) else "",
                            "individual_id": str(mrow.get("individual_id") or "") if not pd.isna(mrow.get("individual_id")) else "",
                            "encounter_id": str(mrow.get("observation_id") or "") if not pd.isna(mrow.get("observation_id")) else "",
                        }
        except Exception:
            pass

    records = []
    if UPLOAD_IMAGES_DIR.is_dir():
        for path in sorted(UPLOAD_IMAGES_DIR.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                rel = str(path.relative_to(REPO_ROOT))
                if rel in existing_map:
                    rec = dict(existing_map[rel])
                else:
                    m_data = manifest_map.get(rel, {})
                    captured_at, lat, lon = image_metadata(path)
                    parent_name = path.parent.name
                    default_enc = parent_name if (parent_name.startswith("upload-") or parent_name.startswith("upload_")) else f"upload-{parent_name}"
                    enc_id = m_data.get("encounter_id") or default_enc
                    if not (enc_id.startswith("upload-") or enc_id.startswith("upload_")):
                        enc_id = f"upload-{enc_id}"
                    rec = {
                        "image_path": rel,
                        "owner": "",
                        "individual_id": m_data.get("individual_id", ""),
                        "source_folder": ".",
                        "captured_at": captured_at.isoformat() if captured_at else "",
                        "encounter_id": enc_id,
                        "latitude": m_data.get("latitude") or (str(lat) if lat is not None else ""),
                        "longitude": m_data.get("longitude") or (str(lon) if lon is not None else ""),
                        "location_source": m_data.get("location_source") or ("exif" if lat is not None else ""),
                    }

                if rel in manifest_map:
                    m_data = manifest_map[rel]
                    for k in ("latitude", "longitude", "location_source", "individual_id", "encounter_id"):
                        if not rec.get(k) and m_data.get(k):
                            rec[k] = m_data[k]
                if not (rec.get("encounter_id", "").startswith("upload-") or rec.get("encounter_id", "").startswith("upload_")):
                    rec["encounter_id"] = f"upload-{rec.get('encounter_id', '')}"
                records.append(rec)

    seen = {r.get("image_path") for r in records}
    for p, rec in existing_map.items():
        if p not in seen and (REPO_ROOT / p).is_file():
            rec_copy = dict(rec)
            if not (rec_copy.get("encounter_id", "").startswith("upload-") or rec_copy.get("encounter_id", "").startswith("upload_")):
                rec_copy["encounter_id"] = f"upload-{rec_copy.get('encounter_id', '')}"
            records.append(rec_copy)
            seen.add(p)

    return pd.DataFrame(records, columns=[
        "image_path", "owner", "individual_id", "source_folder", "captured_at",
        "encounter_id", "latitude", "longitude", "location_source"
    ])



def save_upload_index(frame: pd.DataFrame, path: Path = UPLOAD_RESULTS_PATH) -> None:
    """Atomically save a generated upload index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
