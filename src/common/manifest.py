#!/usr/bin/env python3
"""Manifest schema and I/O for the otter re-ID labeled dataset.

The manifest is *annotation-level*: one row per bounding box. A single image can
have multiple rows (e.g. a whole-body box and a throat-patch box, or several
individuals in one frame). Downstream steps (cropping, embedding, evaluation)
read this file as the single source of ground truth.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
# Load once so every entrypoint (Streamlit app, scripts) sees the same config
# without needing its own load_dotenv() call. Real env vars still win.
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT / "data" / "sources" / "flickr"))
from flickr_encounters import short_flickr_encounter

IMAGES_DIR = REPO_ROOT / "data" / "sources" / "inat" / "images"
DATA_DIR = REPO_ROOT / "data"
LABELS_PATH = DATA_DIR / "labels.csv"
MANIFEST_PATH = LABELS_PATH

# Crop targets we evaluate (whole-body matches MiewID training; throat = the
# distinctive Giant River Otter cream marking; throat_portrait = the subset of
# throat views where the otter faces forward and most of the throat is visible).
BODY_PARTS = ["whole_body", "throat", "throat_portrait"]
INDIVIDUAL_ID_PATTERN = re.compile(r"^otter-(\d{4})$")

ANNOTATION_COLUMNS = [
    "annotation_id",   # unique id for this bounding box
    "image_path",      # path relative to REPO_ROOT
    "observation_id",  # standardized encounter id: <source>_<id> (e.g. inat_53546627)
    "individual_id",   # ground-truth individual label
    "body_part",       # one of BODY_PARTS
    "bbox_x",          # left, in ORIGINAL image pixels
    "bbox_y",          # top, in ORIGINAL image pixels
    "bbox_w",          # width, in ORIGINAL image pixels
    "bbox_h",          # height, in ORIGINAL image pixels
    "img_width",       # original image width (for validation/scaling)
    "img_height",      # original image height
    "split",           # assigned later (query/gallery); blank while labeling
    "notes",
    "created_at",      # ISO-8601 timestamp
]

COLUMNS = [
    "annotation_id",   # unique id for this bounding box
    "image_path",      # path relative to REPO_ROOT
    "observation_id",  # standardized encounter id: <source>_<id> (e.g. inat_53546627)
    "latitude",        # authoritative latitude from results.csv
    "longitude",       # authoritative longitude from results.csv
    "location_source", # inat, flickr, wikimedia, upload_exif, map, blank when unavailable
    "individual_id",   # ground-truth individual label
    "body_part",       # one of BODY_PARTS
    "bbox_x",          # left, in ORIGINAL image pixels
    "bbox_y",          # top, in ORIGINAL image pixels
    "bbox_w",          # width, in ORIGINAL image pixels
    "bbox_h",          # height, in ORIGINAL image pixels
    "img_width",       # original image width (for validation/scaling)
    "img_height",      # original image height
    "split",           # assigned later (query/gallery); blank while labeling
    "notes",
    "created_at",      # ISO-8601 timestamp
]


def is_valid_location(latitude: object, longitude: object) -> bool:
    """Check if latitude and longitude are valid decimal coordinates and not (0, 0) Null Island."""
    try:
        lat = float(latitude)
        lon = float(longitude)
        return (-90 <= lat <= 90) and (-180 <= lon <= 180) and not (lat == 0 and lon == 0)
    except (TypeError, ValueError):
        return False


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def load_manifest(
    path: Path = LABELS_PATH, infer_locations: bool = True
) -> pd.DataFrame:
    """Load the manifest/labels, attaching authoritative location from results.csv."""
    if not path.exists():
        legacy_manifest = DATA_DIR / "manifest.csv"
        if legacy_manifest.exists():
            path = legacy_manifest
        else:
            return empty_manifest()
    df = pd.read_csv(path, dtype={"individual_id": str, "observation_id": str})

    # Attach authoritative location from data/results.csv
    results_csv = path.parent / "results.csv"
    if not results_csv.exists():
        results_csv = DATA_DIR / "results.csv"

    if results_csv.is_file():
        try:
            from common.sources import load_sources_results
            res_df = load_sources_results()
        except Exception:
            res_df = pd.read_csv(results_csv, dtype=str)

        valid_res = res_df[res_df.apply(lambda r: is_valid_location(r.get("latitude"), r.get("longitude")), axis=1)]

        loc_by_img: dict[str, tuple[str, str, str]] = {}
        for _, r in valid_res.iterrows():
            img_p = str(r.get("image_path") or "").strip()
            if img_p:
                loc_by_img[img_p] = (str(r["latitude"]), str(r["longitude"]), str(r.get("location_source") or ""))

        loc_by_enc: dict[str, tuple[str, str, str]] = {}
        for _, r in valid_res.iterrows():
            enc_p = str(r.get("encounter_id") or "").strip()
            if enc_p and enc_p not in loc_by_enc:
                loc_by_enc[enc_p] = (str(r["latitude"]), str(r["longitude"]), str(r.get("location_source") or ""))

        lats = []
        lons = []
        loc_sources = []
        for _, row in df.iterrows():
            img_p = str(row.get("image_path") or "").strip()
            obs_p = str(row.get("observation_id") or "").strip()
            loc_info = loc_by_img.get(img_p) or loc_by_enc.get(obs_p)
            if loc_info:
                lats.append(loc_info[0])
                lons.append(loc_info[1])
                loc_sources.append(loc_info[2])
            else:
                lats.append("")
                lons.append("")
                loc_sources.append("")

        df["latitude"] = lats
        df["longitude"] = lons
        df["location_source"] = loc_sources

    # Guarantee all expected columns exist even for older files.
    for column in COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[COLUMNS]
    if infer_locations:
        try:
            from location.geodistance import infer_encounter_locations

            df = infer_encounter_locations(df)
        except Exception:
            pass
    return df


def filter_manifest_by_license(
    df: pd.DataFrame, repo_root: Path = REPO_ROOT
) -> pd.DataFrame:
    """Filter manifest rows to retain only images with open / ML-usable licenses."""
    if df.empty or "image_path" not in df.columns:
        return df

    results_csv = repo_root / "data" / "results.csv"
    lic_by_path: dict[str, str] = {}
    if results_csv.is_file():
        try:
            res_df = pd.read_csv(
                results_csv, dtype=str, usecols=["image_path", "license"]
            )
            lic_by_path = dict(
                zip(res_df["image_path"].dropna(), res_df["license"].fillna(""))
            )
        except Exception:
            pass

    def is_open(row: pd.Series) -> bool:
        path = str(row.get("image_path", ""))
        lic = str(lic_by_path.get(path, "")).strip()
        parts = path.split("/")
        source = parts[2] if len(parts) > 2 else ""
        if source in ("upload", "wikimedia"):
            return True
        elif source == "flickr":
            try:
                return int(lic) in set(range(1, 17))
            except (ValueError, TypeError):
                return False
        elif source in ("inat"):
            return lic.lower() not in ("", "nan", "none", "null", "0", "unlicensed")
        return False

    mask = df.apply(is_open, axis=1)
    return df[mask].copy()


def save_manifest(df: pd.DataFrame, path: Path = LABELS_PATH) -> None:
    """Persist the manifest annotations atomically (write to temp then replace)."""
    df = df.copy()
    flickr = df["image_path"].astype(str).str.startswith(
        "data/sources/flickr/images/"
    )
    df.loc[flickr, "observation_id"] = (
        df.loc[flickr, "observation_id"]
        .astype(str)
        .map(short_flickr_encounter)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    save_cols = [c for c in ANNOTATION_COLUMNS if c in df.columns]
    df[save_cols].to_csv(temp, index=False)
    temp.replace(path)


def next_individual_id(df: pd.DataFrame) -> str:
    """Return the next unused sequential ``otter-NNNN`` individual ID."""
    numbers = [
        int(match.group(1))
        for value in df["individual_id"].dropna().astype(str).unique()
        if (match := INDIVIDUAL_ID_PATTERN.fullmatch(value))
    ]
    next_number = max(numbers, default=0) + 1
    if next_number > 9999:
        raise ValueError("The four-digit individual ID sequence is exhausted")
    return f"otter-{next_number:04d}"


def observation_id_for(image_path: Path) -> str:
    """Return standardized <source>_<id> encounter identifier."""
    from common.sources import load_sources_results

    relative_path = str(image_path.resolve().relative_to(REPO_ROOT))
    results = load_sources_results()
    matches = results[results["image_path"].astype(str) == relative_path]
    if matches.empty:
        raise ValueError(f"Image is missing from results.csv: {relative_path}")
    return str(matches.iloc[0]["encounter_id"])


def relative_image_path(image_path: Path) -> str:
    return str(image_path.resolve().relative_to(REPO_ROOT))


def list_images(images_dir: Path = IMAGES_DIR) -> list[Path]:
    """All pool images, sorted by observation then filename for stable paging."""
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        (p for p in images_dir.rglob("*") if p.suffix.lower() in exts),
        key=lambda p: (p.parent.name, p.name),
    )
