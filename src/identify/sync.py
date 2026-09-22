#!/usr/bin/env python3
"""Synchronization between annotations, image crops, and re-identification embeddings.

Ensures that additions, modifications, and deletions made in the labeling tool or
manifest stay atomically in sync with:
  - data/crops/<body_part>/<annotation_id>.jpg
  - data/crops/crops.csv
  - data/embeddings/miewid-msv3-finetuned/<body_part>/embeddings.npy
  - data/embeddings/miewid-msv3-finetuned/<body_part>/index.csv
  - data/embeddings/miewid-msv3-finetuned/<body_part>/locations.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import (  # noqa: E402
    BODY_PARTS,
    COLUMNS,
    DATA_DIR,
    LABELS_PATH,
    REPO_ROOT,
    is_valid_location,
    load_manifest,
)
from common.web import QUERY_EMBEDDINGS_DIR  # noqa: E402
from identify.crop import CROPS_DIR, CROPS_INDEX, JPEG_QUALITY, crop_with_margin  # noqa: E402
from identify.miewid.model import (  # noqa: E402
    EMBEDDING_DIM,
    embed_images,
    get_device,
    load_model,
)
from location.geodistance import write_location_cache  # noqa: E402

FINETUNED_CHECKPOINT = DATA_DIR / "models" / "miewid-msv3-otters" / "best.pt"

INDEX_COLUMNS = [
    "annotation_id",
    "individual_id",
    "observation_id",
    "latitude",
    "longitude",
    "location_source",
    "body_part",
    "crop_path",
    "split",
]

_CACHED_MODEL = None
_CACHED_DEVICE = None


def get_identity_model(
    checkpoint: Path | None = None,
    prefer_gpu: bool = True,
):
    """Load and cache the MiewID model in memory for fast inference."""
    global _CACHED_MODEL, _CACHED_DEVICE
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _CACHED_DEVICE

    ckpt = checkpoint if checkpoint is not None else FINETUNED_CHECKPOINT
    ckpt_path = ckpt if (ckpt and Path(ckpt).is_file()) else None
    _CACHED_DEVICE = get_device(prefer_gpu=prefer_gpu)
    _CACHED_MODEL = load_model(_CACHED_DEVICE, ckpt_path)
    return _CACHED_MODEL, _CACHED_DEVICE


def _save_atomic_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV atomically via a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    df.to_csv(temp, index=False)
    temp.replace(path)


def _save_atomic_npy(arr: np.ndarray, path: Path) -> None:
    """Save NumPy array to .npy atomically via a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with open(temp, "wb") as f:
        np.save(f, arr)
    temp.replace(path)


def _rel_repo_path(path: Path | str) -> str:
    """Return repo-relative path string if possible, else str(path)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def create_crop(
    image_path: str | Path,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
    body_part: str,
    annotation_id: str,
    margin: float = 0.0,
    output_dir: Path = CROPS_DIR,
) -> Path | None:
    """Crop bounding box at native resolution and save to output_dir/<body_part>/<aid>.jpg."""
    source_path = Path(image_path)
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path

    if not source_path.is_file():
        print(f"Warning: source image {source_path} not found for crop {annotation_id}", file=sys.stderr)
        return None

    try:
        with Image.open(source_path) as img:
            rgb_img = img.convert("RGB")
            crop = crop_with_margin(rgb_img, int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h), margin)
            target_dir = output_dir / body_part
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / f"{annotation_id}.jpg"
            temp_file = target_dir / f".{annotation_id}.jpg.tmp"
            crop.save(temp_file, "JPEG", quality=JPEG_QUALITY)
            temp_file.replace(target_file)
            return target_file
    except Exception as exc:
        print(f"Error creating crop {annotation_id} from {source_path}: {exc}", file=sys.stderr)
        return None


def sync_annotation_addition(
    rows: Sequence[dict] | pd.DataFrame,
    prune_stale: bool = True,
    crops_dir: Path = CROPS_DIR,
    embeddings_dir: Path = QUERY_EMBEDDINGS_DIR,
    checkpoint: Path | None = None,
) -> None:
    """Generate crops and compute embeddings for newly added annotations."""
    if isinstance(rows, pd.DataFrame):
        row_dicts = rows.to_dict(orient="records")
    else:
        row_dicts = [dict(r) for r in rows]

    if not row_dicts:
        return

    # Optionally prune stale entries first to ensure clean state
    if prune_stale:
        prune_stale_crops_and_embeddings(manifest=None, crops_dir=crops_dir, embeddings_dir=embeddings_dir)

    crops_index_path = crops_dir / "crops.csv"
    existing_crops_df = (
        pd.read_csv(crops_index_path, dtype=str)
        if crops_index_path.is_file()
        else pd.DataFrame(columns=COLUMNS + ["crop_path"])
    )

    new_crop_records = []
    crops_to_embed_by_part: dict[str, list[tuple[dict, Path]]] = {bp: [] for bp in BODY_PARTS}

    for r in row_dicts:
        aid = str(r["annotation_id"])
        body_part = str(r.get("body_part") or "whole_body")
        image_path = str(r.get("image_path") or "")
        bbox_x = int(float(r.get("bbox_x") or 0))
        bbox_y = int(float(r.get("bbox_y") or 0))
        bbox_w = int(float(r.get("bbox_w") or 0))
        bbox_h = int(float(r.get("bbox_h") or 0))

        crop_file = create_crop(
            image_path=image_path,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            body_part=body_part,
            annotation_id=aid,
            margin=0.0,
            output_dir=crops_dir,
        )

        crop_rel_path = _rel_repo_path(crop_file) if crop_file else f"data/crops/{body_part}/{aid}.jpg"

        record = {col: str(r.get(col, "") if r.get(col) is not None else "") for col in COLUMNS}
        record["crop_path"] = crop_rel_path
        new_crop_records.append(record)

        if crop_file and crop_file.is_file():
            crops_to_embed_by_part.setdefault(body_part, []).append((record, crop_file))

    # Append to crops.csv (deduplicating against existing)
    new_aids = {rec["annotation_id"] for rec in new_crop_records}
    if not existing_crops_df.empty:
        existing_crops_df = existing_crops_df[~existing_crops_df["annotation_id"].astype(str).isin(new_aids)]
    updated_crops_df = pd.concat([existing_crops_df, pd.DataFrame(new_crop_records)], ignore_index=True)
    _save_atomic_csv(updated_crops_df, crops_index_path)

    # Compute embeddings and update embeddings index per body part
    for body_part, items in crops_to_embed_by_part.items():
        if not items:
            continue

        model, device = get_identity_model(checkpoint=checkpoint)
        pil_images = []
        valid_items = []
        for rec, path in items:
            try:
                with Image.open(path) as img:
                    pil_images.append(img.convert("RGB"))
                valid_items.append(rec)
            except Exception as exc:
                print(f"Failed to open crop {path} for embedding: {exc}", file=sys.stderr)

        if not pil_images:
            continue

        try:
            new_vectors = embed_images(model, pil_images, device, batch_size=32, normalize=True)
        finally:
            for img in pil_images:
                img.close()

        part_dir = embeddings_dir / body_part
        part_dir.mkdir(parents=True, exist_ok=True)
        npy_path = part_dir / "embeddings.npy"
        index_path = part_dir / "index.csv"

        if npy_path.is_file() and index_path.is_file():
            existing_embeddings = np.load(npy_path)
            existing_index = pd.read_csv(index_path, dtype=str)
            # Remove any matching aids if already present
            existing_aids = existing_index["annotation_id"].astype(str)
            remove_mask = existing_aids.isin(new_aids)
            if remove_mask.any():
                keep_mask = ~remove_mask
                existing_embeddings = existing_embeddings[keep_mask.to_numpy()]
                existing_index = existing_index[keep_mask].reset_index(drop=True)
        else:
            existing_embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            existing_index = pd.DataFrame(columns=INDEX_COLUMNS)

        combined_embeddings = (
            np.vstack([existing_embeddings, new_vectors])
            if len(existing_embeddings)
            else new_vectors
        )

        new_index_rows = [
            {col: item.get(col, "") for col in INDEX_COLUMNS}
            for item in valid_items
        ]
        combined_index = pd.concat([existing_index, pd.DataFrame(new_index_rows)], ignore_index=True)

        _save_atomic_npy(combined_embeddings.astype(np.float32), npy_path)
        _save_atomic_csv(combined_index, index_path)
        write_location_cache(combined_index, part_dir)


def sync_annotation_update(
    annotation_id: str,
    new_individual_id: str,
    new_body_part: str,
    crops_dir: Path = CROPS_DIR,
    embeddings_dir: Path = QUERY_EMBEDDINGS_DIR,
    checkpoint: Path | None = None,
) -> None:
    """Update metadata and transfer crops/embeddings if body part changed."""
    aid = str(annotation_id)
    crops_index_path = crops_dir / "crops.csv"
    if not crops_index_path.is_file():
        return

    crops_df = pd.read_csv(crops_index_path, dtype=str)
    match = crops_df[crops_df["annotation_id"].astype(str) == aid]
    if match.empty:
        return

    row = match.iloc[0]
    old_body_part = str(row.get("body_part") or "")
    old_individual_id = str(row.get("individual_id") or "")

    # Case 1: body part unchanged, only individual_id changed
    if new_body_part == old_body_part:
        if new_individual_id != old_individual_id:
            crops_df.loc[crops_df["annotation_id"].astype(str) == aid, "individual_id"] = new_individual_id
            _save_atomic_csv(crops_df, crops_index_path)

            part_dir = embeddings_dir / old_body_part
            index_path = part_dir / "index.csv"
            if index_path.is_file():
                idx_df = pd.read_csv(index_path, dtype=str)
                idx_df.loc[idx_df["annotation_id"].astype(str) == aid, "individual_id"] = new_individual_id
                _save_atomic_csv(idx_df, index_path)
        return

    # Case 2: body part changed
    # 1. Move crop file on disk
    old_crop_file = crops_dir / old_body_part / f"{aid}.jpg"
    new_crop_dir = crops_dir / new_body_part
    new_crop_dir.mkdir(parents=True, exist_ok=True)
    new_crop_file = new_crop_dir / f"{aid}.jpg"

    if old_crop_file.is_file():
        old_crop_file.replace(new_crop_file)

    new_rel_crop_path = _rel_repo_path(new_crop_file)

    # 2. Update crops.csv
    row_idx = match.index[0]
    crops_df.loc[row_idx, "individual_id"] = new_individual_id
    crops_df.loc[row_idx, "body_part"] = new_body_part
    crops_df.loc[row_idx, "crop_path"] = new_rel_crop_path
    _save_atomic_csv(crops_df, crops_index_path)

    # 3. Transfer embedding vector between body parts
    old_part_dir = embeddings_dir / old_body_part
    old_npy_path = old_part_dir / "embeddings.npy"
    old_index_path = old_part_dir / "index.csv"

    new_part_dir = embeddings_dir / new_body_part
    new_part_dir.mkdir(parents=True, exist_ok=True)
    new_npy_path = new_part_dir / "embeddings.npy"
    new_index_path = new_part_dir / "index.csv"

    vector = None
    index_record = None

    if old_npy_path.is_file() and old_index_path.is_file():
        old_embeddings = np.load(old_npy_path)
        old_index = pd.read_csv(old_index_path, dtype=str)
        match_idx = old_index.index[old_index["annotation_id"].astype(str) == aid].tolist()
        if match_idx:
            idx = match_idx[0]
            vector = old_embeddings[idx : idx + 1]
            index_record = old_index.iloc[idx].to_dict()
            # Remove from old
            updated_old_embeddings = np.delete(old_embeddings, idx, axis=0)
            updated_old_index = old_index.drop(index=idx).reset_index(drop=True)
            _save_atomic_npy(updated_old_embeddings.astype(np.float32), old_npy_path)
            _save_atomic_csv(updated_old_index, old_index_path)
            write_location_cache(updated_old_index, old_part_dir)

    # If vector was not found in old index, re-embed from the new crop file
    if vector is None:
        if new_crop_file.is_file():
            model, device = get_identity_model(checkpoint=checkpoint)
            with Image.open(new_crop_file) as img:
                rgb_img = img.convert("RGB")
                vector = embed_images(model, [rgb_img], device, normalize=True)
            index_record = {col: str(crops_df.loc[row_idx].get(col, "") or "") for col in INDEX_COLUMNS}

    if vector is not None and index_record is not None:
        index_record["individual_id"] = new_individual_id
        index_record["body_part"] = new_body_part
        index_record["crop_path"] = new_rel_crop_path

        if new_npy_path.is_file() and new_index_path.is_file():
            new_embeddings = np.load(new_npy_path)
            new_index = pd.read_csv(new_index_path, dtype=str)
            # Remove any existing instance
            m = new_index["annotation_id"].astype(str) == aid
            if m.any():
                new_embeddings = new_embeddings[(~m).to_numpy()]
                new_index = new_index[~m].reset_index(drop=True)
            combined_new_embeddings = np.vstack([new_embeddings, vector])
            combined_new_index = pd.concat([new_index, pd.DataFrame([index_record])], ignore_index=True)
        else:
            combined_new_embeddings = vector
            combined_new_index = pd.DataFrame([index_record], columns=INDEX_COLUMNS)

        _save_atomic_npy(combined_new_embeddings.astype(np.float32), new_npy_path)
        _save_atomic_csv(combined_new_index, new_index_path)
        write_location_cache(combined_new_index, new_part_dir)


def sync_annotation_deletion(
    annotation_ids: Sequence[str] | set[str] | str,
    crops_dir: Path = CROPS_DIR,
    embeddings_dir: Path = QUERY_EMBEDDINGS_DIR,
) -> None:
    """Delete crop image files and remove rows/vectors from index and embeddings."""
    if isinstance(annotation_ids, (str, Path)):
        to_delete = {str(annotation_ids)}
    else:
        to_delete = {str(a) for a in annotation_ids}

    if not to_delete:
        return

    crops_index_path = crops_dir / "crops.csv"
    if crops_index_path.is_file():
        crops_df = pd.read_csv(crops_index_path, dtype=str)
        del_rows = crops_df[crops_df["annotation_id"].astype(str).isin(to_delete)]
        for _, r in del_rows.iterrows():
            aid = str(r["annotation_id"])
            bp = str(r.get("body_part") or "")
            f1 = crops_dir / bp / f"{aid}.jpg"
            f1.unlink(missing_ok=True)
            # Also try crop_path if present
            crop_path = r.get("crop_path")
            if crop_path:
                (REPO_ROOT / str(crop_path)).unlink(missing_ok=True)

        remaining_crops = crops_df[~crops_df["annotation_id"].astype(str).isin(to_delete)].reset_index(drop=True)
        _save_atomic_csv(remaining_crops, crops_index_path)

    # Remove from embeddings per body part
    for bp in BODY_PARTS:
        part_dir = embeddings_dir / bp
        npy_path = part_dir / "embeddings.npy"
        index_path = part_dir / "index.csv"

        if not (npy_path.is_file() and index_path.is_file()):
            continue

        index = pd.read_csv(index_path, dtype=str)
        embeddings = np.load(npy_path)
        mask = ~index["annotation_id"].astype(str).isin(to_delete)

        if not mask.all():
            filtered_embeddings = embeddings[mask.to_numpy()]
            filtered_index = index[mask].reset_index(drop=True)
            _save_atomic_npy(filtered_embeddings.astype(np.float32), npy_path)
            _save_atomic_csv(filtered_index, index_path)
            write_location_cache(filtered_index, part_dir)


def sync_encounter_location(
    encounter_id: str,
    latitude: float,
    longitude: float,
    location_source: str = "map",
    crops_dir: Path = CROPS_DIR,
    embeddings_dir: Path = QUERY_EMBEDDINGS_DIR,
) -> None:
    """Update coordinates in crops.csv and embedding indexes for encounter."""
    enc = str(encounter_id)
    is_valid = is_valid_location(latitude, longitude)
    lat_val = str(latitude) if is_valid else ""
    lon_val = str(longitude) if is_valid else ""
    loc_src = str(location_source) if is_valid else ""

    crops_index_path = crops_dir / "crops.csv"
    if crops_index_path.is_file():
        crops_df = pd.read_csv(crops_index_path, dtype=str)
        m = crops_df["observation_id"].astype(str) == enc
        if m.any():
            crops_df.loc[m, "latitude"] = lat_val
            crops_df.loc[m, "longitude"] = lon_val
            if "location_source" in crops_df.columns:
                crops_df.loc[m, "location_source"] = loc_src
            _save_atomic_csv(crops_df, crops_index_path)

    for bp in BODY_PARTS:
        part_dir = embeddings_dir / bp
        index_path = part_dir / "index.csv"
        if not index_path.is_file():
            continue
        index = pd.read_csv(index_path, dtype=str)
        if "location_source" not in index.columns:
            index["location_source"] = ""
        m = index["observation_id"].astype(str) == enc
        if m.any():
            index.loc[m, "latitude"] = lat_val
            index.loc[m, "longitude"] = lon_val
            index.loc[m, "location_source"] = loc_src
            _save_atomic_csv(index, index_path)
            write_location_cache(index, part_dir)

    results_path = DATA_DIR / "results.csv"
    if results_path.is_file():
        results_df = pd.read_csv(results_path, dtype=str)
        m = results_df["encounter_id"].astype(str) == enc
        if m.any():
            results_df.loc[m, "latitude"] = lat_val
            results_df.loc[m, "longitude"] = lon_val
            results_df.loc[m, "location_source"] = loc_src
            _save_atomic_csv(results_df, results_path)

    try:
        import streamlit as st

        st.cache_data.clear()
    except Exception:
        pass


def prune_stale_crops_and_embeddings(
    manifest: pd.DataFrame | None = None,
    crops_dir: Path = CROPS_DIR,
    embeddings_dir: Path = QUERY_EMBEDDINGS_DIR,
) -> dict[str, int]:
    """Remove any orphan crop files, crop index rows, or embedding entries not in manifest."""
    if manifest is None:
        if not LABELS_PATH.is_file():
            return {"crop_files_removed": 0, "crop_rows_removed": 0, "embedding_rows_removed": 0}
        manifest = load_manifest(LABELS_PATH)

    valid_aids = set(manifest["annotation_id"].astype(str))
    valid_pairs = {
        (str(r["annotation_id"]), str(r["body_part"]))
        for _, r in manifest.iterrows()
    }

    stats = {
        "crop_files_removed": 0,
        "crop_rows_removed": 0,
        "embedding_rows_removed": 0,
    }

    # 1. Clean orphan crop files in crops_dir
    if crops_dir.is_dir():
        for bp_dir in crops_dir.iterdir():
            if not bp_dir.is_dir() or bp_dir.name.startswith("."):
                continue
            bp = bp_dir.name
            for crop_file in bp_dir.glob("*.jpg"):
                aid = crop_file.stem
                if aid not in valid_aids or (aid, bp) not in valid_pairs:
                    crop_file.unlink(missing_ok=True)
                    stats["crop_files_removed"] += 1

    # 2. Clean crops.csv
    crops_index_path = crops_dir / "crops.csv"
    if crops_index_path.is_file():
        crops_df = pd.read_csv(crops_index_path, dtype=str)
        mask = crops_df["annotation_id"].astype(str).isin(valid_aids)
        if not mask.all():
            stats["crop_rows_removed"] = int((~mask).sum())
            _save_atomic_csv(crops_df[mask].reset_index(drop=True), crops_index_path)

    # 3. Clean embeddings indexes and vectors
    for bp in BODY_PARTS:
        part_dir = embeddings_dir / bp
        npy_path = part_dir / "embeddings.npy"
        index_path = part_dir / "index.csv"

        if not (npy_path.is_file() and index_path.is_file()):
            continue

        index = pd.read_csv(index_path, dtype=str)
        embeddings = np.load(npy_path)

        mask = index["annotation_id"].astype(str).isin(valid_aids) & (index["body_part"].astype(str) == bp)
        if not mask.all():
            n_removed = int((~mask).sum())
            stats["embedding_rows_removed"] += n_removed
            filtered_embeddings = embeddings[mask.to_numpy()]
            filtered_index = index[mask].reset_index(drop=True)
            _save_atomic_npy(filtered_embeddings.astype(np.float32), npy_path)
            _save_atomic_csv(filtered_index, index_path)
            write_location_cache(filtered_index, part_dir)

    return stats
