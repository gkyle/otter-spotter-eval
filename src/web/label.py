#!/usr/bin/env python3
"""Local labeling page for Giant River Otter re-identification.

Draw bounding boxes on pool images and assign each box an individual id and a
crop target (whole_body, throat, or throat_portrait). Annotations are written to
data/labels.csv (one row per box). Encounter id is filled automatically from
the folder name.

Run with:
    uv run streamlit run src/web/app.py
"""

from __future__ import annotations

# --- Compatibility shim -----------------------------------------------------
# streamlit-drawable-canvas 0.9.x calls the legacy
# `streamlit.elements.image.image_to_url(image, width, clamp, channels,
# output_format, image_id)` and expects a *server-relative* media URL back
# (the component's frontend prepends the Streamlit base URL, so a data: URI
# would be corrupted into "http://host/…data:image/png…" and never load).
# Newer Streamlit removed that symbol and changed the signature, so we install
# a self-contained replacement that registers the PNG with Streamlit's media
# file manager and returns its "/media/…" URL.
import base64  # noqa: E402
import csv  # noqa: E402
import html  # noqa: E402
import io  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402

import streamlit.elements.image as _st_image  # noqa: E402
from PIL import Image as _PILImage  # noqa: E402
from streamlit import runtime as _st_runtime  # noqa: E402


def _image_to_media_url(
    image,
    width=None,
    clamp=False,
    channels="RGB",
    output_format="PNG",
    image_id="",
):
    """Legacy-signature drop-in that serves the background via MediaFileManager."""
    if not isinstance(image, _PILImage.Image):
        image = _PILImage.fromarray(image)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    if _st_runtime.exists():
        return _st_runtime.get_instance().media_file_mgr.add(
            buffer.getvalue(), "image/png", image_id
        )
    return ""


_st_image.image_to_url = _image_to_media_url  # type: ignore[attr-defined]
# ---------------------------------------------------------------------------

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.manifest import (  # noqa: E402
    BODY_PARTS,
    DATA_DIR,
    IMAGES_DIR,
    MANIFEST_PATH,
    REPO_ROOT,
    is_valid_location,
    load_manifest,
    list_images,
    next_individual_id,
    observation_id_for,
    relative_image_path,
    save_manifest,
)
from identify.sync import (  # noqa: E402
    sync_annotation_addition,
    sync_annotation_deletion,
    sync_annotation_update,
    sync_encounter_location,
)
from location_picker import location_picker_dialog  # noqa: E402
from common.upload import (  # noqa: E402
    UPLOAD_IMAGES_DIR,
    build_upload_index,
    image_metadata,
    save_upload_index,
)
import importlib
try:
    import common.sources
    importlib.reload(common.sources)
except Exception:
    pass
from common.sources import (  # noqa: E402
    RESULTS_PATH,
    RESULTS_REVIEW_CSV,
    load_sources_results,
    load_review_results,
    save_review_results,
    results_file_token,
    review_file_token,
    remove_review_record,
    update_source_records,
)

SOURCE_FILTER_OPTIONS = ("All", "inat", "flickr", "wikimedia", "upload", "review")

from common.web import require_login  # noqa: E402

CANVAS_WIDTH = 580  # fixed canvas render width in px matching 3:2 column split in 1024px container
MIN_BOX_PX = 4       # ignore accidental tiny boxes (in original pixels)
BODY_PART_COLORS = {
    "whole_body": "#00E5FF",
    "throat": "#FF4081",
    "throat_portrait": "#76FF03",
}
DETECTOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "data/models/otter-detector/yolo26n-640-final/weights/best.pt"
)
DETECTOR_CONFIDENCE = 0.25
LABELS_NEGATIVE_PATH = DATA_DIR / "labels_negative.csv"
DETECTOR_NEGATIVES_PATH = LABELS_NEGATIVE_PATH
UPLOAD_SESSIONS_DIR = REPO_ROOT / "data" / "sources" / "upload" / "sessions"
PENDING_CSV = REPO_ROOT / "data" / "sources" / "upload" / "upload_pending.csv"
DETECTOR_NEGATIVE_COLUMNS = [
    "negative_id",
    "image_path",
    "observation_id",
    "body_part",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "img_width",
    "img_height",
    "detector_model",
    "confidence",
    "created_at",
]
@st.cache_data(show_spinner=False)
def cached_inat_image_list() -> list[str]:
    return [str(p) for p in list_images(IMAGES_DIR)]


def upload_tree_token() -> tuple[tuple[str, int, int], ...]:
    """Return metadata that invalidates the upload index when files change."""
    if not UPLOAD_IMAGES_DIR.is_dir():
        return ()
    return tuple(
        (
            str(path.relative_to(UPLOAD_IMAGES_DIR)),
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in sorted(UPLOAD_IMAGES_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
    )



@st.cache_data(show_spinner=False)
def cached_upload_index(
    tree_token: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    """Index uploaded images, invalidated by any image change."""
    del tree_token
    return build_upload_index()


def upload_index() -> pd.DataFrame:
    return cached_upload_index(upload_tree_token())


def upload_record_for(image_path: Path) -> pd.Series | None:
    relative = str(relative_image_path(image_path))
    if PENDING_CSV.is_file():
        try:
            df_pending = pd.read_csv(PENDING_CSV, dtype=str)
            matches = df_pending[df_pending["image_path"].astype(str) == relative]
            if not matches.empty:
                r = matches.iloc[0]
                return pd.Series({
                    "encounter_id": r.get("upload_id") or image_path.parent.name,
                    "individual_id": r.get("individual_id") or "",
                    "latitude": r.get("latitude") or "",
                    "longitude": r.get("longitude") or "",
                    "location_source": r.get("location_source") or "",
                })
        except Exception:
            pass
    matches = upload_index()
    matches = matches[matches["image_path"].astype(str) == relative]
    return matches.iloc[0] if len(matches) else None


def review_tree_token() -> tuple[tuple[str, int, int], ...]:
    """Return metadata that invalidates the review index when session files change."""
    if not UPLOAD_SESSIONS_DIR.is_dir():
        return ()
    return tuple(
        (
            str(path.relative_to(UPLOAD_SESSIONS_DIR)),
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in sorted(UPLOAD_SESSIONS_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
    )


@st.cache_data(show_spinner=False)
def cached_review_image_list(
    tree_token: tuple[tuple[str, int, int], ...],
) -> list[str]:
    """List images sitting in upload review sessions, invalidated by file changes."""
    del tree_token
    if not UPLOAD_SESSIONS_DIR.is_dir():
        return []
    images = []
    for path in sorted(UPLOAD_SESSIONS_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}:
            images.append(str(path))
    return sorted(images)


def resolve_upload_image_by_encounter_or_path(
    encounter_id: str | None = None,
    image_rel_or_path: str | None = None,
) -> Path | None:
    """Find a committed upload image by relative path or encounter ID."""
    if image_rel_or_path:
        full_p = REPO_ROOT / image_rel_or_path
        if full_p.is_file():
            return full_p
        path_obj = Path(image_rel_or_path)
        if path_obj.is_file():
            return path_obj

    if encounter_id:
        enc_dir = UPLOAD_IMAGES_DIR / encounter_id
        if enc_dir.is_dir():
            files = sorted(
                p for p in enc_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
            )
            if files:
                return files[0]

    return None


def commit_review_image(
    image_path: Path,
    manifest: pd.DataFrame,
) -> None:
    """Transfer an image from upload session directory to uploads/images/<encounter_id>/ and commit to manifest."""
    rel_old = relative_image_path(image_path)
    rel_old_str = str(rel_old)

    session_id = image_path.parent.name
    target_dir = UPLOAD_IMAGES_DIR / session_id
    target_dir.mkdir(parents=True, exist_ok=True)

    target_filename = image_path.name
    dest_path = target_dir / target_filename
    if dest_path.is_file() and dest_path.resolve() != image_path.resolve():
        short_id = session_id.split("-")[-1] if "-" in session_id else "rev"
        target_filename = f"{image_path.stem}_{short_id}{image_path.suffix}"
        dest_path = target_dir / target_filename

    shutil.move(str(image_path), str(dest_path))
    rel_new = relative_image_path(dest_path)
    rel_new_str = str(rel_new)

    updated_manifest = manifest.copy()
    existing_mask = updated_manifest["image_path"].astype(str) == rel_old_str

    new_encounter = encounter_id_for(dest_path)

    if existing_mask.any():
        updated_manifest.loc[existing_mask, "image_path"] = rel_new_str
        if new_encounter:
            updated_manifest.loc[existing_mask, "observation_id"] = new_encounter


    pending_row = {}
    new_rows = []
    if PENDING_CSV.is_file():
        try:
            df_pending = pd.read_csv(PENDING_CSV, dtype=str)
            pending_matches = df_pending[df_pending["image_path"].astype(str) == rel_old_str]
            if not pending_matches.empty:
                pending_row = pending_matches.iloc[0].to_dict()

            if not existing_mask.any() and not pending_matches.empty:
                for _, prow in pending_matches.iterrows():
                    bw = int(float(prow.get("bbox_w") or 0))
                    bh = int(float(prow.get("bbox_h") or 0))
                    if bw > 0 and bh > 0:
                        new_rows.append({
                            "annotation_id": uuid.uuid4().hex[:12],
                            "image_path": rel_new_str,
                            "observation_id": new_encounter,
                            "latitude": prow.get("latitude") or "",
                            "longitude": prow.get("longitude") or "",
                            "location_source": prow.get("location_source") or "user",
                            "individual_id": prow.get("individual_id") or "",

                            "body_part": prow.get("body_part") or "throat_portrait",
                            "bbox_x": int(float(prow.get("bbox_x") or 0)),
                            "bbox_y": int(float(prow.get("bbox_y") or 0)),
                            "bbox_w": bw,
                            "bbox_h": bh,
                            "img_width": int(float(prow.get("img_width") or 0)),
                            "img_height": int(float(prow.get("img_height") or 0)),
                            "split": "",
                            "notes": f"Committed from session {image_path.parent.name}",
                            "created_at": now_iso(),
                        })

            df_remaining = df_pending[df_pending["image_path"].astype(str) != rel_old_str]
            if df_remaining.empty:
                PENDING_CSV.unlink(missing_ok=True)
            else:
                temporary = PENDING_CSV.with_name(f".{PENDING_CSV.name}.tmp")
                df_remaining.to_csv(temporary, index=False)
                temporary.replace(PENDING_CSV)
        except Exception as exc:
            print(f"Error processing upload_pending.csv: {exc}", file=sys.stderr)

    if new_rows:
        updated_manifest = pd.concat([updated_manifest, pd.DataFrame(new_rows)], ignore_index=True)

    st.session_state.manifest = updated_manifest
    save_manifest(st.session_state.manifest)
    if new_rows:
        sync_annotation_addition(new_rows, prune_stale=True)

    session_dir = image_path.parent
    if session_dir.is_dir():
        remaining_in_session = [
            p for p in session_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
        ]
        if not remaining_in_session:
            try:
                session_dir.rmdir()
            except OSError:
                pass

    try:
        df_upload = build_upload_index()
        mask_u = df_upload["image_path"].astype(str) == rel_new_str
        if mask_u.any():
            idx_u = df_upload[mask_u].index[0]
            for col in ("owner", "individual_id", "source_folder", "captured_at", "latitude", "longitude", "location_source"):
                val = pending_row.get(col)
                if val and not pd.isna(val):
                    df_upload.at[idx_u, col] = str(val)
        else:
            captured_at_exif, lat_exif, lon_exif = image_metadata(dest_path)
            new_rec = {
                "image_path": rel_new_str,
                "owner": str(pending_row.get("owner") or ""),
                "individual_id": str(pending_row.get("individual_id") or ""),
                "source_folder": str(pending_row.get("source_folder") or "."),
                "captured_at": str(pending_row.get("captured_at") or (captured_at_exif.isoformat() if captured_at_exif else "")),
                "encounter_id": new_encounter,
                "latitude": str(pending_row.get("latitude") or (str(lat_exif) if lat_exif is not None else "")),
                "longitude": str(pending_row.get("longitude") or (str(lon_exif) if lon_exif is not None else "")),
                "location_source": str(pending_row.get("location_source") or ("exif" if lat_exif is not None else "")),
            }
            df_upload = pd.concat([df_upload, pd.DataFrame([new_rec])], ignore_index=True)
        save_upload_index(df_upload)
    except Exception as exc:
        print(f"Error rebuilding upload index: {exc}", file=sys.stderr)

    try:
        df_rev = load_review_results()
        rev_match = df_rev[df_rev["image_path"].astype(str) == rel_old_str]
        if not rev_match.empty:
            rev_rec = rev_match.iloc[0].to_dict()
            rev_rec["image_path"] = rel_new_str
            rev_rec["source"] = "upload"
            if new_encounter:
                rev_rec["encounter_id"] = new_encounter
            update_source_records("upload", [rev_rec])
            remove_review_record(rel_old_str)
        else:
            captured_at_exif, lat_exif, lon_exif = image_metadata(dest_path)
            lat_val = str(pending_row.get("latitude") or (str(lat_exif) if lat_exif is not None else ""))
            lon_val = str(pending_row.get("longitude") or (str(lon_exif) if lon_exif is not None else ""))
            new_res_rec = {
                "source": "upload",
                "encounter_id": new_encounter or f"upload_{session_id}",
                "photo_id": dest_path.stem,
                "image_path": rel_new_str,
                "image_url": "",
                "page_url": "",
                "observed_at": str(pending_row.get("captured_at") or (captured_at_exif.isoformat() if captured_at_exif else "")),
                "observer": str(pending_row.get("owner") or ""),
                "latitude": lat_val,
                "longitude": lon_val,
                "location_source": str(pending_row.get("location_source") or ("exif" if lat_val and lon_val else "")),
                "locality": "",
                "license": "",
                "quality_grade": "",
            }
            update_source_records("upload", [new_res_rec])
    except Exception as exc:
        print(f"Error transferring review record to results.csv: {exc}", file=sys.stderr)

    # Maintain "review" source mode and stay on the committed image via query params & session state
    st.session_state["_pending_label_image_source"] = "review"
    st.session_state["review_override_image"] = str(dest_path)
    st.session_state.current_encounter_id = new_encounter

    st.query_params["source"] = "review"
    st.query_params["encounter_id"] = new_encounter
    st.query_params["image"] = rel_new_str

    st.session_state.canvas_nonce += 1
    st.toast(f"Committed {dest_path.name} to uploads as encounter {new_encounter}!")
    st.rerun()


@st.cache_data(show_spinner=False)
def load_label_records(
    source_filter: str,
    res_token: tuple[int, int],
    rev_token: tuple[int, int],
) -> pd.DataFrame:
    """Load records for the requested source filter from results.csv and/or results_review.csv."""
    del res_token, rev_token
    src = normalize_source_filter(source_filter)
    if src == "review":
        return load_review_results()

    df_res = load_sources_results()
    if src == "All":
        return df_res
    return df_res[df_res["source"] == src].copy()


def normalize_source_filter(val: str | None) -> str:
    """Normalize a source name or query parameter to one of SOURCE_FILTER_OPTIONS."""
    if not val:
        return "All"
    v = str(val).strip().lower()
    if v in ("all", "*"):
        return "All"
    if v in ("inat", "inaturalist"):
        return "inat"
    if v in ("flickr", "fl"):
        return "flickr"
    if v in ("wikimedia", "wm"):
        return "wikimedia"
    if v in ("upload", "uploads"):
        return "upload"
    if v in ("review", "sessions"):
        return "review"
    return "All"


def images_for_source(source_key: str) -> list[str]:
    """Return image file paths for a single data source."""
    s = normalize_source_filter(source_key)
    res_tok = results_file_token()
    rev_tok = review_file_token()
    df = load_label_records(s, res_tok, rev_tok)
    if df.empty or "image_path" not in df.columns:
        return []
    return [str(REPO_ROOT / p) for p in df["image_path"].dropna()]


def all_source_images() -> list[str]:
    """Return unified list of images across all data sources in consistent order."""
    imgs = []
    for s in ("inat", "flickr", "wikimedia", "upload", "review"):
        imgs.extend(images_for_source(s))
    return imgs


@st.cache_data(show_spinner=False)
def cached_results_metadata(
    res_token: tuple[int, int],
    rev_token: tuple[int, int],
) -> dict[str, dict[str, object]]:
    """Index metadata from results.csv and results_review.csv by relative image path."""
    del res_token, rev_token
    meta_by_path: dict[str, dict[str, object]] = {}
    res = load_sources_results()
    rev = load_review_results()
    combined = pd.concat([res, rev], ignore_index=True) if not rev.empty else res
    for _, row in combined.iterrows():
        p = str(row.get("image_path") or "").strip()
        if p:
            lat = pd.to_numeric(pd.Series([row.get("latitude")]), errors="coerce").iloc[0]
            lon = pd.to_numeric(pd.Series([row.get("longitude")]), errors="coerce").iloc[0]
            valid_loc = is_valid_location(lat, lon)
            meta_by_path[p] = {
                "source": str(row.get("source") or ""),
                "encounter_id": str(row.get("encounter_id") or ""),
                "latitude": float(lat) if valid_loc else "",
                "longitude": float(lon) if valid_loc else "",
                "location_source": str(row.get("location_source") or row.get("source") or "") if valid_loc else "",
                "owner": str(row.get("observer") or "").strip(),
                "captured_at": str(row.get("observed_at") or "").strip(),
                "locality": str(row.get("locality") or "").strip(),
            }
    return meta_by_path


def encounter_id_for(image_path: Path) -> str:
    """Return the standardized encounter ID (<source>_<id>) for an image."""
    rel_p = str(relative_image_path(image_path))
    meta_map = cached_results_metadata(results_file_token(), review_file_token())
    if rel_p in meta_map and meta_map[rel_p]["encounter_id"]:
        return str(meta_map[rel_p]["encounter_id"])
    return ""


def source_for_image(image_path: Path) -> str:
    """Return the source recorded for an image in results metadata."""
    rel_p = str(relative_image_path(image_path))
    meta_map = cached_results_metadata(results_file_token(), review_file_token())
    return str(meta_map.get(rel_p, {}).get("source") or "")


def source_for_encounter(encounter_id: str) -> str | None:
    """Return the source recorded for an encounter in results.csv."""
    wanted = str(encounter_id).strip()
    if not wanted:
        return None
    results = load_sources_results()
    values = results["encounter_id"].astype(str)
    matches = results[
        (values == wanted)
        | values.str.endswith(f"_{wanted}")
        | values.str.endswith(f"-{wanted}")
    ]
    if matches.empty:
        return None
    return str(matches.iloc[0]["source"])


def location_fields_for(image_path: Path) -> dict[str, object]:
    """Return manifest location fields for one source image."""
    rel_path = str(relative_image_path(image_path))
    meta_map = cached_results_metadata(results_file_token(), review_file_token())
    if rel_path in meta_map:
        info = meta_map[rel_path]
        if info["latitude"] != "" and info["longitude"] != "":
            return {
                "latitude": info["latitude"],
                "longitude": info["longitude"],
                "location_source": info["location_source"],
            }
    return {"latitude": "", "longitude": "", "location_source": ""}


def metadata_fields_for(image_path: Path) -> dict[str, object]:
    """Return location, owner, and timestamp fields for one source image across all data sources."""
    location_fields = location_fields_for(image_path)
    rel_path = str(relative_image_path(image_path))
    meta_map = cached_results_metadata(results_file_token(), review_file_token())
    if rel_path in meta_map:
        info = meta_map[rel_path]
        return {
            **location_fields,
            "owner": str(info.get("owner") or ""),
            "captured_at": str(info.get("captured_at") or ""),
        }
    return {
        **location_fields,
        "owner": "",
        "captured_at": "",
    }



def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rects_to_rows(
    canvas_json: dict,
    orig_w: int,
    orig_h: int,
    scale: float,
    image_path: Path,
    individual_id: str,
    body_part: str,
    notes: str,
    skip: int = 0,
) -> list[dict]:
    """Convert canvas rectangles (display px) into manifest rows (original px).

    ``skip`` drops the first N rectangles, which are the read-only saved boxes
    pre-loaded onto the canvas (fabric keeps them ahead of newly drawn ones).
    """
    rows: list[dict] = []
    rects = [o for o in (canvas_json or {}).get("objects", []) if o.get("type") == "rect"]
    for obj in rects[skip:]:
        x = obj["left"] * scale
        y = obj["top"] * scale
        w = obj["width"] * obj.get("scaleX", 1) * scale
        h = obj["height"] * obj.get("scaleY", 1) * scale
        # Clamp to image bounds.
        x0 = max(0, min(orig_w, round(x)))
        y0 = max(0, min(orig_h, round(y)))
        w0 = max(0, min(orig_w - x0, round(w)))
        h0 = max(0, min(orig_h - y0, round(h)))
        if w0 < MIN_BOX_PX or h0 < MIN_BOX_PX:
            continue
        rows.append(
            {
                "annotation_id": uuid.uuid4().hex[:12],
                "image_path": relative_image_path(image_path),
                "observation_id": encounter_id_for(image_path),
                **location_fields_for(image_path),
                "individual_id": individual_id,
                "body_part": body_part,
                "bbox_x": x0,
                "bbox_y": y0,
                "bbox_w": w0,
                "bbox_h": h0,
                "img_width": orig_w,
                "img_height": orig_h,
                "split": "",
                "notes": notes,
                "created_at": now_iso(),
            }
        )
    return rows


def saved_boxes_drawing(current: pd.DataFrame, scale: float) -> dict:
    """Build read-only Fabric boxes and individual-id labels."""
    objects = []
    for _, row in current.iterrows():
        left = float(row.bbox_x) / scale
        top = float(row.bbox_y) / scale
        height = float(row.bbox_h) / scale
        color = BODY_PART_COLORS.get(row.body_part, "#FFEB3B")
        objects.append(
            {
                "type": "rect",
                "left": left,
                "top": top,
                "width": float(row.bbox_w) / scale,
                "height": height,
                "fill": "rgba(0,0,0,0)",
                "stroke": color,
                "strokeWidth": 2,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "selectable": False,
                "evented": False,
            }
        )
        objects.append(
            {
                "type": "text",
                "text": str(row.individual_id),
                # Place the id just outside the bottom-left edge so it does not
                # cover any of the annotated image content.
                "left": left,
                "top": top + height + 3,
                "fill": "#FFFFFF",
                "backgroundColor": "rgba(0,0,0,0.72)",
                "fontSize": 15,
                "fontFamily": "sans-serif",
                "fontWeight": "bold",
                "selectable": False,
                "evented": False,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
            }
        )
    return {"version": "4.4.0", "objects": objects}


def generated_boxes_drawing(generated: list[dict], scale: float) -> dict:
    """Build read-only dashed Fabric boxes for detector suggestions."""
    objects = []
    for prediction in generated:
        left = prediction["bbox_x"] / scale
        top = prediction["bbox_y"] / scale
        color = BODY_PART_COLORS.get(prediction["body_part"], "#FFEB3B")
        objects.extend(
            [
                {
                    "type": "rect",
                    "left": left,
                    "top": top,
                    "width": prediction["bbox_w"] / scale,
                    "height": prediction["bbox_h"] / scale,
                    "fill": "rgba(0,0,0,0)",
                    "stroke": color,
                    "strokeWidth": 2,
                    "strokeDashArray": [7, 5],
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                    "selectable": False,
                    "evented": False,
                },
                {
                    "type": "text",
                    "text": (
                        f"generated {prediction['body_part']} "
                        f"{prediction['confidence']:.0%}"
                    ),
                    "left": left,
                    "top": max(0, top - 19),
                    "fill": "#FFFFFF",
                    "backgroundColor": "rgba(0,0,0,0.72)",
                    "fontSize": 14,
                    "fontFamily": "sans-serif",
                    "selectable": False,
                    "evented": False,
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                },
            ]
        )
    return {"version": "4.4.0", "objects": objects}


def combine_drawings(*drawings: dict | None) -> dict | None:
    """Combine Fabric drawings while preserving their object order."""
    objects = [
        obj
        for drawing in drawings
        if drawing
        for obj in drawing.get("objects", [])
    ]
    return {"version": "4.4.0", "objects": objects} if objects else None


@st.cache_resource(show_spinner=False)
def load_detector(model_path: str, model_mtime_ns: int):
    """Load the detector once per model version."""
    del model_mtime_ns  # Included only to invalidate the cache on replacement.
    from ultralytics import YOLO

    return YOLO(model_path)


@st.cache_data(show_spinner=False)
def detector_predictions(
    image_path: str, model_path: str, model_mtime_ns: int
) -> list[dict]:
    """Return detector boxes in original-image pixel coordinates."""
    model = load_detector(model_path, model_mtime_ns)
    result = model.predict(
        source=image_path,
        imgsz=640,
        conf=DETECTOR_CONFIDENCE,
        verbose=False,
    )[0]
    predictions = []
    if result.boxes is None:
        return predictions
    for index, (xyxy, confidence, class_id) in enumerate(
        zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.int().cpu().tolist(),
        )
    ):
        x1, y1, x2, y2 = xyxy
        predictions.append(
            {
                "generated_id": f"{class_id}-{index}-{round(x1)}-{round(y1)}",
                "body_part": str(result.names[class_id]),
                "confidence": float(confidence),
                "bbox_x": max(0, round(x1)),
                "bbox_y": max(0, round(y1)),
                "bbox_w": max(0, round(x2) - round(x1)),
                "bbox_h": max(0, round(y2) - round(y1)),
            }
        )
    return predictions


def intersection_over_union(prediction: dict, row: pd.Series) -> float:
    """Calculate IoU between one generated and one stored xywh box."""
    ax1, ay1 = prediction["bbox_x"], prediction["bbox_y"]
    ax2 = ax1 + prediction["bbox_w"]
    ay2 = ay1 + prediction["bbox_h"]
    bx1, by1 = float(row.bbox_x), float(row.bbox_y)
    bx2, by2 = bx1 + float(row.bbox_w), by1 + float(row.bbox_h)
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    union = (
        prediction["bbox_w"] * prediction["bbox_h"]
        + float(row.bbox_w) * float(row.bbox_h)
        - intersection
    )
    return intersection / union if union else 0.0


def unpromoted_predictions(
    predictions: list[dict], current: pd.DataFrame
) -> list[dict]:
    """Hide predictions that already have a near-identical stored annotation."""
    remaining = []
    for prediction in predictions:
        same_part = current[current["body_part"] == prediction["body_part"]]
        if not any(
            intersection_over_union(prediction, row) >= 0.9
            for _, row in same_part.iterrows()
        ):
            remaining.append(prediction)
    return remaining


def prediction_to_row(
    prediction: dict,
    image_path: Path,
    image_size: tuple[int, int],
    individual_id: str,
) -> dict:
    """Convert a detector suggestion into a normal manifest annotation row."""
    orig_w, orig_h = image_size
    return {
        "annotation_id": uuid.uuid4().hex[:12],
        "image_path": relative_image_path(image_path),
        "observation_id": encounter_id_for(image_path),
        **location_fields_for(image_path),
        "individual_id": individual_id,
        "body_part": prediction["body_part"],
        "bbox_x": prediction["bbox_x"],
        "bbox_y": prediction["bbox_y"],
        "bbox_w": prediction["bbox_w"],
        "bbox_h": prediction["bbox_h"],
        "img_width": orig_w,
        "img_height": orig_h,
        "split": "",
        "notes": (
            f"Generated by {DETECTOR_PATH.name}; "
            f"confidence={prediction['confidence']:.6f}"
        ),
        "created_at": now_iso(),
    }


def load_detector_negatives() -> pd.DataFrame:
    """Load rejected detector boxes used as hard-negative training crops."""
    path = DETECTOR_NEGATIVES_PATH
    if not path.exists():
        legacy = DATA_DIR / "detector_negatives.csv"
        if legacy.exists():
            path = legacy
        else:
            return pd.DataFrame(columns=DETECTOR_NEGATIVE_COLUMNS)
    negatives = pd.read_csv(
        path, dtype={"observation_id": str}
    )
    for column in DETECTOR_NEGATIVE_COLUMNS:
        if column not in negatives:
            negatives[column] = pd.NA
    return negatives[DETECTOR_NEGATIVE_COLUMNS]


def save_detector_negatives(negatives: pd.DataFrame) -> None:
    """Persist hard negatives atomically."""
    DETECTOR_NEGATIVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = DETECTOR_NEGATIVES_PATH.with_name(
        f".{DETECTOR_NEGATIVES_PATH.name}.tmp"
    )
    negatives[DETECTOR_NEGATIVE_COLUMNS].to_csv(temporary, index=False)
    temporary.replace(DETECTOR_NEGATIVES_PATH)


def prediction_to_negative(
    prediction: dict,
    image_path: Path,
    image_size: tuple[int, int],
) -> dict:
    """Convert a rejected suggestion into a persistent hard-negative record."""
    orig_w, orig_h = image_size
    return {
        "negative_id": uuid.uuid4().hex[:12],
        "image_path": relative_image_path(image_path),
        "observation_id": encounter_id_for(image_path),
        "body_part": prediction["body_part"],
        "bbox_x": prediction["bbox_x"],
        "bbox_y": prediction["bbox_y"],
        "bbox_w": prediction["bbox_w"],
        "bbox_h": prediction["bbox_h"],
        "img_width": orig_w,
        "img_height": orig_h,
        "detector_model": DETECTOR_PATH.name,
        "confidence": prediction["confidence"],
        "created_at": now_iso(),
    }


def update_annotation(
    manifest: pd.DataFrame,
    annotation_id: str,
    individual_id: str,
    body_part: str,
) -> pd.DataFrame:
    """Return the manifest with one annotation's editable labels updated."""
    updated = manifest.copy()
    matches = updated["annotation_id"].astype(str) == str(annotation_id)
    updated.loc[matches, ["individual_id", "body_part"]] = [individual_id, body_part]
    return updated


def encounter_location(
    manifest: pd.DataFrame, encounter_id: str
) -> tuple[float | None, float | None]:
    """Return the currently-saved lat/lon for an encounter's manifest rows, if any."""
    rows = manifest[manifest["observation_id"].astype(str) == str(encounter_id)]
    if rows.empty:
        return None, None
    latitude = pd.to_numeric(pd.Series([rows.iloc[0].get("latitude")]), errors="coerce").iloc[0]
    longitude = pd.to_numeric(pd.Series([rows.iloc[0].get("longitude")]), errors="coerce").iloc[0]
    if not is_valid_location(latitude, longitude):
        return None, None
    return float(latitude), float(longitude)


def update_encounter_location(
    manifest: pd.DataFrame, encounter_id: str, latitude: float, longitude: float
) -> pd.DataFrame:
    """Return the manifest with every row for an encounter set to one location."""
    updated = manifest.copy()
    matches = updated["observation_id"].astype(str) == str(encounter_id)
    if not is_valid_location(latitude, longitude):
        updated.loc[matches, ["latitude", "longitude", "location_source"]] = [
            "",
            "",
            "",
        ]
    else:
        updated.loc[matches, ["latitude", "longitude", "location_source"]] = [
            latitude,
            longitude,
            "map",
        ]
    return updated


def duplicate_annotation(
    manifest: pd.DataFrame, annotation_id: str
) -> tuple[pd.DataFrame, str]:
    """Copy one annotation with a fresh id and creation timestamp."""
    matches = manifest[manifest["annotation_id"].astype(str) == str(annotation_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected one annotation for id {annotation_id!r}")

    duplicate = matches.iloc[0].copy()
    duplicate_id = uuid.uuid4().hex[:12]
    duplicate["annotation_id"] = duplicate_id
    duplicate["created_at"] = now_iso()
    updated = pd.concat([manifest, duplicate.to_frame().T], ignore_index=True)
    return updated, duplicate_id


def normalize_encounter_query(encounter_id: str) -> set[str]:
    """Generate potential matching encounter IDs across legacy and new formats."""
    wanted = encounter_id.strip()
    if not wanted:
        return set()
    candidates = {wanted, wanted.lower()}
    if wanted.isdigit():
        candidates.add(f"inat_{wanted}")
    if wanted.startswith("fl-"):
        candidates.add(f"flickr_{wanted[3:]}")
    if wanted.startswith("wm-"):
        candidates.add(f"wikimedia_{wanted[3:]}")
    if wanted.startswith("upload-"):
        candidates.add(f"upload_{wanted[7:]}")
        candidates.add(wanted[7:])
    if wanted.startswith("upload_"):
        candidates.add(f"upload-{wanted[7:]}")
        candidates.add(wanted[7:])
    if re.match(r"^\d{8}-[a-f0-9]{8}$", wanted, re.IGNORECASE):
        candidates.add(f"upload-{wanted}")
        candidates.add(f"upload_{wanted}")
    if wanted.startswith("inat_"):
        candidates.add(wanted[5:])
    if wanted.startswith("flickr_"):
        candidates.add(f"fl-{wanted[7:]}")
    if wanted.startswith("wikimedia_"):
        candidates.add(f"wm-{wanted[10:]}")
    return candidates


def encounter_index(images: list[Path], encounter_id: str) -> int | None:
    """Return the first image index for an encounter folder."""
    candidates = normalize_encounter_query(encounter_id)
    if not candidates:
        return None
    return next(
        (
            index
            for index, path in enumerate(images)
            if encounter_id_for(path) in candidates
            or path.parent.name in candidates
            or path.stem in candidates
        ),
        None,
    )


def apply_annotation_defaults(image_rel: str, default_id: str) -> None:
    """Set initial label and default individual ID state for a newly loaded image."""
    st.session_state.adding_new = True
    st.session_state.new_individual_id = default_id
    st.session_state.last_individual = default_id
    generated_defaults = st.session_state.setdefault(
        "generated_label_defaults", {}
    )
    generated_defaults[image_rel] = default_id
    generated_nonces = st.session_state.setdefault(
        "generated_label_nonces", {}
    )
    generated_nonces[image_rel] = generated_nonces.get(image_rel, 0) + 1
    generated_key_prefix = f"generated-label-{image_rel}-"
    for key in list(st.session_state):
        if key.startswith(generated_key_prefix):
            st.session_state[key] = default_id


def main() -> None:
    require_login()
    if "manifest" not in st.session_state:
        st.session_state.manifest = load_manifest()

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    if "canvas_nonce" not in st.session_state:
        st.session_state.canvas_nonce = 0
    if "last_individual" not in st.session_state:
        st.session_state.last_individual = ""
    if "adding_new" not in st.session_state:
        st.session_state.adding_new = False
    # Widget-backed state must be changed before the widget is instantiated.
    if st.session_state.pop("show_all_for_encounter", False):
        st.session_state.only_unlabeled = False

    manifest: pd.DataFrame = st.session_state.manifest
    requested_image = st.query_params.get("image")
    requested_source = st.query_params.get("source")
    requested_encounter = st.query_params.get("encounter_id")
    if isinstance(requested_image, list):
        requested_image = requested_image[-1] if requested_image else None
    if isinstance(requested_source, list):
        requested_source = requested_source[-1] if requested_source else None
    if isinstance(requested_encounter, list):
        requested_encounter = requested_encounter[-1] if requested_encounter else None

    if "_pending_source_filter" in st.session_state:
        st.session_state.label_source_filter = normalize_source_filter(
            st.session_state.pop("_pending_source_filter")
        )
    elif "_pending_label_image_source" in st.session_state:
        st.session_state.label_source_filter = normalize_source_filter(
            st.session_state.pop("_pending_label_image_source")
        )
    elif "label_source_filter" not in st.session_state:
        if requested_encounter:
            enc_src = source_for_encounter(requested_encounter)
            if requested_source:
                req_s = normalize_source_filter(requested_source)
                if enc_src and req_s != "All" and req_s != enc_src:
                    st.session_state.label_source_filter = "All"
                else:
                    st.session_state.label_source_filter = req_s
            else:
                st.session_state.label_source_filter = "All"
        elif requested_source:
            st.session_state.label_source_filter = normalize_source_filter(requested_source)
        elif requested_image:
            img_str = str(requested_image)
            if "sessions" in img_str:
                st.session_state.label_source_filter = "review"
            else:
                image_meta = cached_results_metadata(
                    results_file_token(), review_file_token()
                ).get(img_str, {})
                p_src = str(image_meta.get("source") or "")
                st.session_state.label_source_filter = p_src if p_src else "All"
        else:
            st.session_state.label_source_filter = "All"
    elif requested_encounter:
        enc_src = source_for_encounter(requested_encounter)
        curr = st.session_state.get("label_source_filter", "All")
        if enc_src and curr != "All" and enc_src != curr:
            st.session_state.label_source_filter = "All"

    def _on_source_change():
        st.session_state.idx = 0
        st.session_state.pop("review_override_image", None)
        curr = st.session_state.get("label_source_filter")
        if curr and curr != "All":
            st.query_params["source"] = curr
        else:
            st.query_params.pop("source", None)
        st.query_params.pop("image", None)
        st.query_params.pop("encounter_id", None)

    with st.sidebar:
        has_filter_state = "label_source_filter" in st.session_state
        cur_filter = st.session_state.get("label_source_filter", "All")
        source = st.segmented_control(
            "Filter by source",
            SOURCE_FILTER_OPTIONS,
            default=cur_filter if not has_filter_state else None,
            selection_mode="single",
            required=True,
            key="label_source_filter",
            on_change=_on_source_change,
        ) or cur_filter

    if source == "All":
        image_names = all_source_images()
    else:
        image_names = images_for_source(source)

    if "_pending_target_image_rel" in st.session_state:
        target_rel = st.session_state.pop("_pending_target_image_rel")
        try:
            dest_idx = next(i for i, p in enumerate(image_names) if target_rel in p)
            st.session_state.idx = dest_idx
        except (Exception, StopIteration):
            st.session_state.idx = 0

    all_images = [Path(path) for path in image_names]

    if requested_image:
        target_index = next(
            (
                index
                for index, path in enumerate(all_images)
                if str(relative_image_path(path)) == str(requested_image)
            ),
            None,
        )
        if target_index is not None:
            st.session_state.idx = target_index
            st.session_state.only_unlabeled = False
            st.session_state.current_encounter_id = encounter_id_for(
                all_images[target_index]
            )
        else:
            global_names = all_source_images()
            try:
                dest_idx = next(
                    i for i, p in enumerate(global_names)
                    if str(requested_image) in p
                )
                st.session_state.label_source_filter = "All"
                image_names = global_names
                all_images = [Path(path) for path in image_names]
                st.session_state.idx = dest_idx
                st.session_state.only_unlabeled = False
                st.session_state.current_encounter_id = encounter_id_for(
                    all_images[dest_idx]
                )
            except StopIteration:
                pass
    elif requested_encounter:
        target_index = encounter_index(all_images, requested_encounter)
        if target_index is not None:
            st.session_state.idx = target_index
            st.session_state.only_unlabeled = False
            st.session_state.current_encounter_id = encounter_id_for(
                all_images[target_index]
            )
        else:
            global_all = [Path(p) for p in all_source_images()]
            target_index = encounter_index(global_all, requested_encounter)
            if target_index is not None:
                st.session_state.label_source_filter = "All"
                all_images = global_all
                st.session_state.idx = target_index
                st.session_state.only_unlabeled = False
                st.session_state.current_encounter_id = encounter_id_for(
                    all_images[target_index]
                )
    labeled_paths = set(manifest["image_path"].astype(str))
    if all_images and "current_encounter_id" not in st.session_state:
        initial_index = max(0, min(st.session_state.idx, len(all_images) - 1))
        st.session_state.current_encounter_id = encounter_id_for(
            all_images[initial_index]
        )

    # ---- Sidebar: stats, filters, individual + body-part selection ----------
    with st.sidebar:
        st.header("Progress")
        n_ann = len(manifest)
        n_imgs = manifest["image_path"].nunique() if n_ann else 0
        n_ind = manifest["individual_id"].nunique() if n_ann else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Boxes", n_ann)
        c2.metric("Images", n_imgs)
        c3.metric("Individuals", n_ind)
        if n_ann:
            counts = (
                manifest.groupby("individual_id")
                .agg(boxes=("annotation_id", "count"),
                     encounters=("observation_id", "nunique"))
                .sort_index()
            )
            st.caption("Per-individual (boxes / encounters)")
            st.dataframe(counts, width="stretch", height=180)

        st.divider()
        only_unlabeled = st.checkbox(
            "Only unlabeled images", value=False, key="only_unlabeled"
        )
        show_saved = st.checkbox("Show saved boxes on canvas", value=True)
        st.caption(f"Labels: {MANIFEST_PATH.relative_to(REPO_ROOT)}")

    # ---- Working image list (respecting the unlabeled filter) ---------------
    working = (
        [p for p in all_images if str(relative_image_path(p)) not in labeled_paths]
        if only_unlabeled
        else all_images
    )

    override_image = st.session_state.get("review_override_image")
    if not override_image and source in ("review", "All"):
        resolved = resolve_upload_image_by_encounter_or_path(requested_encounter, requested_image)
        if resolved:
            override_image = str(resolved)

    if override_image and Path(override_image).is_file():
        image_path = Path(override_image)
        if working:
            st.session_state.idx = max(0, min(st.session_state.idx, len(working) - 1))
    elif working:
        st.session_state.idx = max(0, min(st.session_state.idx, len(working) - 1))
        image_path = working[st.session_state.idx]
    else:
        st.success("No images to show with the current filter. 🎉")
        return

    is_uncommitted_review = str(relative_image_path(image_path)).startswith("data/sources/upload/sessions/")
    st.session_state.current_encounter_id = encounter_id_for(image_path)
    st.query_params["source"] = source
    st.query_params["image"] = str(relative_image_path(image_path))
    if st.session_state.current_encounter_id:
        st.query_params["encounter_id"] = st.session_state.current_encounter_id
    loaded_image_rel = relative_image_path(image_path)
    if st.session_state.get("annotation_default_image") != loaded_image_rel:
        st.session_state.annotation_default_image = loaded_image_rel
        upload_record = upload_record_for(image_path)
        default_id = (
            str(upload_record.individual_id)
            if upload_record is not None
            else next_individual_id(manifest)
        )
        apply_annotation_defaults(loaded_image_rel, default_id)

    with st.sidebar:
        st.divider()
        st.subheader("Annotation")
        existing = sorted(
            i for i in manifest["individual_id"].dropna().unique().astype(str) if i
        )
        # New individuals are entered via a dedicated button (below), not buried
        # as an option inside the dropdown.
        if st.session_state.adding_new or not existing:
            if "new_individual_id" not in st.session_state:
                st.session_state.new_individual_id = next_individual_id(manifest)
            individual_id = st.text_input(
                "New individual id", key="new_individual_id"
            ).strip()
            if existing and st.button("↩ Choose existing", width="stretch"):
                st.session_state.adding_new = False
                st.rerun()
        else:
            default_ix = (
                existing.index(st.session_state.last_individual)
                if st.session_state.last_individual in existing
                else 0
            )
            individual_id = st.selectbox("Individual id", existing, index=default_ix)
            if st.button("➕ New individual", width="stretch"):
                st.session_state.adding_new = True
                st.session_state.new_individual_id = next_individual_id(manifest)
                st.rerun()
        body_part = st.radio(
            "Crop target",
            BODY_PARTS,
            horizontal=True,
            help=(
                "Choose throat_portrait instead of throat when the otter is "
                "facing forward and most of its throat is visible."
            ),
        )
        notes = st.text_input("Notes (optional)", value="")

    # Add some spacing at the top
    st.markdown("<br><br> ", unsafe_allow_html=True)


    # ---- Navigation ---------------------------------------------------------
    nav = st.columns([1, 1, 1.3, 1.3, 1.6, 2])
    if nav[0].button("⏮ Prev", width="stretch"):
        st.session_state.pop("review_override_image", None)
        st.query_params.clear()
        st.session_state.idx = max(0, st.session_state.idx - 1)
        st.rerun()
    if nav[1].button("Next ⏭", width="stretch"):
        st.session_state.pop("review_override_image", None)
        st.query_params.clear()
        st.session_state.idx = min(max(0, len(working) - 1), st.session_state.idx + 1)
        st.rerun()
    if nav[2].button("Next unlabeled", width="stretch"):
        st.session_state.pop("review_override_image", None)
        st.query_params.clear()
        for j in range(st.session_state.idx + 1, len(working)):
            if str(relative_image_path(working[j])) not in labeled_paths:
                st.session_state.idx = j
                break
        st.rerun()
    if nav[3].button("Next labeled", width="stretch"):
        st.session_state.pop("review_override_image", None)
        st.query_params.clear()
        for j in range(st.session_state.idx + 1, len(working)):
            if str(relative_image_path(working[j])) in labeled_paths:
                st.session_state.idx = j
                break
        st.rerun()
    jump_max = max(1, len(working))
    jump_val = max(1, min(st.session_state.idx + 1, jump_max))
    jump = nav[4].number_input(
        "Go to #", min_value=1, max_value=jump_max,
        value=jump_val, step=1, label_visibility="collapsed",
    )
    if jump - 1 != st.session_state.idx:
        st.session_state.pop("review_override_image", None)
        st.query_params.clear()
        st.session_state.idx = int(jump) - 1
        st.rerun()
    curr_enc = encounter_id_for(image_path)
    curr_src = source_for_image(image_path)
    src_badge = f" ({curr_src})" if curr_src else ""
    nav[5].markdown(
        f"**{st.session_state.idx + 1} / {len(working)}**{src_badge}  \n"
        f"encounter `{curr_enc}`"
    )

    with st.form("jump-to-encounter", clear_on_submit=False):
        encounter_cols = st.columns([3, 1, 3, 5])
        requested_encounter = encounter_cols[0].text_input(
            "Encounter ID",
            placeholder="Enter encounter ID",
            label_visibility="collapsed",
        )
        jump_to_encounter = encounter_cols[1].form_submit_button(
            "Go", width="stretch"
        )
        use_next_label = encounter_cols[2].form_submit_button(
            "Use next otter ID",
            width="stretch",
            help=(
                "Set the next sequential individual ID as the default for new "
                "manual and generated annotations on this image."
            ),
        )
        if use_next_label:
            st.session_state.pending_annotation_default = {
                "individual_id": next_individual_id(manifest),
                "image_path": relative_image_path(image_path),
            }
            st.rerun()
        if jump_to_encounter:
            st.session_state.pop("review_override_image", None)
            st.query_params.clear()
            req_enc = requested_encounter.strip()
            if not req_enc:
                encounter_cols[3].warning("Enter an encounter ID.")
            else:
                enc_src = source_for_encounter(req_enc)
                prefix_mismatch = (
                    source != "All"
                    and enc_src is not None
                    and enc_src != source
                )

                if prefix_mismatch:
                    # Prefix does not match current filter -> search all sources and switch to "All"
                    global_all = [Path(p) for p in all_source_images()]
                    target_index = encounter_index(global_all, req_enc)
                    if target_index is not None:
                        st.session_state["_pending_source_filter"] = "All"
                        st.session_state.idx = target_index
                        st.session_state.show_all_for_encounter = True
                        st.session_state.current_encounter_id = encounter_id_for(
                            global_all[target_index]
                        )
                        st.rerun()
                    else:
                        encounter_cols[3].error(
                            f"Encounter {req_enc!r} was not found."
                        )
                else:
                    target_index = encounter_index(working, req_enc)
                    if target_index is not None:
                        st.session_state.idx = target_index
                        st.session_state.current_encounter_id = encounter_id_for(
                            working[target_index]
                        )
                        st.rerun()

                    # The unlabeled filter may hide a valid labeled encounter.
                    target_index = encounter_index(all_images, req_enc)
                    if target_index is not None:
                        st.session_state.idx = target_index
                        st.session_state.show_all_for_encounter = True
                        st.session_state.current_encounter_id = encounter_id_for(
                            all_images[target_index]
                        )
                        st.rerun()

                    # If not found in current source filter, search across all sources and switch to "All"
                    if source != "All":
                        global_all = [Path(p) for p in all_source_images()]
                        target_index = encounter_index(global_all, req_enc)
                        if target_index is not None:
                            st.session_state["_pending_source_filter"] = "All"
                            st.session_state.idx = target_index
                            st.session_state.show_all_for_encounter = True
                            st.session_state.current_encounter_id = encounter_id_for(
                                global_all[target_index]
                            )
                            st.rerun()

                    encounter_cols[3].error(
                        f"Encounter {req_enc!r} was not found."
                    )

    # ---- Canvas -------------------------------------------------------------
    try:
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    except OSError as exc:
        st.error(f"Cannot open {image_path}: {exc}")
        return
    orig_w, orig_h = image.size

    rel = relative_image_path(image_path)
    current = manifest[manifest["image_path"] == rel]
    if current.empty and PENDING_CSV.is_file():
        try:
            df_pending = pd.read_csv(PENDING_CSV, dtype=str)
            rel_str = str(rel)
            pending_boxes = df_pending[df_pending["image_path"].astype(str) == rel_str]
            if not pending_boxes.empty:
                pending_rows = []
                for _, prow in pending_boxes.iterrows():
                    try:
                        bx = int(float(prow.get("bbox_x") or 0))
                        by = int(float(prow.get("bbox_y") or 0))
                        bw = int(float(prow.get("bbox_w") or 0))
                        bh = int(float(prow.get("bbox_h") or 0))
                    except ValueError:
                        continue
                    if bw > 0 and bh > 0:
                        pending_rows.append({
                            "annotation_id": prow.get("annotation_id") or uuid.uuid4().hex[:12],
                            "image_path": rel_str,
                            "observation_id": prow.get("upload_id") or "",
                            "latitude": prow.get("latitude") or "",
                            "longitude": prow.get("longitude") or "",
                            "location_source": prow.get("location_source") or "",
                            "individual_id": prow.get("individual_id") or "",
                            "body_part": prow.get("body_part") or "throat_portrait",
                            "bbox_x": bx,
                            "bbox_y": by,
                            "bbox_w": bw,
                            "bbox_h": bh,
                            "img_width": int(float(prow.get("img_width") or orig_w)),
                            "img_height": int(float(prow.get("img_height") or orig_h)),
                            "split": "",
                            "notes": "Pending upload review",
                            "created_at": prow.get("created_at") or "",
                        })
                if pending_rows:
                    current = pd.DataFrame(pending_rows)
        except Exception:
            pass

    detector_negatives = load_detector_negatives()
    current_negatives = detector_negatives[
        detector_negatives["image_path"] == rel
    ]
    predictions: list[dict] = []
    detector_error = None
    if DETECTOR_PATH.is_file():
        try:
            with st.spinner("Generating annotation suggestions…"):
                predictions = detector_predictions(
                    str(image_path),
                    str(DETECTOR_PATH),
                    DETECTOR_PATH.stat().st_mtime_ns,
                )
            predictions = unpromoted_predictions(predictions, current)
            predictions = unpromoted_predictions(
                predictions,
                current_negatives.rename(
                    columns={"negative_id": "annotation_id"}
                ),
            )
        except (ImportError, RuntimeError, OSError) as exc:
            detector_error = str(exc)
    else:
        detector_error = f"Model not found: {DETECTOR_PATH}"


    left, right = st.columns([3, 2])
    canvas_width = CANVAS_WIDTH


    scale = orig_w / canvas_width
    display_h = max(1, round(canvas_width * orig_h / orig_w))
    background = image.resize((canvas_width, display_h))

    # Render the right column first because its generated-box toggle determines
    # the canvas initial drawing in the left column.
    with right:
        if source == "review" or is_uncommitted_review:
            st.subheader("Review & Commit")
            if is_uncommitted_review:
                st.caption(
                    "Transfer this photo from the review session to permanent uploads "
                    "in data/sources/upload/images/."
                )
                if st.button(
                    "Commit Image",
                    type="primary",
                    icon=":material/publish:",
                    key=f"commit-btn-{rel}",
                    width="stretch",
                ):
                    commit_review_image(image_path, manifest)
            else:
                st.success("✅ Image committed to Upload storage.")
                st.caption(f"Encounter ID: `{encounter_id_for(image_path)}`")
            st.divider()

        st.subheader("Annotations on this image")

        if current.empty:
            st.info("None yet.")
        else:
            displayed_annotations = current.sort_values(
                ["individual_id", "body_part"],
                key=lambda values: values.astype(str),
                kind="stable",
            )
            for _, row in displayed_annotations.iterrows():
                box = f"[{row.bbox_x},{row.bbox_y},{row.bbox_w},{row.bbox_h}]"
                annotation_id = str(row.annotation_id)
                st.markdown(
                    f"<small>{box}</small>",
                    unsafe_allow_html=True,
                )
                edit_cols = st.columns([3, 2])
                edited_id = edit_cols[0].text_input(
                    "Individual id",
                    value=str(row.individual_id),
                    key=f"edit-id-{annotation_id}",
                    label_visibility="collapsed",
                ).strip()
                current_body_part = str(row.body_part)
                body_part_index = (
                    BODY_PARTS.index(current_body_part)
                    if current_body_part in BODY_PARTS
                    else 0
                )
                edited_body_part = edit_cols[1].selectbox(
                    "Crop target",
                    BODY_PARTS,
                    index=body_part_index,
                    key=f"edit-part-{annotation_id}",
                    label_visibility="collapsed",
                )
                changed = (
                    edited_id != str(row.individual_id)
                    or edited_body_part != current_body_part
                )
                actions = st.columns([3, 1, 1])
                if actions[0].button(
                    "Save",
                    key=f"save-annotation-{annotation_id}",
                    disabled=not changed or not edited_id,
                    width="stretch",
                ):
                    sync_annotation_update(
                        annotation_id, edited_id, edited_body_part
                    )
                    st.session_state.manifest = update_annotation(
                        manifest, annotation_id, edited_id, edited_body_part
                    )
                    save_manifest(st.session_state.manifest)
                    st.session_state.last_individual = edited_id
                    st.session_state.canvas_nonce += 1
                    st.toast("Annotation updated.")
                    st.rerun()
                if actions[1].button(
                    "⧉",
                    key=f"duplicate-{annotation_id}",
                    help="Duplicate annotation",
                    width="stretch",
                ):
                    st.session_state.manifest, duplicate_id = duplicate_annotation(
                        manifest, annotation_id
                    )
                    save_manifest(st.session_state.manifest)
                    dup_row = st.session_state.manifest[
                        st.session_state.manifest["annotation_id"].astype(str) == duplicate_id
                    ]
                    if not dup_row.empty:
                        sync_annotation_addition(dup_row.to_dict(orient="records"), prune_stale=True)
                    st.session_state.canvas_nonce += 1
                    st.toast(f"Duplicated annotation as {duplicate_id}.")
                    st.rerun()
                if actions[2].button(
                    "🗑", key=f"del-{annotation_id}", width="stretch"
                ):
                    sync_annotation_deletion(annotation_id)
                    st.session_state.manifest = manifest[
                        manifest["annotation_id"].astype(str) != annotation_id
                    ].reset_index(drop=True)
                    save_manifest(st.session_state.manifest)
                    st.session_state.canvas_nonce += 1
                    st.rerun()
                if changed and not edited_id:
                    st.caption("Individual id cannot be empty.")

        st.divider()
        st.subheader("Generated annotations")
        generated_toggle_key = f"show-generated-{rel}"
        has_saved_annotations = not current.empty
        # Annotation availability determines only the initial value. Preserve
        # the user's choice when promoting or deleting annotations.
        if generated_toggle_key not in st.session_state:
            st.session_state[generated_toggle_key] = not has_saved_annotations
        show_generated = st.toggle(
            "Show generated boxes on canvas",
            key=generated_toggle_key,
            disabled=not predictions,
        )
        if detector_error:
            st.warning(
                f"Generated annotations unavailable. {detector_error}\n\n"
                "Run the labeling tool with `uv run streamlit "
                "run src/web/app.py`."
            )
        elif not predictions:
            st.info("No unpromoted suggestions for this image.")
        else:
            st.caption("Dashed boxes are generated suggestions.")
            default_generated_id = st.session_state.get(
                "generated_label_defaults", {}
            ).get(
                rel,
                individual_id
                or st.session_state.get("current_encounter_id", ""),
            )
            generated_label_nonce = st.session_state.get(
                "generated_label_nonces", {}
            ).get(rel, 0)
            for prediction in predictions:
                generated_id = prediction["generated_id"]
                box = (
                    f"[{prediction['bbox_x']},{prediction['bbox_y']},"
                    f"{prediction['bbox_w']},{prediction['bbox_h']}]"
                )
                st.markdown(
                    f"`{prediction['body_part']}` · "
                    f"**{prediction['confidence']:.0%}** · <small>{box}</small>",
                    unsafe_allow_html=True,
                )
                promote_cols = st.columns([3, 2, 2])
                generated_individual_id = promote_cols[0].text_input(
                    "Individual id",
                    value=default_generated_id,
                    key=(
                        f"generated-label-{rel}-{generated_id}-"
                        f"{generated_label_nonce}"
                    ),
                    placeholder="Individual id",
                    label_visibility="collapsed",
                ).strip()
                if promote_cols[1].button(
                    "Promote",
                    key=f"promote-{rel}-{generated_id}",
                    disabled=not generated_individual_id or is_uncommitted_review,
                    help="Commit this image to the Upload pool first to promote annotations." if is_uncommitted_review else None,
                    width="stretch",
                ):

                    row = prediction_to_row(
                        prediction,
                        image_path,
                        (orig_w, orig_h),
                        generated_individual_id,
                    )
                    st.session_state.manifest = pd.concat(
                        [manifest, pd.DataFrame([row])], ignore_index=True
                    )
                    save_manifest(st.session_state.manifest)
                    sync_annotation_addition([row], prune_stale=True)
                    st.session_state.last_individual = generated_individual_id
                    st.session_state.canvas_nonce += 1
                    st.toast(
                        f"Promoted {prediction['body_part']} for "
                        f"{generated_individual_id}."
                    )
                    st.rerun()
                if promote_cols[2].button(
                    "Negative",
                    key=f"negative-{rel}-{generated_id}",
                    disabled=is_uncommitted_review,
                    help=(
                        "Commit this image to the Upload pool first to save detector negatives."
                        if is_uncommitted_review
                        else "Save this rejected box as an empty-label hard-negative crop for future detector training."
                    ),
                    width="stretch",
                ):

                    negative = prediction_to_negative(
                        prediction, image_path, (orig_w, orig_h)
                    )
                    updated_negatives = pd.concat(
                        [detector_negatives, pd.DataFrame([negative])],
                        ignore_index=True,
                    )
                    save_detector_negatives(updated_negatives)
                    st.session_state.canvas_nonce += 1
                    st.toast(
                        f"Added rejected {prediction['body_part']} as a "
                        "detector negative."
                    )
                    st.rerun()

        st.divider()
        st.subheader("Metadata")
        current_encounter_id = encounter_id_for(image_path)
        lat_key = f"label-lat-{current_encounter_id}"
        lon_key = f"label-lon-{current_encounter_id}"
        owner_key = f"label-owner-{current_encounter_id}"
        date_key = f"label-date-{current_encounter_id}"

        meta_info = metadata_fields_for(image_path)

        if (
            lat_key not in st.session_state
            or owner_key not in st.session_state
            or date_key not in st.session_state
        ):
            fallback_latitude, fallback_longitude = encounter_location(
                manifest, current_encounter_id
            )
            if fallback_latitude is None:
                fallback_latitude = meta_info.get("latitude", "")
                fallback_longitude = meta_info.get("longitude", "")

            if is_valid_location(fallback_latitude, fallback_longitude):
                st.session_state[lat_key] = f"{float(fallback_latitude):.6f}"
                st.session_state[lon_key] = f"{float(fallback_longitude):.6f}"
            else:
                st.session_state[lat_key] = ""
                st.session_state[lon_key] = ""

            st.session_state[owner_key] = str(meta_info.get("owner") or "")
            st.session_state[date_key] = str(meta_info.get("captured_at") or "")

        meta_cols = st.columns(2)
        meta_cols[0].text_input("Owner / Author", key=owner_key)
        meta_cols[1].text_input("Date / Time", key=date_key)

        def apply_picked_location(latitude: float, longitude: float) -> None:
            st.session_state[lat_key] = f"{latitude:.6f}"
            st.session_state[lon_key] = f"{longitude:.6f}"

        if st.button(
            "Choose location on map",
            icon=":material/pin_drop:",
            key=f"open-location-picker-{current_encounter_id}",
        ):
            try:
                dialog_latitude = float(st.session_state[lat_key])
                dialog_longitude = float(st.session_state[lon_key])
                if not is_valid_location(dialog_latitude, dialog_longitude):
                    dialog_latitude = dialog_longitude = None
            except (ValueError, TypeError):
                dialog_latitude = dialog_longitude = None
            location_picker_dialog(
                f"label-{current_encounter_id}",
                dialog_latitude,
                dialog_longitude,
                apply_picked_location,
            )

        location_cols = st.columns(2)
        location_cols[0].text_input("Latitude", key=lat_key)
        location_cols[1].text_input("Longitude", key=lon_key)

        if st.button(
            "Save metadata",
            key=f"save-location-{current_encounter_id}",
            width="stretch",
        ):
            lat_str = str(st.session_state.get(lat_key, "")).strip()
            lon_str = str(st.session_state.get(lon_key, "")).strip()
            if not lat_str and not lon_str:
                st.session_state.manifest = update_encounter_location(
                    manifest, current_encounter_id, 0.0, 0.0
                )
                save_manifest(st.session_state.manifest)
                sync_encounter_location(
                    current_encounter_id, 0.0, 0.0, location_source=""
                )
                st.toast(f"Cleared location for encounter {current_encounter_id}.")
                st.rerun()
            else:
                try:
                    new_latitude = float(lat_str)
                    new_longitude = float(lon_str)
                except ValueError:
                    st.warning("Enter valid latitude/longitude numbers first.")
                else:
                    if new_latitude == 0 and new_longitude == 0:
                        st.warning(
                            "Coordinates 0, 0 (Null Island) are treated as no known location. Leave fields blank to clear location."
                        )
                    elif -90 <= new_latitude <= 90 and -180 <= new_longitude <= 180:
                        st.session_state.manifest = update_encounter_location(
                            manifest, current_encounter_id, new_latitude, new_longitude
                        )
                        save_manifest(st.session_state.manifest)
                        sync_encounter_location(
                            current_encounter_id,
                            new_latitude,
                            new_longitude,
                            location_source="map",
                        )
                        st.toast(f"Updated metadata for encounter {current_encounter_id}.")
                        st.rerun()
                    else:
                        st.warning(
                            "Latitude must be -90..90 and longitude -180..180."
                        )




    with left:
        saved_overlay = show_saved and not current.empty
        generated_overlay = show_generated and bool(predictions)
        initial_drawing = combine_drawings(
            saved_boxes_drawing(current, scale) if saved_overlay else None,
            generated_boxes_drawing(predictions, scale)
            if generated_overlay
            else None,
        )
        n_preloaded = (
            (len(current) if saved_overlay else 0)
            + (len(predictions) if generated_overlay else 0)
        )

        stroke = BODY_PART_COLORS[body_part]
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.05)",
            stroke_width=2,
            stroke_color=stroke,
            background_image=background,
            update_streamlit=True,
            height=display_h,
            width=canvas_width,
            drawing_mode="rect",
            initial_drawing=initial_drawing,
            key=(
                f"canvas-{st.session_state.idx}-{int(show_saved)}-"
                f"{int(show_generated)}-{st.session_state.canvas_nonce}-"
                f"{canvas_width}"
            ),
        )
        caption = (
            f"`{rel}` · {orig_w}×{orig_h}px · drag to draw box(es), then Add."
        )
        if saved_overlay:
            caption += f" Showing {len(current)} saved box(es)."
        if generated_overlay:
            caption += (
                f" Showing {len(predictions)} generated box(es) "
                "(dashed)."
            )
        st.caption(caption)

        if is_uncommitted_review:
            st.info("💡 **Review Mode**: Select/enter individual ID and click **Commit Image** on the right sidebar to transfer this photo to the Upload dataset before adding annotations.")
        else:
            add_col, clear_col = st.columns([2, 1])
            if add_col.button("➕ Add box(es) as annotations", type="primary",
                              width="stretch"):
                if not individual_id:
                    st.warning("Enter an individual id first.")
                else:
                    rows = rects_to_rows(
                        canvas_result.json_data, orig_w, orig_h, scale, image_path,
                        individual_id, body_part, notes, skip=n_preloaded,
                    )
                    if not rows:
                        st.warning("No valid box drawn.")
                    else:
                        st.session_state.manifest = pd.concat(
                            [manifest, pd.DataFrame(rows)], ignore_index=True
                        )
                        save_manifest(st.session_state.manifest)
                        sync_annotation_addition(rows, prune_stale=True)
                        st.session_state.last_individual = individual_id
                        st.session_state.adding_new = False
                        st.session_state.canvas_nonce += 1
                        st.toast(f"Added {len(rows)} box(es) for {individual_id}.")
                        st.rerun()
            if clear_col.button("Clear canvas", width="stretch"):
                st.session_state.canvas_nonce += 1
                st.rerun()


    # ---- Links for otters in this image -------------------------------------
    st.divider()
    active_otters = sorted(
        {
            str(value).strip()
            for value in current["individual_id"].dropna().unique()
            if str(value).strip()
        }
    )
    if not active_otters:
        candidate_id = str(individual_id).strip() if individual_id else ""
        if candidate_id:
            otters_to_show = [candidate_id]
            is_saved = False
        else:
            otters_to_show = []
            is_saved = False
    else:
        otters_to_show = active_otters
        is_saved = True

    if not otters_to_show:
        st.info("No otters labeled in this image yet.")
    else:
        for otter_id in otters_to_show:
            matching_other_images = manifest[
                (manifest["individual_id"].astype(str) == otter_id)
                & (manifest["image_path"].astype(str) != str(rel))
            ]
            n_other = matching_other_images["image_path"].nunique()
            if n_other > 0:
                view_text = f"Other images with this otter ({n_other})"
            else:
                view_text = "No other images with this otter"

            view_url = f"view?otter={quote(otter_id, safe='')}"
            query_url = f"query?otter={quote(otter_id, safe='')}"
            query_text = "Query page for this otter"

            suffix = " *(selected for next box)*" if not is_saved else ""
            st.markdown(f"**{otter_id}**{suffix}")
            st.markdown(
                f'- <a href="{view_url}">{view_text}</a>\n'
                f'- <a href="{query_url}">{query_text}</a>',
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()
