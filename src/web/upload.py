#!/usr/bin/env python3
"""Streamlit page for uploading new otter photos and assigning identities.

Run with:
    uv run streamlit run src/web/app.py
"""

from __future__ import annotations

import hashlib
import io
import sys
import uuid
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pillow_heif
import streamlit as st
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()  # Lets PIL open HEIC/HEIF photos from phones.

WEB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB_DIR))
sys.path.insert(0, str(WEB_DIR.parent))
from common.manifest import DATA_DIR, REPO_ROOT, is_valid_location, load_manifest  # noqa: E402
from common.web import (  # noqa: E402
    QUERY_EMBEDDINGS_DIR,
    QUERY_LOCATION_WEIGHT,
    get_match_confidence,
    require_login,
)
from identify.miewid.model import embed_images, get_device, load_model  # noqa: E402
from identify.crop import crop_with_margin  # noqa: E402
from location.geodistance import (  # noqa: E402
    load_location_cache,
    location_score,
    pairwise_haversine_km,
)
from common.max_encounter_distance import haversine_km  # noqa: E402
from location_picker import location_picker, location_picker_dialog  # noqa: E402
from view import BODY_PART_COLORS, draw_annotation_boxes  # noqa: E402

from common.upload import (  # noqa: E402
    PENDING_CSV,
    UPLOAD_IMAGES_DIR,
    UPLOAD_SESSIONS_DIR,
    image_metadata,
)

PENDING_COLUMNS = [
    "upload_id",
    "annotation_id",
    "owner",
    "sighting_date",
    "sighting_time",
    "image_path",
    "captured_at",
    "latitude",
    "longitude",
    "location_source",
    "body_part",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "img_width",
    "img_height",
    "individual_id",
    "match_score",
    "match_distance_km",
    "created_at",
]
DETECTOR_PATH = (
    REPO_ROOT / "data/models/otter-detector/yolo26n-640-final/weights/best.pt"
)
DETECTOR_CONFIDENCE = 0.25
FINETUNED_CHECKPOINT = DATA_DIR / "models/miewid-msv3-otters/best.pt"
CROP_MARGIN = 0.0
TOP_K_MATCHES = 5
UNKNOWN_OTTER = "Unknown otter"


def current_user_name() -> str:
    """Return the signed-in Google name, falling back to the email address."""
    name = str(getattr(st.user, "name", "") or "").strip()
    return name or str(getattr(st.user, "email", "") or "").strip()


def new_upload_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    digest = uuid.uuid4().hex[:8]
    return f"upload-{timestamp}-{digest}"


def file_key(uploaded_file) -> str:
    """Return a stable per-file session-state key for this upload widget value."""
    file_id = getattr(uploaded_file, "file_id", None)
    if file_id:
        return str(file_id)
    digest = hashlib.sha1(
        f"{uploaded_file.name}:{uploaded_file.size}".encode()
    ).hexdigest()
    return digest


@st.cache_resource(show_spinner=False)
def load_detector(model_path: str, model_mtime_ns: int):
    """Load the detector once per model version."""
    del model_mtime_ns  # Included only to invalidate the cache on replacement.
    from ultralytics import YOLO

    return YOLO(model_path)


@st.cache_data(show_spinner="Detecting otter features...")
def detector_predictions(
    image_path: str, model_path: str, model_mtime_ns: int
) -> list[dict]:
    """Return detector boxes in original-image pixel coordinates."""
    model = load_detector(model_path, model_mtime_ns)
    result = model.predict(
        source=image_path, imgsz=640, conf=DETECTOR_CONFIDENCE, verbose=False
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
                "box_id": f"{class_id}-{index}-{round(x1)}-{round(y1)}",
                "body_part": str(result.names[class_id]),
                "confidence": float(confidence),
                "bbox_x": max(0, round(x1)),
                "bbox_y": max(0, round(y1)),
                "bbox_w": max(0, round(x2) - round(x1)),
                "bbox_h": max(0, round(y2) - round(y1)),
            }
        )
    return predictions


@st.cache_resource(show_spinner=False)
def load_identity_model():
    """Load the finetuned MiewID model once."""
    device = get_device()
    checkpoint = FINETUNED_CHECKPOINT if FINETUNED_CHECKPOINT.is_file() else None
    model = load_model(device, checkpoint)
    return model, device


@st.cache_data(show_spinner=False)
def load_gallery(cache_token: tuple[tuple[str, int, int], ...]):
    """Load the throat_portrait gallery embeddings and location cache."""
    del cache_token  # Invalidates the cache when the gallery is rebuilt.
    part_dir = QUERY_EMBEDDINGS_DIR / "throat_portrait"
    index = pd.read_csv(
        part_dir / "index.csv", dtype={"individual_id": str, "observation_id": str}
    )
    embeddings = np.load(part_dir / "embeddings.npy")
    location_coords, location_available = load_location_cache(
        part_dir, len(index)
    )
    return index, embeddings, location_coords, location_available


def gallery_cache_token() -> tuple[tuple[str, int, int], ...]:
    part_dir = QUERY_EMBEDDINGS_DIR / "throat_portrait"
    token = []
    for name in ("index.csv", "embeddings.npy", "locations.csv"):
        path = part_dir / name
        if path.is_file():
            stat = path.stat()
            token.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(token)


def top_matches(
    embedding: np.ndarray, lonlat: tuple[float, float] | None
) -> list[dict]:
    """Return the top-K nearest gallery crops, blended with location when known."""
    index, embeddings, location_coords, location_available = load_gallery(
        gallery_cache_token()
    )
    embedding = np.asarray(embedding, dtype=np.float32)
    if len(index) == 0:
        return []
    image_similarity = embeddings @ embedding
    combined = image_similarity.copy()
    location_similarity = np.full(len(index), np.nan)
    available = np.zeros(len(index), dtype=bool)
    if lonlat is not None:
        available = location_available.copy()
        distances = pairwise_haversine_km(
            np.array([lonlat], dtype=np.float64), location_coords[available]
        )[0]
        raw_location = location_score(distances)
        location_similarity[available] = raw_location
        combined[available] = (
            (1 - QUERY_LOCATION_WEIGHT) * image_similarity[available]
            + QUERY_LOCATION_WEIGHT * raw_location
        )
    order = np.argsort(-combined)[:TOP_K_MATCHES]
    matches = []
    for rank in order:
        row = index.iloc[int(rank)]
        distance_km = None
        row_latitude = pd.to_numeric(pd.Series([row.get("latitude")]), errors="coerce").iloc[0]
        row_longitude = pd.to_numeric(pd.Series([row.get("longitude")]), errors="coerce").iloc[0]
        if lonlat is not None and is_valid_location(row_latitude, row_longitude):
            distance_km = haversine_km(lonlat, (float(row_longitude), float(row_latitude)))
        matches.append(
            {
                "individual_id": str(row.individual_id),
                "crop_path": str(row.crop_path),
                "image_score": float(image_similarity[rank]),
                "location_score": (
                    float(location_similarity[rank]) if available[rank] else None
                ),
                "distance_km": distance_km,
                "combined_score": float(combined[rank]),
            }
        )
    return matches


@st.cache_data(show_spinner=False)
def crop_for_box(
    rel_path: str, bbox: tuple[int, int, int, int], mtime_ns: int
) -> Image.Image:
    """Return the margin-padded crop for one detected box."""
    del mtime_ns  # Included only to invalidate the cache on replacement.
    with Image.open(REPO_ROOT / rel_path) as source_image:
        return crop_with_margin(source_image.convert("RGB"), *bbox, CROP_MARGIN)


@st.cache_data(show_spinner="Embedding throat portrait crop...")
def embed_box(
    rel_path: str, bbox: tuple[int, int, int, int], mtime_ns: int
) -> np.ndarray:
    """Return the MiewID embedding for one detected throat_portrait box."""
    crop = crop_for_box(rel_path, bbox, mtime_ns)
    model, device = load_identity_model()
    return embed_images(model, [crop], device)[0]


@st.cache_data(show_spinner=False)
def matches_for_box(
    rel_path: str,
    bbox: tuple[int, int, int, int],
    mtime_ns: int,
    lonlat: tuple[float, float] | None,
) -> list[dict]:
    """Return the top matches for one detected throat_portrait box."""
    embedding = embed_box(rel_path, bbox, mtime_ns)
    return top_matches(embedding, lonlat)


def nearest_box(box: dict, candidates: list[dict]) -> dict | None:
    """Return the candidate box whose center is closest to ``box``'s center."""
    if not candidates:
        return None
    center_x = box["bbox_x"] + box["bbox_w"] / 2
    center_y = box["bbox_y"] + box["bbox_h"] / 2

    def squared_distance(candidate: dict) -> float:
        candidate_x = candidate["bbox_x"] + candidate["bbox_w"] / 2
        candidate_y = candidate["bbox_y"] + candidate["bbox_h"] / 2
        return (center_x - candidate_x) ** 2 + (center_y - candidate_y) ** 2

    return min(candidates, key=squared_distance)


def append_pending_rows(rows: list[dict]) -> None:
    """Atomically append newly submitted annotation rows to the pending CSV."""
    frame = pd.DataFrame(rows, columns=PENDING_COLUMNS)
    if PENDING_CSV.is_file() and PENDING_CSV.stat().st_size > 0:
        try:
            existing = pd.read_csv(PENDING_CSV, dtype=str)
        except Exception:
            existing = pd.DataFrame(columns=PENDING_COLUMNS)
    else:
        existing = pd.DataFrame(columns=PENDING_COLUMNS)
    combined = pd.concat([existing, frame], ignore_index=True)
    PENDING_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = PENDING_CSV.with_name(f".{PENDING_CSV.name}.tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(PENDING_CSV)

    try:
        from common.sources import append_review_records
        review_records = []
        seen_paths = set()
        for r in rows:
            img_p = str(r.get("image_path") or "").strip()
            if img_p and img_p not in seen_paths:
                seen_paths.add(img_p)
                p = Path(img_p)
                dt_str = str(r.get("sighting_date") or "").strip()
                tm_str = str(r.get("sighting_time") or "").strip()
                observed = f"{dt_str}T{tm_str}" if dt_str and tm_str else (dt_str or str(r.get("captured_at") or ""))
                review_records.append({
                    "source": "upload",
                    "encounter_id": str(r.get("upload_id") or ""),
                    "photo_id": p.stem,
                    "image_path": img_p,
                    "image_url": "",
                    "page_url": "",
                    "observed_at": observed,
                    "observer": str(r.get("owner") or ""),
                    "latitude": str(r.get("latitude") or ""),
                    "longitude": str(r.get("longitude") or ""),
                    "location_source": str(r.get("location_source") or ""),
                    "locality": "",
                    "license": "",
                    "quality_grade": "",
                })
        if review_records:
            append_review_records(review_records)
    except Exception as exc:
        print(f"Error appending to results_review.csv: {exc}", file=sys.stderr)


def format_candidate_label(match: dict) -> str:
    label = f"{match['individual_id']} · score {match['combined_score']:.3f}"
    if match.get("image_score") is not None:
        label += f" (img {match['image_score']:.3f}"
        if match.get("location_score") is not None:
            label += f", loc {match['location_score']:.3f}"
        label += ")"
    if match.get("distance_km") is not None:
        if match["distance_km"] < 1.0:
            label += f" ({match['distance_km'] * 1000:.0f} m)"
        else:
            label += f" ({match['distance_km']:.0f} km)"
    return label


def parsed_lonlat(key: str) -> tuple[float, float] | None:
    try:
        latitude = float(st.session_state[f"lat_{key}"])
        longitude = float(st.session_state[f"lon_{key}"])
    except (KeyError, ValueError):
        return None
    if not is_valid_location(latitude, longitude):
        return None
    return longitude, latitude


def render_location_controls(key: str, exif_latitude: float | None, exif_longitude: float | None) -> None:
    """Render editable lat/lon inputs plus a popup click-to-pick map."""
    lat_key, lon_key, source_key = f"lat_{key}", f"lon_{key}", f"locsrc_{key}"
    if lat_key not in st.session_state:
        has_exif = is_valid_location(exif_latitude, exif_longitude)
        st.session_state[lat_key] = f"{float(exif_latitude):.6f}" if has_exif else ""
        st.session_state[lon_key] = f"{float(exif_longitude):.6f}" if has_exif else ""
        st.session_state[source_key] = "exif" if has_exif else ""

    def apply_picked_location(latitude: float, longitude: float) -> None:
        st.session_state[lat_key] = f"{latitude:.6f}"
        st.session_state[lon_key] = f"{longitude:.6f}"
        st.session_state[source_key] = "map"

    location_columns = st.columns([2, 2, 2], vertical_alignment="bottom")
    location_columns[0].text_input("Latitude", key=lat_key)
    location_columns[1].text_input("Longitude", key=lon_key)

    has_location = parsed_lonlat(key) is not None
    button_type = "secondary" if has_location else "primary"

    if location_columns[2].button(
        "Choose location on map",
        icon=":material/pin_drop:",
        key=f"open_picker_{key}",
        type=button_type,
    ):
        try:
            current_latitude = float(st.session_state[lat_key])
            current_longitude = float(st.session_state[lon_key])
        except ValueError:
            current_latitude = current_longitude = None
        location_picker_dialog(key, current_latitude, current_longitude, apply_picked_location)


def save_uploaded_file(uploaded_file, batch_id: str) -> Path | None:
    """Save an uploaded file as JPEG into the session directory if not already saved."""
    session_dir = UPLOAD_SESSIONS_DIR / batch_id
    dest_path = session_dir / f"{Path(uploaded_file.name).stem}.jpg"
    if not dest_path.is_file():
        session_dir.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(io.BytesIO(uploaded_file.getvalue())) as source_image:
                transposed = ImageOps.exif_transpose(source_image)
                exif = transposed.getexif()
                transposed.convert("RGB").save(
                    dest_path, format="JPEG", quality=95, **({"exif": exif} if exif else {})
                )
        except Exception as exc:
            st.error(f"Could not read {uploaded_file.name}: {exc}")
            return None
    return dest_path





@st.dialog("Select Otter Identity", width="large")
def identity_selection_dialog(
    key: str,
    box_id: str,
    rel_path: str,
    bbox: tuple[int, int, int, int],
    mtime_ns: int,
    matches: list[dict],
) -> None:
    match_key = f"match_{key}_{box_id}"
    source_key = f"source_{match_key}"
    crop = crop_for_box(rel_path, bbox, mtime_ns)

    level, blurb_text, circle_icon = get_match_confidence(matches)

    if match_key not in st.session_state:
        if level == 1:
            st.session_state[match_key] = 0
        else:
            st.session_state[match_key] = "unknown"
        st.session_state[source_key] = "model"

    current_val = st.session_state[match_key]

    st.caption(
        "Review candidate matches from the gallery and select an otter identity."
    )

    crop_col, info_col = st.columns([1, 3], vertical_alignment="center")
    with crop_col:
        st.image(crop, width="stretch", caption="Uploaded Crop")
    with info_col:
        if matches:
            top_match = matches[0]
            st.markdown(f"**Top Candidate (Rank 1)**: `{top_match['individual_id']}`")
            st.markdown(f"**Top Combined Score**: `{top_match['combined_score']:.3f}`")
            st.markdown(f"**Image Score**: `{top_match['image_score']:.3f}`")
        else:
            st.info("No gallery candidate matches available for comparison.")

    st.divider()
    st.subheader("Top-5 Matches")

    top_matches = matches[:5] if matches else []
    if top_matches:
        cols = st.columns(len(top_matches))
        for idx, (col, match) in enumerate(zip(cols, top_matches)):
            with col:
                st.markdown(f"**Rank {idx + 1}**")
                match_crop = REPO_ROOT / match["crop_path"]
                if match_crop.is_file():
                    st.image(str(match_crop), width="stretch")
                st.markdown(f"**`{match['individual_id']}`**")
                st.write(f"**Combined**: `{match['combined_score']:.3f}`")
                st.write(f"**Image score**: `{match['image_score']:.3f}`")
                loc_str = (
                    f"`{match['location_score']:.3f}`"
                    if match["location_score"] is not None
                    else "N/A"
                )
                st.write(f"**Location score**: {loc_str}")
                dist_str = (
                    f"`{match['distance_km']:.0f} km`"
                    if match["distance_km"] is not None
                    else "N/A"
                )
                st.write(f"**Distance**: {dist_str}")
    else:
        st.info("No labeled throat_portrait gallery is available yet.")

    st.divider()

    if matches:
        st.markdown(f"#### {circle_icon} {blurb_text}")

    options = ["unknown"] + list(range(len(top_matches)))

    def format_option(opt):
        if opt == "unknown":
            return f"❓ {UNKNOWN_OTTER}"
        m = top_matches[opt]
        return f"Rank {opt + 1}: {m['individual_id']} — Combined: {m['combined_score']:.3f}"

    selected = st.radio(
        "Choose identity for this box:",
        options,
        index=options.index(current_val) if current_val in options else 0,
        format_func=format_option,
        key=f"modal_radio_{match_key}",
    )

    btn_col1, btn_col2 = st.columns([1, 1])
    if btn_col1.button("Confirm Selection", type="primary", width="stretch"):
        st.session_state[match_key] = selected
        st.session_state[source_key] = "user"
        st.rerun()
    if btn_col2.button("Cancel", width="stretch"):
        st.rerun()


def render_upload_item(
    uploaded_file, batch_id: str, known_individual_ids: list[str]
) -> None:
    key = file_key(uploaded_file)
    dest_path = save_uploaded_file(uploaded_file, batch_id)
    if dest_path is None:
        return
    rel_path = str(dest_path.relative_to(REPO_ROOT))

    captured_at, exif_latitude, exif_longitude = image_metadata(dest_path)
    predictions = detector_predictions(
        rel_path, str(DETECTOR_PATH), DETECTOR_PATH.stat().st_mtime_ns
    )
    with Image.open(dest_path) as source_image:
        width, height = source_image.size

    st.subheader(uploaded_file.name)
    boxes_frame = pd.DataFrame(
        predictions, columns=["bbox_x", "bbox_y", "bbox_w", "bbox_h", "body_part"]
    )
    st.image(draw_annotation_boxes(rel_path, boxes_frame, True), width=800)
    if predictions:
        legend = " &nbsp; ".join(
            f'<span style="color:{color}">&#9632;</span> {part.replace("_", " ")}'
            for part, color in BODY_PART_COLORS.items()
        )
        st.markdown(
            f'<p style="font-size:0.85rem">{legend}</p>', unsafe_allow_html=True
        )
    else:
        st.caption("No otter features were detected in this photo.")

    throat_portrait_boxes = [
        p for p in predictions if p["body_part"] == "throat_portrait"
    ]
    lonlat = parsed_lonlat("batch")
    dest_mtime_ns = dest_path.stat().st_mtime_ns

    if throat_portrait_boxes:
        for box in throat_portrait_boxes:
            bbox = (box["bbox_x"], box["bbox_y"], box["bbox_w"], box["bbox_h"])
            match_key = f"match_{key}_{box['box_id']}"
            source_key = f"source_{match_key}"
            matches = matches_for_box(rel_path, bbox, dest_mtime_ns, lonlat)
            level, blurb_text, circle_icon = get_match_confidence(matches)

            if match_key not in st.session_state:
                if level == 1:
                    st.session_state[match_key] = 0
                else:
                    st.session_state[match_key] = "unknown"
                st.session_state[source_key] = "model"
            elif st.session_state.get(source_key, "model") == "model":
                if level == 1:
                    st.session_state[match_key] = 0
                else:
                    st.session_state[match_key] = "unknown"

            current_val = st.session_state[match_key]

            if current_val == "unknown":
                selected_name = UNKNOWN_OTTER
            elif isinstance(current_val, int) and matches and current_val < len(matches):
                selected_name = matches[current_val]["individual_id"]
            else:
                selected_name = UNKNOWN_OTTER

            selected_display = f"**Selected identity**: {selected_name}"
            model_display = f"**Model result**: {circle_icon} {blurb_text}"

            info_col1, info_col2 = st.columns([3, 1], vertical_alignment="center")
            info_col1.markdown(f"{selected_display}  \n{model_display}")
            if info_col2.button(
                "Select identity",
                icon=":material/person_search:",
                key=f"open_dialog_{match_key}",
                type="secondary",
            ):
                identity_selection_dialog(
                    key, box["box_id"], rel_path, bbox, dest_mtime_ns, matches
                )
    else:
        st.selectbox(
            "Otter identity for this photo",
            [UNKNOWN_OTTER] + known_individual_ids,
            key=f"identity_{key}",
        )

    st.session_state.setdefault("upload_item_meta", {})[key] = {
        "rel_path": rel_path,
        "captured_at": captured_at,
        "width": width,
        "height": height,
        "predictions": predictions,
        "throat_portrait_boxes": throat_portrait_boxes,
    }


def build_submission_rows(batch_id: str, item_keys: list[str]) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    owner = str(st.session_state.get("upload_owner", "")).strip()
    sighting_date = st.session_state.get("upload_sighting_date")
    sighting_time = st.session_state.get("upload_sighting_time")
    rows: list[dict] = []
    all_meta = st.session_state.get("upload_item_meta", {})
    for key in item_keys:
        meta = all_meta.get(key)
        if meta is None:
            continue
        latitude = st.session_state.get("lat_batch", "")
        longitude = st.session_state.get("lon_batch", "")
        location_source = st.session_state.get("locsrc_batch", "")
        if latitude and longitude and not location_source:
            location_source = "user"
        captured_at = meta["captured_at"]

        tp_boxes = meta["throat_portrait_boxes"]
        lonlat = parsed_lonlat("batch")
        dest_mtime_ns = (REPO_ROOT / meta["rel_path"]).stat().st_mtime_ns
        tp_matches = {
            box["box_id"]: matches_for_box(
                meta["rel_path"],
                (box["bbox_x"], box["bbox_y"], box["bbox_w"], box["bbox_h"]),
                dest_mtime_ns,
                lonlat,
            )
            for box in tp_boxes
        }
        tp_choice: dict[str, tuple[str, float | None, float | None]] = {}
        for box in tp_boxes:
            selection = st.session_state.get(f"match_{key}_{box['box_id']}", "unknown")
            if selection == "unknown":
                tp_choice[box["box_id"]] = ("", None, None)
            else:
                match = tp_matches[box["box_id"]][selection]
                tp_choice[box["box_id"]] = (
                    match["individual_id"],
                    match["combined_score"],
                    match["distance_km"],
                )

        manual_identity = None
        if not tp_boxes:
            selected = st.session_state.get(f"identity_{key}", UNKNOWN_OTTER)
            manual_identity = "" if selected == UNKNOWN_OTTER else selected

        for box in meta["predictions"]:
            if box["body_part"] == "throat_portrait":
                individual_id, match_score, distance_km = tp_choice[box["box_id"]]
            elif tp_boxes:
                nearest = nearest_box(box, tp_boxes)
                individual_id = tp_choice[nearest["box_id"]][0] if nearest else ""
                match_score = distance_km = None
            else:
                individual_id = manual_identity or ""
                match_score = distance_km = None

            rows.append(
                {
                    "upload_id": batch_id,
                    "annotation_id": uuid.uuid4().hex[:12],
                    "owner": owner,
                    "sighting_date": (
                        sighting_date.isoformat()
                        if isinstance(sighting_date, date)
                        else ""
                    ),
                    "sighting_time": (
                        sighting_time.isoformat(timespec="seconds")
                        if isinstance(sighting_time, time)
                        else ""
                    ),
                    "image_path": meta["rel_path"],
                    "captured_at": (
                        captured_at.isoformat(timespec="seconds") if captured_at else ""
                    ),
                    "latitude": latitude,
                    "longitude": longitude,
                    "location_source": location_source,
                    "body_part": box["body_part"],
                    "bbox_x": box["bbox_x"],
                    "bbox_y": box["bbox_y"],
                    "bbox_w": box["bbox_w"],
                    "bbox_h": box["bbox_h"],
                    "img_width": meta["width"],
                    "img_height": meta["height"],
                    "individual_id": individual_id,
                    "match_score": "" if match_score is None else round(match_score, 4),
                    "match_distance_km": (
                        "" if distance_km is None else round(distance_km, 1)
                    ),
                    "created_at": now,
                }
            )
    return rows


def reset_batch() -> None:
    """Clear session state for the current batch so a fresh upload can begin."""
    for key in list(st.session_state.keys()):
        if key.startswith(
            ("lat_", "lon_", "locsrc_", "picker_", "match_", "identity_", "source_")
        ) or key in (
            "upload_batch_id",
            "upload_item_meta",
            "upload_owner",
            "upload_sighting_date",
            "upload_sighting_time",
        ):
            del st.session_state[key]
    st.session_state["uploader_generation"] = (
        st.session_state.get("uploader_generation", 0) + 1
    )


def main() -> None:
    require_login()
    st.title("Upload photos")

    if st.session_state.get("submission_success"):
        st.success("🎉 **Thank you for your submission!**")
        st.info(
            "Your uploaded photos and metadata have been received. "
            "They will be reviewed before they are available on the site."
        )
        if st.button("Upload more images", type="primary", icon=":material/upload:"):
            st.session_state["submission_success"] = False
            reset_batch()
            st.rerun()
        return

    st.caption(
        "Upload one or more photos from the same encounter (photos taken at the same time and location). "
        "If enough of the otter is visible, the form will try to determine the identity of the otter. "
        "Confirm the identity if known, or select 'unknown' if the otter is not identifiable from the photos. "
        "All submissions will be reviewed by project staff before being added to the database."
    )

    generation = st.session_state.get("uploader_generation", 0)
    uploaded_files = st.file_uploader(
        "Photos",
        type=["jpg", "jpeg", "png", "heic", "heif", "webp"],
        accept_multiple_files=True,
        key=f"uploader_{generation}",
    )
    if not uploaded_files:
        st.info("Select one or more photos to begin.")
        return

    if "upload_batch_id" not in st.session_state:
        st.session_state["upload_batch_id"] = new_upload_id()
    batch_id = st.session_state["upload_batch_id"]

    if "upload_owner" not in st.session_state:
        st.session_state["upload_owner"] = current_user_name()
    if "upload_sighting_date" not in st.session_state:
        st.session_state["upload_sighting_date"] = datetime.now().date()
    if "upload_sighting_time" not in st.session_state:
        st.session_state["upload_sighting_time"] = (
            datetime.now().time().replace(microsecond=0)
        )

    metadata_columns = st.columns(3)
    metadata_columns[0].text_input("Owner", key="upload_owner")
    metadata_columns[1].date_input("Sighting date", key="upload_sighting_date")
    metadata_columns[2].time_input("Sighting time", key="upload_sighting_time")

    default_lat: float | None = None
    default_lon: float | None = None
    for uploaded_file in uploaded_files:
        dest_path = save_uploaded_file(uploaded_file, batch_id)
        if dest_path:
            _, plat, plon = image_metadata(dest_path)
            if is_valid_location(plat, plon):
                default_lat, default_lon = plat, plon
                break

    render_location_controls("batch", default_lat, default_lon)

    known_individual_ids = sorted(
        load_manifest()["individual_id"].dropna().astype(str).unique()
    )

    item_keys = []
    for uploaded_file in uploaded_files:
        key = file_key(uploaded_file)
        item_keys.append(key)
        with st.container(border=True):
            render_upload_item(uploaded_file, batch_id, known_individual_ids)

    st.divider()
    error_container = st.container()
    if st.button("Submit upload", type="primary"):
        errors: list[str] = []
        owner = str(st.session_state.get("upload_owner", "")).strip()
        if not owner:
            errors.append("Please enter an Owner name.")
        if not st.session_state.get("upload_sighting_date"):
            errors.append("Please select a Sighting date.")
        if not st.session_state.get("upload_sighting_time"):
            errors.append("Please select a Sighting time.")

        lat_str = str(st.session_state.get("lat_batch", "")).strip()
        lon_str = str(st.session_state.get("lon_batch", "")).strip()
        if not lat_str or not lon_str:
            errors.append(
                "Please provide both Latitude and Longitude for the sighting."
            )
        else:
            try:
                lat_val = float(lat_str)
                lon_val = float(lon_str)
                if not is_valid_location(lat_val, lon_val):
                    if lat_val == 0 and lon_val == 0:
                        errors.append(
                            "Coordinates 0, 0 (Null Island) cannot be used as a valid sighting location."
                        )
                    else:
                        errors.append(
                            "Latitude must be between -90 and 90, and Longitude between -180 and 180."
                        )
            except ValueError:
                errors.append("Latitude and Longitude must be valid numbers.")

        if errors:
            with error_container:
                for error in errors:
                    st.error(error)
        else:
            rows = build_submission_rows(batch_id, item_keys)
            append_pending_rows(rows)
            reset_batch()
            st.session_state["submission_success"] = True
            st.rerun()


if __name__ == "__main__":
    main()
