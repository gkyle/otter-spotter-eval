#!/usr/bin/env python3
"""Streamlit page for browsing every local photo from one observation.

Run with:
    uv run streamlit run src/web/app.py
"""

from __future__ import annotations

import html
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
import streamlit as st

WEB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB_DIR))
sys.path.insert(0, str(WEB_DIR.parent))
from common.manifest import REPO_ROOT, load_manifest  # noqa: E402
from common.sources import load_sources_results, results_file_token  # noqa: E402
from common.web import encounter_url_for_encounter  # noqa: E402
from common.web import label_url_for_encounter  # noqa: E402
from view import (  # noqa: E402
    BODY_PART_COLORS,
    BODY_PARTS,
    draw_annotation_boxes,
    image_url_for,
    observation_photos,
    open_image,
    photo_details,
    photo_rating,
    resolve_image_source,
    stars,
    valid_location_mask,
)

import streamlit.components.v2 as ccv2

RESULTS_PATH = REPO_ROOT / "data/results.csv"

CAROUSEL_CSS = """
.encounter-carousel-wrapper {
    width: 100%;
    display: flex;
    justify-content: center;
}
.encounter-carousel-container {
    height: 100px;
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    justify-content: safe center;
    gap: 12px;
    overflow-x: auto;
    overflow-y: hidden;
    box-sizing: border-box;
    scrollbar-width: thin;
    scrollbar-color: #4b8b8b rgba(0, 0, 0, 0.05);
    padding: 0 8px;
    max-width: 100%;
    width: max-content;
}
.encounter-carousel-container::-webkit-scrollbar {
    height: 6px;
}
.encounter-carousel-container::-webkit-scrollbar-thumb {
    background-color: #4b8b8b;
    border-radius: 3px;
}
.encounter-carousel-container::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.05);
    border-radius: 3px;
}
.encounter-thumb-card {
    flex: 0 0 auto;
    height: 80px;
    border-radius: 8px;
    overflow: hidden;
    border: 3px solid transparent;
    box-sizing: border-box;
    transition: all 0.18s ease-in-out;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.04);
    text-decoration: none;
}
.encounter-thumb-card:hover {
    opacity: 1 !important;
    border-color: #4b8b8b;
    transform: translateY(-2px);
}
.encounter-thumb-card.active {
    border-color: #4b8b8b;
    opacity: 1;
    box-shadow: 0 0 0 1px #4b8b8b, 0 4px 12px rgba(75, 139, 139, 0.35);
}
.encounter-thumb-card.inactive {
    opacity: 0.65;
}
.encounter-thumb-card img {
    height: 100%;
    width: auto;
    max-width: 260px;
    object-fit: contain;
    display: block;
    border-radius: 5px;
    user-select: none;
    -webkit-user-drag: none;
}
"""

CAROUSEL_HTML = """
<div class="encounter-carousel-wrapper">
    <div id="carousel-container" class="encounter-carousel-container"></div>
</div>
"""

CAROUSEL_JS = """\
export default function (component) {
    const { data, parentElement, setTriggerValue } = component;
    const container = parentElement.querySelector("#carousel-container");
    if (!container) return;

    container.innerHTML = "";
    const items = data?.items || [];
    const selectedIdx = data?.selectedIndex ?? 0;

    items.forEach((item, idx) => {
        const card = document.createElement("div");
        card.className = "encounter-thumb-card " + (idx === selectedIdx ? "active" : "inactive");
        card.title = "Photo " + (idx + 1);
        card.setAttribute("role", "button");
        card.setAttribute("tabindex", "0");

        const img = document.createElement("img");
        img.src = item.src;
        img.alt = "Photo " + (idx + 1);
        img.loading = "lazy";
        card.appendChild(img);

        const selectThis = (e) => {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }
            if (idx !== selectedIdx) {
                setTriggerValue("select_photo", idx);
            }
        };

        card.onclick = selectThis;
        card.onkeydown = (e) => {
            if (e.key === "Enter" || e.key === " ") {
                selectThis(e);
            }
        };

        container.appendChild(card);
    });
}
"""

encounter_thumbnail_carousel = ccv2.component(
    "encounter_thumbnail_carousel",
    html=CAROUSEL_HTML,
    css=CAROUSEL_CSS,
    js=CAROUSEL_JS,
    isolate_styles=False,
)


@st.cache_data
def load_encounter_results(encounter_id: str, token: str = "") -> pd.DataFrame:
    """Read encounter records from data/results.csv."""
    if not RESULTS_PATH.is_file():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_PATH, dtype=str)
    return df[df["encounter_id"] == encounter_id]


@st.cache_data
def load_all_encounter_photo_counts(token: str = "") -> pd.Series:
    """Return series of photo counts indexed by encounter_id."""
    results_counts = pd.Series(dtype=int)
    if RESULTS_PATH.is_file():
        try:
            res = pd.read_csv(RESULTS_PATH, dtype=str, usecols=["encounter_id", "photo_id"])
            res = res.dropna(subset=["encounter_id"])
            results_counts = res.groupby("encounter_id")["photo_id"].nunique()
        except Exception:
            pass
    try:
        manifest = load_manifest()
        manifest_counts = (
            manifest.dropna(subset=["observation_id"])
            .groupby("observation_id")["image_path"]
            .nunique()
        )
    except Exception:
        manifest_counts = pd.Series(dtype=int)

    all_counts = results_counts.combine(manifest_counts, max, fill_value=0).astype(int)
    all_counts = all_counts[all_counts > 0]
    return all_counts.sort_index()


@st.cache_data
def load_annotated_encounter_ids() -> set[str]:
    """Return set of encounter IDs that have annotations in manifest."""
    try:
        manifest = load_manifest()
        obs = manifest.dropna(subset=["observation_id"])["observation_id"].astype(str)
        return set(obs.unique())
    except Exception:
        return set()


def store_selected_encounter_in_url() -> None:
    """Persist the completed selector change without racing the widget rerun."""
    st.query_params["observation"] = st.session_state.selected_encounter
    st.query_params.pop("photo", None)


def render_single_pin_map(
    lat: float, lon: float, title: str, detail: str = ""
) -> None:
    """Render a Leaflet map centered on a single point with one pin."""
    safe_title = html.escape(title)
    safe_detail = html.escape(detail)
    detail_line = f"<br>{safe_detail}" if safe_detail else ""
    leaflet_html = rf"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <style>
        html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
        #map {{ position: relative; overflow: hidden; border-radius: 0.5rem; background: #e7ecef; }}
        a {{ color: #4b8b8b; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
      </style>
    </head>
    <body>
      <div id="map"></div>
      <script>
        const map = L.map('map', {{
          center: [{lat:.6f}, {lon:.6f}],
          zoom: 12,
          scrollWheelZoom: false,
          zoomAnimation: false
        }});
        L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
          maxZoom: 19,
          attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);
        const marker = L.circleMarker([{lat:.6f}, {lon:.6f}], {{
          radius: 9,
          color: '#ffffff',
          weight: 2,
          fillColor: '#198287',
          fillOpacity: 0.95
        }}).addTo(map);
        marker.bindPopup('<b>{safe_title}</b>{detail_line}').openPopup();
        let resizeTimer;
        const resizeMap = () => {{
          clearTimeout(resizeTimer);
          resizeTimer = setTimeout(() => map.invalidateSize({{ pan: false }}), 50);
        }};
        new ResizeObserver(resizeMap).observe(document.getElementById('map'));
        window.addEventListener('load', resizeMap);
        setTimeout(resizeMap, 200);
      </script>
    </body>
    </html>
    """
    st.iframe(leaflet_html, width="stretch", height=280)


def main() -> None:
    token = results_file_token()
    photo_counts = load_all_encounter_photo_counts(token)
    if photo_counts.empty:
        st.title("Encounter")
        st.info("No encounters are available yet.")
        return

    annotated_encounter_ids = load_annotated_encounter_ids()
    requested_observation = str(st.query_params.get("observation", ""))
    requested_otter = str(st.query_params.get("otter", ""))

    if requested_observation and requested_observation not in photo_counts.index:
        result_rows = load_sources_results()
        encounter_values = result_rows["encounter_id"].astype(str)
        matches = result_rows[
            (encounter_values == requested_observation)
            | encounter_values.str.endswith(f"_{requested_observation}")
            | encounter_values.str.endswith(f"-{requested_observation}")
        ]
        if not matches.empty:
            requested_observation = str(matches.iloc[0]["encounter_id"])
        else:
            photo_counts[requested_observation] = 1

    if "encounter_show_all" not in st.session_state:
        st.session_state.encounter_show_all = False

    if (
        requested_observation in photo_counts.index
        and (
            requested_observation != st.session_state.get("encounter_applied_url_observation", "")
            # Navigating to another page clears this widget's state, so the
            # URL observation must be reapplied even if it matches the last-applied value.
            or "selected_encounter" not in st.session_state
        )
    ):
        st.session_state.selected_encounter = requested_observation
        # A direct link to an unannotated encounter should remain viewable and selectable.
        if requested_observation not in annotated_encounter_ids:
            st.session_state.encounter_show_all = True
    st.session_state.encounter_applied_url_observation = requested_observation

    show_all = st.session_state.encounter_show_all
    eligible = (
        photo_counts
        if show_all
        else photo_counts[photo_counts.index.isin(annotated_encounter_ids)]
    )
    encounter_ids = eligible.index.astype(str).tolist()

    if not encounter_ids:
        st.title("Encounter")
        st.info(
            "No encounters have labeled annotations yet. "
            "Turn on “Show all” to view all encounters."
        )
        return

    if st.session_state.get("selected_encounter") not in encounter_ids:
        st.session_state.selected_encounter = encounter_ids[0]
        st.query_params["observation"] = encounter_ids[0]
        st.query_params.pop("photo", None)

    observation_id = st.session_state.selected_encounter

    st.title(f"Encounter: {observation_id}")

    with st.container(horizontal=True, vertical_alignment="center"):
        st.write("Encounter")
        st.selectbox(
            "Encounter",
            encounter_ids,
            key="selected_encounter",
            format_func=lambda value: (
                f"{value} ({photo_counts[value]} "
                f"{'photo' if photo_counts[value] == 1 else 'photos'})"
            ),
            on_change=store_selected_encounter_in_url,
            label_visibility="collapsed",
            width=280,
        )
        st.toggle(
            "Show all",
            key="encounter_show_all",
            help=(
                "Off: only encounters with labeled annotations. "
                "On: every encounter."
            ),
        )

    st.html("""
    <style>
    div[data-testid="stImage"] img {
        max-height: 520px;
        object-fit: contain;
        margin: 0 auto;
        border-radius: 6px;
    }
    </style>
    """)

    # Load annotations for this observation
    manifest = load_manifest().dropna(subset=["observation_id"]).copy()
    manifest["observation_id"] = manifest["observation_id"].astype(str)
    manifest["individual_id"] = manifest["individual_id"].astype(str)
    observation_annotations = manifest[manifest["observation_id"] == observation_id]

    # Gather all photos in this encounter
    all_photo_paths = observation_photos(observation_id, observation_annotations)
    if not all_photo_paths:
        st.info(f"No photos are available for encounter {observation_id}.")
        return

    # Attach rating details for photos
    details = photo_details(all_photo_paths, observation_annotations)
    labeled_paths = {path for path, _, _ in details}
    # Include any encounter photos that are not yet labeled
    for p in all_photo_paths:
        if p not in labeled_paths:
            details.append((p, set(), 0))

    if not details:
        st.info(f"No photos found for encounter {observation_id}.")
        return

    # Track selected image in session_state & query_params
    selected_key = f"encounter_selected_img_{observation_id}"
    carousel_comp_key = f"encounter_carousel_comp_{observation_id}"

    # Handle carousel selection callback if triggered
    def on_carousel_photo_selected() -> None:
        comp_obj = st.session_state.get(carousel_comp_key)
        val = getattr(comp_obj, "select_photo", None)
        if val is not None:
            try:
                val_int = int(val)
                st.session_state[selected_key] = val_int
                st.query_params["photo"] = str(val_int)
            except (ValueError, TypeError):
                pass

    photo_param = st.query_params.get("photo")
    if photo_param is not None and str(photo_param).isdigit():
        param_idx = int(photo_param)
        if 0 <= param_idx < len(details):
            st.session_state[selected_key] = param_idx

    if selected_key not in st.session_state:
        initial_idx = 0
        if requested_otter:
            otter_photos = set(
                observation_annotations[
                    observation_annotations["individual_id"] == requested_otter
                ]["image_path"].astype(str)
            )
            for idx, (path, _, _) in enumerate(details):
                if path in otter_photos:
                    initial_idx = idx
                    break
        st.session_state[selected_key] = initial_idx

    # Clamp index
    if st.session_state[selected_key] >= len(details) or st.session_state[selected_key] < 0:
        st.session_state[selected_key] = 0
    selected_idx = st.session_state[selected_key]

    current_photo_path, current_parts, current_rating = details[selected_idx]
    current_image_annotations = observation_annotations[
        observation_annotations["image_path"].astype(str) == current_photo_path
    ]

    # -------------------------------------------------------------
    # 1. Image Carousel & Featured Photo
    # -------------------------------------------------------------

    # Annotation boxes toggle
    show_boxes = True

    # Render large featured image
    if show_boxes:
        display_img = draw_annotation_boxes(current_photo_path, current_image_annotations)
        st.image(display_img, width="stretch")
    else:
        st.image(resolve_image_source(current_photo_path), width="stretch")

    # Photo carousel navigation controls
    if len(details) > 1:
        with st.container(horizontal=True, horizontal_alignment="center", gap="small"):
            if st.button("◀ Previous", disabled=(selected_idx == 0), key="carousel_prev", width=160):
                new_idx = selected_idx - 1
                st.session_state[selected_key] = new_idx
                st.query_params["photo"] = str(new_idx)
                st.rerun()
            if st.button("Next ▶", disabled=(selected_idx >= len(details) - 1), key="carousel_next", width=160):
                new_idx = selected_idx + 1
                st.session_state[selected_key] = new_idx
                st.query_params["photo"] = str(new_idx)
                st.rerun()

    # Compact thumbnail carousel (click thumbnail to select, height <= 200px)
    if len(details) > 1:
        thumb_items: list[dict[str, str]] = []
        for i, (t_path, _, _) in enumerate(details):
            img_src = image_url_for(
                resolve_image_source(t_path),
                f"encounter-thumb-{observation_id}-{i}",
            )
            thumb_items.append({"src": img_src})

        if thumb_items:
            result = encounter_thumbnail_carousel(
                key=carousel_comp_key,
                data={
                    "items": thumb_items,
                    "selectedIndex": selected_idx,
                },
                on_select_photo_change=on_carousel_photo_selected,
            )
            if result and getattr(result, "select_photo", None) is not None:
                try:
                    new_val = int(result.select_photo)
                    if new_val != selected_idx:
                        st.session_state[selected_key] = new_val
                        st.query_params["photo"] = str(new_val)
                        st.rerun()
                except (ValueError, TypeError):
                    pass

    # -------------------------------------------------------------
    # 2. Section with Tabs (Identity, Metadata, Annotations)
    # -------------------------------------------------------------
    tab_identity, tab_metadata, tab_annotations = st.tabs(
        ["Identity", "Metadata", "Annotations"]
    )

    # -------------------------------------------------------------
    # Tab 1: Identity
    # -------------------------------------------------------------
    with tab_identity:
        rating_stars = stars(current_rating)
        st.markdown(
            f'<div style="margin-bottom: 0.85rem; font-size: 1rem;">'
            f'<b>Photo rating:</b> '
            f'<span style="color: #e5a100; font-size: 1.15rem;" title="{current_rating} of 4 stars">{rating_stars}</span> '
            f'<span style="opacity: 0.8; font-size: 0.9rem;">({current_rating} of 4 stars)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Otters in this photo")
        otters_in_image = sorted(
            current_image_annotations["individual_id"].dropna().unique()
        )
        if otters_in_image:
            for oid in otters_in_image:
                parts = current_image_annotations[
                    current_image_annotations["individual_id"] == oid
                ]["body_part"].tolist()
                parts_str = ", ".join(p.replace("_", " ").title() for p in parts)
                view_link = f'<a href="view?otter={urllib.parse.quote(oid)}" style="font-weight: 600; font-size: 1.05rem;">{oid}</a>'
                st.markdown(
                    f"&bull; {view_link} &mdash; <span style='opacity: 0.8;'>labeled parts: <b>{parts_str}</b></span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No labeled otters in this photo.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Other otters in this encounter")
        all_encounter_otters = sorted(
            observation_annotations["individual_id"].dropna().unique()
        )
        other_otters = [oid for oid in all_encounter_otters if oid not in otters_in_image]
        if other_otters:
            for oid in other_otters:
                photos_with_otter = [
                    f"#{i + 1}"
                    for i, (p, _, _) in enumerate(details)
                    if oid in observation_annotations[
                        observation_annotations["image_path"].astype(str) == p
                    ]["individual_id"].values
                ]
                photos_ref = f" (in photo {', '.join(photos_with_otter)})" if photos_with_otter else ""
                view_link = f'<a href="view?otter={urllib.parse.quote(oid)}" style="font-weight: 600; font-size: 1.05rem;">{oid}</a>'
                st.markdown(
                    f"&bull; {view_link}<span style='opacity: 0.8;'>{photos_ref}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No other otters were identified in this encounter.")

    # -------------------------------------------------------------
    # Tab 2: Metadata (for selected photo)
    # -------------------------------------------------------------
    with tab_metadata:
        results_df = load_encounter_results(observation_id, token)

        # Match metadata row for this photo
        meta_row = None
        if not results_df.empty:
            matched = results_df[results_df["image_path"] == current_photo_path]
            if not matched.empty:
                meta_row = matched.iloc[0]
            else:
                stem = Path(current_photo_path).stem
                matched_stem = results_df[results_df["photo_id"] == stem]
                if not matched_stem.empty:
                    meta_row = matched_stem.iloc[0]
                else:
                    meta_row = results_df.iloc[0]

        meta_col1, meta_col2 = st.columns([1, 1])
        with meta_col1:
            st.markdown("#### Photo & encounter details")
            st.markdown(f"**Encounter:** `{observation_id}`")

            observed_at = meta_row.get("observed_at") if meta_row is not None else None
            if observed_at and pd.notna(observed_at):
                st.markdown(f"**Date / Time:** {observed_at}")

            observer = meta_row.get("observer") if meta_row is not None else None
            if observer and pd.notna(observer) and str(observer).strip() not in ("", "Unknown"):
                st.markdown(f"**Author / Observer:** {observer}")

            source = meta_row.get("source") if meta_row is not None else None
            if source and pd.notna(source):
                source_display = {
                    "inat": "iNaturalist",
                    "flickr": "Flickr",
                    "wikimedia": "Wikimedia Commons",
                    "upload": "User Upload",
                }.get(str(source).lower(), str(source).title())
                st.markdown(f"**Data Source:** {source_display}")

            locality = meta_row.get("locality") if meta_row is not None else None
            if locality and pd.notna(locality):
                st.markdown(f"**Locality:** {locality}")

            # Check for coordinates: check annotations first (for user-set map locations or photo-level coords),
            # then fall back to meta_row from results.csv.
            located = pd.DataFrame()
            if not current_image_annotations.empty:
                located = current_image_annotations.loc[
                    valid_location_mask(current_image_annotations)
                ]
            if located.empty and not observation_annotations.empty:
                located = observation_annotations.loc[
                    valid_location_mask(observation_annotations)
                ]
            if located.empty and meta_row is not None:
                meta_frame = pd.DataFrame([meta_row])
                if not meta_frame.empty:
                    located = meta_frame.loc[valid_location_mask(meta_frame)]

            has_coords = False
            lat_f = lon_f = 0.0
            loc_src = ""
            if not located.empty:
                loc_row = located.iloc[0]
                try:
                    lat_f = float(loc_row["latitude"])
                    lon_f = float(loc_row["longitude"])
                    loc_src = str(loc_row.get("location_source") or "")
                    has_coords = True
                except (ValueError, TypeError):
                    pass

            if has_coords:
                coord_text = f"{lat_f:.5f}, {lon_f:.5f}"
                if loc_src == "inferred_temporal_centroid":
                    coord_text += " *(inferred)*"
                elif loc_src == "map":
                    coord_text += " *(user-set)*"
                st.markdown(f"**Coordinates:** {coord_text}")

            # Original source links
            page_url = meta_row.get("page_url") if meta_row is not None else None
            image_url = meta_row.get("image_url") if meta_row is not None else None
            if (page_url and pd.notna(page_url)) or (image_url and pd.notna(image_url)):
                st.markdown("<br>**Original source links:**", unsafe_allow_html=True)
                if page_url and pd.notna(page_url):
                    st.markdown(
                        f'- <a href="{html.escape(page_url)}" target="_blank" rel="noopener">Open observation webpage ↗</a>',
                        unsafe_allow_html=True,
                    )
                if image_url and pd.notna(image_url):
                    st.markdown(
                        f'- <a href="{html.escape(image_url)}" target="_blank" rel="noopener">Open full original image ↗</a>',
                        unsafe_allow_html=True,
                    )

        with meta_col2:
            st.markdown("#### Location map")
            if has_coords:
                try:
                    loc_text = str(locality) if locality and pd.notna(locality) else "Location"
                    render_single_pin_map(
                        lat_f,
                        lon_f,
                        title=f"Encounter {observation_id}",
                        detail=loc_text,
                    )
                except Exception:
                    st.caption("Unable to parse coordinates for mapping.")
            else:
                st.info("No geographic coordinates are available for this encounter.")

    # -------------------------------------------------------------
    # Tab 3: Annotations
    # -------------------------------------------------------------
    with tab_annotations:
        st.markdown("#### Labels for this photo")

        label_url = label_url_for_encounter(observation_id, current_photo_path)
        st.markdown(
            f'<div style="padding-top: 4px;">'
            f'<a href="{html.escape(label_url)}">Edit labels ↗</a>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not current_image_annotations.empty:
            rows_html = []
            for row in current_image_annotations.itertuples():
                oid = str(row.individual_id)
                part = str(row.body_part)
                color = BODY_PART_COLORS.get(part, "#FFEB3B")
                part_display = part.replace("_", " ").title()
                bbox_str = (
                    f"x: {round(float(row.bbox_x))}, "
                    f"y: {round(float(row.bbox_y))}, "
                    f"w: {round(float(row.bbox_w))}, "
                    f"h: {round(float(row.bbox_h))}"
                )
                otter_link = f"<a href='view?otter={urllib.parse.quote(oid)}' style='font-weight: 600;'>{oid}</a>"
                badge = f"<span style='background-color: {color}; color: #000; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem;'>{part_display}</span>"
                rows_html.append(
                    f"<tr style='border-bottom: 1px solid rgba(128,128,128,0.2);'>"
                    f"<td style='padding: 8px 12px;'>{otter_link}</td>"
                    f"<td style='padding: 8px 12px;'>{badge}</td>"
                    f"<td style='padding: 8px 12px; font-family: monospace; font-size: 0.85rem;'>{bbox_str}</td>"
                    f"</tr>"
                )
            table_html = (
                "<table style='width: 100%; border-collapse: collapse; margin-top: 8px;'>"
                "<thead><tr style='border-bottom: 2px solid rgba(128,128,128,0.4); text-align: left;'>"
                "<th style='padding: 8px 12px;'>Otter ID</th>"
                "<th style='padding: 8px 12px;'>Body Part</th>"
                "<th style='padding: 8px 12px;'>Bounding Box</th>"
                "</tr></thead>"
                f"<tbody>{''.join(rows_html)}</tbody>"
                "</table>"
            )
            st.html(table_html)
        else:
            st.info("No annotations have been labeled on this photo yet.")
            label_url = label_url_for_encounter(observation_id, current_photo_path)
            st.markdown(
                f'<a href="{html.escape(label_url)}">Open in labeling tool to add labels ↗</a>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
