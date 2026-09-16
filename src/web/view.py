#!/usr/bin/env python3
"""Streamlit page for browsing labeled otter encounters.

Run with:
    uv run streamlit run src/web/app.py
"""

from __future__ import annotations

import html
import json
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

LABEL_TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LABEL_TOOL_DIR))
sys.path.insert(0, str(LABEL_TOOL_DIR.parent))
from common.image_source import (  # noqa: E402
    image_available,
    open_image,
    resolve_image_source,
)
from common.manifest import REPO_ROOT, load_manifest  # noqa: E402
from common.manifest import REPO_ROOT, is_valid_location, load_manifest  # noqa: E402
from location.geodistance import infer_encounter_locations  # noqa: E402
from common.web import encounter_url_for_encounter  # noqa: E402

RESULTS_PATH = REPO_ROOT / "data/results.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
BODY_PARTS = ("whole_body", "throat", "throat_portrait")
BODY_PART_COLORS = {
    "whole_body": "#00E5FF",
    "throat": "#FF4081",
    "throat_portrait": "#76FF03",
}
MIN_INITIAL_MAP_AREA_SQUARE_MILES = 10
COUNTRY_NAMES = {
    "AR": "Argentina",
    "BO": "Bolivia",
    "BR": "Brazil",
    "CO": "Colombia",
    "EC": "Ecuador",
    "GF": "French Guiana",
    "GY": "Guyana",
    "PE": "Peru",
    "PY": "Paraguay",
    "SR": "Suriname",
    "VE": "Venezuela",
}


def file_token(path: Path) -> tuple[int, int]:
    """Return stable cache-invalidation metadata for a file."""
    if not path.is_file():
        return 0, 0
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False)
def load_source_metadata(
    *tokens: object,
    **kwargs: object,
) -> pd.DataFrame:
    """Load and normalize encounter metadata from results.csv."""
    del tokens, kwargs
    if not RESULTS_PATH.is_file():
        return pd.DataFrame(
            columns=[
                "observation_id",
                "observed_at",
                "observer",
                "latitude",
                "longitude",
                "source",
                "locality",
                "country",
                "region",
                "individual_id",
            ]
        )

    res = pd.read_csv(RESULTS_PATH, dtype=str)
    res = res.rename(columns={"encounter_id": "observation_id"})
    metadata = res.groupby("observation_id", as_index=False).agg(
        {
            "observed_at": "first",
            "observer": "first",
            "latitude": "first",
            "longitude": "first",
            "source": "first",
            "locality": "first",
        }
    )
    metadata["country"] = pd.NA
    metadata["region"] = pd.NA
    manifest = load_manifest()
    obs_to_otter = (
        manifest.dropna(subset=["observation_id", "individual_id"])
        .groupby("observation_id")["individual_id"]
        .first()
        .to_dict()
    )
    metadata["individual_id"] = metadata["observation_id"].map(obs_to_otter)
    return infer_encounter_locations(metadata)


@st.cache_data(show_spinner=False)
def cached_encounter_photos(token: tuple[int, int]) -> dict[str, list[str]]:
    """Index encounter_id -> list of image paths from results.csv."""
    del token
    if not RESULTS_PATH.is_file():
        return {}
    res = pd.read_csv(RESULTS_PATH, dtype=str, usecols=["encounter_id", "image_path"])
    res = res.dropna(subset=["encounter_id", "image_path"])
    photos_by_enc: dict[str, list[str]] = {}
    for enc_id, group in res.groupby("encounter_id"):
        photos_by_enc[str(enc_id)] = [
            str(p).strip() for p in group["image_path"] if str(p).strip()
        ]
    return photos_by_enc


def first_present(values: pd.Series, fallback: str = "Unknown") -> str:
    """Return the first non-empty value in a metadata series."""
    present = values.dropna().astype(str).str.strip()
    present = present[present != ""]
    return present.iloc[0] if len(present) else fallback


def valid_location_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify real coordinates, excluding missing, invalid, and 0,0 values."""
    latitude = pd.to_numeric(frame["latitude"], errors="coerce")
    longitude = pd.to_numeric(frame["longitude"], errors="coerce")
    return (
        latitude.notna()
        & longitude.notna()
        & latitude.between(-90, 90)
        & longitude.between(-180, 180)
        & ~((latitude == 0) & (longitude == 0))
    )


def encounter_rows(manifest: pd.DataFrame, individual_id: str) -> pd.DataFrame:
    """Build one summary row per encounter for an individual."""
    annotations = manifest[
        manifest["individual_id"].astype(str) == individual_id
    ].copy()
    metadata = load_source_metadata(file_token(RESULTS_PATH))
    rows = []
    for observation_id, group in annotations.groupby("observation_id", sort=False):
        observation_id = str(observation_id)
        source_rows = metadata[metadata["observation_id"] == observation_id]
        observed_at = pd.to_datetime(
            first_present(source_rows.get("observed_at", pd.Series(dtype=str)), ""),
            errors="coerce",
            utc=True,
        )
        observer = first_present(source_rows.get("observer", pd.Series(dtype=str)))
        source = first_present(
            source_rows.get("source", pd.Series(dtype=str)),
            first_present(group["location_source"], "Unknown"),
        )
        country_code = first_present(source_rows.get("country", pd.Series(dtype=str)))
        country = COUNTRY_NAMES.get(country_code.upper(), country_code)
        region = first_present(
            source_rows.get("region", pd.Series(dtype=str)),
            first_present(source_rows.get("locality", pd.Series(dtype=str))),
        )
        located = group.loc[valid_location_mask(group)]
        if located.empty and not source_rows.empty:
            located = source_rows.loc[valid_location_mask(source_rows)]
        latitude = pd.to_numeric(located["latitude"], errors="coerce")
        longitude = pd.to_numeric(located["longitude"], errors="coerce")
        loc_src = first_present(
            group["location_source"],
            first_present(source_rows.get("location_source", pd.Series(dtype=str)), ""),
        )
        rows.append(
            {
                "observation_id": observation_id,
                "observed_at": observed_at,
                "observer": observer,
                "source": source.upper() if source != "Unknown" else source,
                "location_source": str(loc_src),
                "country": country,
                "region": region,
                "latitude": latitude.iloc[0] if len(latitude) else pd.NA,
                "longitude": longitude.iloc[0] if len(longitude) else pd.NA,
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        "observed_at", ascending=False, na_position="last"
    ).reset_index(drop=True)


def observation_photos(observation_id: str, annotations: pd.DataFrame) -> list[str]:
    """Return every locally available source photo for an encounter."""
    photos_by_enc = cached_encounter_photos(file_token(RESULTS_PATH))
    photos: set[str] = set(photos_by_enc.get(observation_id, []))
    photos.update(annotations["image_path"].dropna().astype(str))
    return sorted(path for path in photos if image_available(path))


def photo_rating(parts: set[str], location_available: bool) -> int:
    """Return the cumulative detection quality rating for a photo."""
    if not parts:
        return 0
    if not set(BODY_PARTS).issubset(parts):
        return 2 if {"whole_body", "throat"}.issubset(parts) else 1
    return 4 if location_available else 3


def photo_details(
    photos: list[str], annotations: pd.DataFrame
) -> list[tuple[str, set[str], int]]:
    """Attach selected-otter labels and ratings, omitting unlabeled photos."""
    located_paths = set(
        annotations.loc[valid_location_mask(annotations), "image_path"].astype(str)
    )
    parts_by_path = (
        annotations.groupby("image_path")["body_part"]
        .agg(lambda values: set(values.dropna().astype(str)))
        .to_dict()
    )
    details = [
        (
            path,
            parts_by_path[path],
            photo_rating(parts_by_path[path], path in located_paths),
        )
        for path in photos
        if path in parts_by_path and parts_by_path[path]
    ]
    return sorted(
        details,
        key=lambda item: (
            "throat_portrait" not in item[1],
            -item[2],
            Path(item[0]).name,
        ),
    )


def stars(rating: int) -> str:
    return "★" * rating + "☆" * (4 - rating)


def render_encounter_map(
    encounters: pd.DataFrame,
    title_column: str | None = None,
    detail_column: str | None = None,
    otter_ids_column: str | None = None,
) -> None:
    """Render locations with Leaflet, without requiring WebGL."""
    mapped = encounters.loc[valid_location_mask(encounters)].copy()
    if mapped.empty:
        st.info("None of these encounters has location data.")
        return
    mapped["latitude"] = pd.to_numeric(mapped["latitude"], errors="coerce")
    mapped["longitude"] = pd.to_numeric(mapped["longitude"], errors="coerce")
    mapped = mapped.dropna(subset=["latitude", "longitude"])
    if title_column is None:
        mapped["map_title"] = mapped["observed_at"].map(
            lambda value: (
                value.strftime("%B %-d, %Y") if pd.notna(value) else "Date unknown"
            )
        )
        title_column = "map_title"
    if detail_column is None:
        mapped["map_detail"] = mapped["observer"].astype(str)
        detail_column = "map_detail"

    points = []
    for row in mapped.itertuples(index=False):
        otter_ids = (
            [html.escape(str(value)) for value in getattr(row, otter_ids_column)]
            if otter_ids_column is not None
            else []
        )
        points.append(
            {
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "title": html.escape(str(getattr(row, title_column))),
                "detail": html.escape(str(getattr(row, detail_column))),
                "encounterId": html.escape(str(row.observation_id)),
                "otter_ids": otter_ids,
            }
        )
    points_json = json.dumps(points).replace("</", "<\\/")
    leaflet_html = rf"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
          <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
          <style>
            html, body, #map {{ height: 100%; margin: 0; }}
            #map {{ position: relative; overflow: hidden; border-radius: 0.5rem;
                    background: #e7ecef; outline-offset: 1px; }}
            .leaflet-pane, .leaflet-tile, .leaflet-marker-icon,
            .leaflet-marker-shadow, .leaflet-tile-container,
            .leaflet-pane > svg, .leaflet-pane > canvas,
            .leaflet-zoom-box, .leaflet-image-layer, .leaflet-layer {{
              position: absolute;
              left: 0;
              top: 0;
            }}
            .leaflet-container {{ overflow: hidden; -webkit-tap-highlight-color: transparent; }}
            .leaflet-tile, .leaflet-marker-icon, .leaflet-marker-shadow {{
              user-select: none;
              -webkit-user-drag: none;
            }}
            .leaflet-container .leaflet-overlay-pane svg,
            .leaflet-container .leaflet-marker-pane img,
            .leaflet-container .leaflet-shadow-pane img,
            .leaflet-container .leaflet-tile-pane img,
            .leaflet-container img.leaflet-image-layer,
            .leaflet-container .leaflet-tile {{ max-width: none !important; max-height: none !important; }}
            .leaflet-tile {{ visibility: hidden; }}
            .leaflet-tile-loaded {{ visibility: inherit; }}
            .leaflet-zoom-animated {{ transform-origin: 0 0; }}
            .leaflet-pane {{ z-index: 400; }}
            .leaflet-tile-pane {{ z-index: 200; }}
            .leaflet-overlay-pane {{ z-index: 400; }}
            .leaflet-shadow-pane {{ z-index: 500; }}
            .leaflet-marker-pane {{ z-index: 600; }}
            .leaflet-tooltip-pane {{ z-index: 650; }}
            .leaflet-popup-pane {{ z-index: 700; }}
            .leaflet-control {{ position: relative; z-index: 800; pointer-events: auto; }}
            .leaflet-top, .leaflet-bottom {{ position: absolute; z-index: 1000; pointer-events: none; }}
            .leaflet-top {{ top: 0; }} .leaflet-right {{ right: 0; }}
            .leaflet-bottom {{ bottom: 0; }} .leaflet-left {{ left: 0; }}
            .leaflet-interactive {{ cursor: pointer; }}
            a {{ color: #4b8b8b; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
          </style>
        </head>
        <body>
          <div id="map"></div>
          <script>
            const points = {points_json};
            const applyJitter = (pts) => {{
              const baseRadius = 0.00035;
              const keyMap = new Map();
              pts.forEach((pt) => {{
                const key = `${{pt.latitude.toFixed(5)}},${{pt.longitude.toFixed(5)}}`;
                if (!keyMap.has(key)) keyMap.set(key, []);
                keyMap.get(key).push(pt);
              }});
              const result = [];
              keyMap.forEach((group) => {{
                if (group.length === 1) {{
                  result.push(group[0]);
                }} else {{
                  group.forEach((pt, idx) => {{
                    const ring = Math.floor(idx / 6);
                    const posInRing = idx % 6;
                    const ptsInThisRing = Math.min(6, group.length - ring * 6);
                    const radius = baseRadius * (1 + ring * 0.7);
                    const angle = (2 * Math.PI * posInRing) / ptsInThisRing + (ring * Math.PI / 6);
                    const cosLat = Math.cos((pt.latitude * Math.PI) / 180);
                    const scaleLon = Math.abs(cosLat) > 0.0001 ? 1 / cosLat : 1;
                    const jitterLat = pt.latitude + radius * Math.cos(angle);
                    const jitterLon = pt.longitude + radius * Math.sin(angle) * scaleLon;
                    result.push({{
                      ...pt,
                      latitude: jitterLat,
                      longitude: jitterLon
                    }});
                  }});
                }}
              }});
              return result;
            }};
            const jitteredPoints = applyJitter(points);
            const map = L.map('map', {{
              preferCanvas: false,
              zoomAnimation: false,
              fadeAnimation: false,
              markerZoomAnimation: false
            }});
            L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
              maxZoom: 19,
              attribution: '&copy; OpenStreetMap contributors'
            }}).addTo(map);
            const otterUrl = (individualId) => {{
              return `view?otter=${{encodeURIComponent(individualId)}}`;
            }};
            const markers = jitteredPoints.map((point) => {{
              const marker = L.circleMarker(
                [point.latitude, point.longitude],
                {{ radius: 8, color: '#ffffff', weight: 2,
                   fillColor: '#198287', fillOpacity: 0.9 }}
              ).addTo(map);
              const detailLine = point.detail ? `<br>${{point.detail}}` : '';
              marker.bindTooltip(
                `<b>${{point.title}}</b><br>Encounter: <a href="encounter?observation=${{encodeURIComponent(point.encounterId)}}">${{point.encounterId}}</a>${{detailLine}}`
              );
              if (point.otter_ids.length === 1) {{
                marker.on('click', () => {{
                  window.open(otterUrl(point.otter_ids[0]), '_blank', 'noopener');
                }});
              }} else if (point.otter_ids.length > 1) {{
                const links = point.otter_ids.map((individualId) =>
                  `<a href="${{otterUrl(individualId)}}" target="_blank" ` +
                  `rel="noopener">` +
                  `${{individualId}}</a>`
                ).join('<br>');
                marker.bindPopup(`<b>View otter</b><br>${{links}}`);
              }}
              return marker;
            }});
            const encounterBounds = L.featureGroup(markers).getBounds();
            const center = encounterBounds.getCenter();
            const minimumSideMiles = Math.sqrt({MIN_INITIAL_MAP_AREA_SQUARE_MILES});
            const latitudeHalfSpan = minimumSideMiles / (2 * 69.0);
            const longitudeMilesPerDegree = Math.max(
              1.0,
              69.0 * Math.cos(center.lat * Math.PI / 180)
            );
            const longitudeHalfSpan = minimumSideMiles / (2 * longitudeMilesPerDegree);
            encounterBounds.extend([
              center.lat - latitudeHalfSpan,
              center.lng - longitudeHalfSpan
            ]);
            encounterBounds.extend([
              center.lat + latitudeHalfSpan,
              center.lng + longitudeHalfSpan
            ]);
            map.fitBounds(encounterBounds, {{ padding: [24, 24] }});
            let resizeTimer;
            const resizeMap = () => {{
              clearTimeout(resizeTimer);
              resizeTimer = setTimeout(() => map.invalidateSize({{ pan: false }}), 50);
            }};
            new ResizeObserver(resizeMap).observe(document.getElementById('map'));
            window.addEventListener('load', resizeMap);
            setTimeout(resizeMap, 250);
          </script>
        </body>
        </html>
    """
    st.iframe(leaflet_html, width="stretch", height=250)
    missing = len(encounters) - len(mapped)
    if missing:
        st.caption(f"{missing} encounter(s) without coordinates are not mapped.")


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: tuple[float, float],
    p2: tuple[float, float],
    fill: str,
    width: int,
    dash_len: float = 14,
    gap_len: float = 10,
) -> None:
    """Draw a dashed line segment between p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = (dx**2 + dy**2) ** 0.5
    if dist == 0:
        return
    ux = dx / dist
    uy = dy / dist

    curr = 0.0
    drawing = True
    while curr < dist:
        seg = dash_len if drawing else gap_len
        next_curr = min(curr + seg, dist)
        if drawing:
            sx = x1 + ux * curr
            sy = y1 + uy * curr
            ex = x1 + ux * next_curr
            ey = y1 + uy * next_curr
            draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
        curr = next_curr
        drawing = not drawing


def draw_annotation_boxes(
    image_path: str, boxes: pd.DataFrame, dashed: bool = False
) -> Image.Image:
    """Return the source photo with each annotation's bounding box outlined."""
    image = open_image(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    stroke_width = max(2, round(min(image.size) / 300))
    scale = max(1.0, stroke_width / 2.0)
    dash_len = 7.0 * scale
    gap_len = 5.0 * scale
    for box in boxes.itertuples(index=False):
        x1, y1 = float(box.bbox_x), float(box.bbox_y)
        x2, y2 = x1 + float(box.bbox_w), y1 + float(box.bbox_h)
        color = BODY_PART_COLORS.get(box.body_part, "#FFEB3B")
        if dashed:
            draw_dashed_line(
                draw,
                (x1, y1),
                (x2, y1),
                fill=color,
                width=stroke_width,
                dash_len=dash_len,
                gap_len=gap_len,
            )
            draw_dashed_line(
                draw,
                (x2, y1),
                (x2, y2),
                fill=color,
                width=stroke_width,
                dash_len=dash_len,
                gap_len=gap_len,
            )
            draw_dashed_line(
                draw,
                (x2, y2),
                (x1, y2),
                fill=color,
                width=stroke_width,
                dash_len=dash_len,
                gap_len=gap_len,
            )
            draw_dashed_line(
                draw,
                (x1, y2),
                (x1, y1),
                fill=color,
                width=stroke_width,
                dash_len=dash_len,
                gap_len=gap_len,
            )
        else:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke_width)
    return image


def render_photo_grid(
    details: list[tuple[str, set[str], int]], annotations: pd.DataFrame | None = None
) -> None:
    """Render all encounter photos with their cumulative quality rating.

    When ``annotations`` is given, each photo is drawn with its labeled
    bounding boxes overlaid, colored by body part.
    """
    if not details:
        st.info("No photos are available for this observation.")
        return
    columns = st.columns(4)
    for position, (path, parts, rating) in enumerate(details):
        column = columns[position % len(columns)]
        if annotations is not None:
            boxes = annotations[annotations["image_path"].astype(str) == path]
            column.image(draw_annotation_boxes(path, boxes), width="stretch")
        else:
            column.image(resolve_image_source(path), width="stretch")
        labels = ", ".join(
            part.replace("_", " ") for part in BODY_PARTS if part in parts
        )
        column.markdown(
            f'<div class="photo-rating" title="{rating} of 4 stars">'
            f"{stars(rating)}</div>",
            unsafe_allow_html=True,
        )
        column.caption(labels or "No detection for this otter")


def store_selected_otter_in_url() -> None:
    """Persist the completed selector change without racing the widget rerun."""
    st.query_params["otter"] = st.session_state.selected_otter


def image_url_for(image_source: str, image_id: str) -> str:
    """Return a URL suitable for an HTML img tag."""
    try:
        from streamlit.elements.lib.image_utils import image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig

        return image_to_url(
            image_source,
            LayoutConfig(),
            False,
            "RGB",
            "auto",
            image_id,
        )
    except Exception:
        return ""


def render_timeline(
    encounters: pd.DataFrame, annotations: pd.DataFrame, individual_id: str
) -> None:
    """Render a recent-first observation timeline with favored thumbnails."""
    st.subheader("Observation timeline")

    st.html("""
    <style>
    .timeline {
        position: relative;
        list-style-type: none;
        padding-left: 0;
        margin: 0rem 0;
    }

    /* The vertical line */
    .timeline::after {
        content: '';
        position: absolute;
        width: 3px;
        background-color: #4b8b8b; /* Timeline line color */
        top: 14px;
        bottom: 14px;
        left: 10px;
    }

    /* Timeline list item container */
    .timeline li {
        position: relative;
        background-color: transparent;
        width: 100%;
        box-sizing: border-box;
        padding-left: 25px; /* Space for the line and circles */
        padding-top: 0px;
        margin-bottom: 28px;
        left: -10px;
    }

    /* The circles on the timeline */
    .timeline li::after {
        content: "⦿"; /* Custom character or emoji */
        position: absolute;
        width: 18px;
        height: 18px;
        left: 0px;
        top: 4px;
        color: #4b8b8b;
        font-size: 1.15rem;
        line-height: 1;
        z-index: 1;
    }

    .timeline-date {
        font-weight: bold;
        font-size: 1.05rem;
    }

    .timeline-encounter {
        display: flex;
        flex-direction: row;
        gap: 1.5rem;
        align-items: flex-start;
    }

    .timeline-info {
        flex: 0 0 220px;
        max-width: 240px;
    }

    .timeline-obs-link {
        margin-top: 4px;
        font-size: 0.95rem;
    }

    .timeline-obs-link a {
        color: #4b8b8b;
        text-decoration: none;
        font-weight: 600;
    }

    .timeline-obs-link a:hover {
        text-decoration: underline;
    }

    .timeline-caption {
        font-size: 0.85rem;
        opacity: 0.75;
        margin-top: 4px;
        line-height: 1.35;
        word-break: break-word;
    }

    .timeline-photos-link {
        margin-top: 8px;
    }

    .timeline-photos-link a {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.85rem;
        color: #4b8b8b;
        text-decoration: none;
        font-weight: 500;
    }

    .timeline-photos-link a:hover {
        text-decoration: underline;
    }

    .timeline-thumbnails {
        flex: 1;
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
    }

    .timeline-thumb-card {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .timeline-thumb-img {
        width: 100%;
        aspect-ratio: 4 / 3;
        object-fit: cover;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
        display: block;
    }

    @media (max-width: 768px) {
        .timeline-encounter {
            flex-direction: column;
            gap: 0.75rem;
        }
        .timeline-info {
            flex: 1 1 100%;
            max-width: 100%;
        }
        .timeline-thumbnails {
            grid-template-columns: repeat(2, 1fr);
            width: 100%;
        }
    }
    </style>
    """)

    if encounters.empty:
        st.info("No encounters available.")
        return

    timeline_items: list[str] = []
    for encounter in encounters.itertuples(index=False):
        observation_id = str(encounter.observation_id)
        observation_annotations = annotations[
            annotations["observation_id"].astype(str) == observation_id
        ]
        details = photo_details(
            observation_photos(observation_id, observation_annotations),
            observation_annotations,
        )
        date_label = (
            encounter.observed_at.strftime("%B %-d, %Y")
            if pd.notna(encounter.observed_at)
            else "Date unknown"
        )
        enc_url = encounter_url_for_encounter(observation_id)
        enc_link = f'<a href="{html.escape(enc_url)}">{html.escape(observation_id)}</a>'

        observer_html = ""
        if encounter.observer and encounter.observer.strip() not in ("", "Unknown"):
            observer_html = f'<div class="timeline-caption">Author: {html.escape(encounter.observer)}</div>'

        if is_valid_location(encounter.latitude, encounter.longitude):
            coordinate_label = f"{encounter.latitude:.5f}, {encounter.longitude:.5f}"
            if getattr(encounter, "location_source", "") == "inferred_temporal_centroid":
                coordinate_label += " (inferred)"
        else:
            coordinate_label = "Coordinates unavailable"
        place_parts = [
            str(value)
            for value in (encounter.country, encounter.region)
            if pd.notna(value) and str(value).strip() not in ("", "Unknown")
        ]
        place_label = f" ({' / '.join(place_parts)})" if place_parts else ""
        location_text = f"""Location: {coordinate_label}  
        {place_label}"""
        location_html = f'<div class="timeline-caption">{html.escape(location_text)}</div>'

        thumb_cards: list[str] = []
        for position, (path, _, rating) in enumerate(details[:4]):
            img_src = image_url_for(
                resolve_image_source(path),
                f"timeline-{observation_id}-{position}",
            )
            star_label = stars(rating)
            if img_src:
                card = (
                    f'<div class="timeline-thumb-card">'
                    f'<img src="{html.escape(img_src)}" class="timeline-thumb-img" alt="Encounter photo" loading="lazy" />'
                    f'<div title="{rating} of 4 stars">{html.escape(star_label)}</div>'
                    f'</div>'
                )
                thumb_cards.append(card)

        thumbnails_html = "".join(thumb_cards)

        item_html = (
            f'<li>'
            f'<div class="timeline-content">'
            f'<div class="timeline-encounter">'
            f'<div class="timeline-info">'
            f'<div class="timeline-date">{html.escape(date_label)}</div>'
            f'<div class="timeline-obs-link">{enc_link}</div>'
            f'{observer_html}'
            f'{location_html}'
            f'</div>'
            f'<div class="timeline-thumbnails">'
            f'{thumbnails_html}'
            f'</div>'
            f'</div>'
            f'</div>'
            f'</li>'
        )
        timeline_items.append(item_html)

    all_items_html = "".join(timeline_items)
    st.html(f'<ul class="timeline">{all_items_html}</ul>')


def main() -> None:
    manifest = load_manifest().dropna(subset=["individual_id", "observation_id"])
    manifest = manifest.copy()
    manifest["individual_id"] = manifest["individual_id"].astype(str)
    manifest["observation_id"] = manifest["observation_id"].astype(str)
    encounter_counts = manifest.groupby("individual_id")["observation_id"].nunique()
    encounter_counts = encounter_counts.sort_index()
    multiple_encounter_ids = set(encounter_counts[encounter_counts > 1].index)
    requested_individual = str(st.query_params.get("otter", ""))



    if encounter_counts.empty:
        st.info("No labeled otters are available yet.")
        return

    if "view_show_all_otters" not in st.session_state:
        st.session_state.view_show_all_otters = False
    if (
        requested_individual in encounter_counts.index
        and (
            requested_individual != st.session_state.get("view_applied_url_otter", "")
            # Navigating to another page clears this widget's state, so the
            # URL otter must be reapplied even if it matches the last-applied value.
            or "selected_otter" not in st.session_state
        )
    ):
        st.session_state.selected_otter = requested_individual
        # A direct link to a singleton should remain viewable and selectable.
        if requested_individual not in multiple_encounter_ids:
            st.session_state.view_show_all_otters = True
    st.session_state.view_applied_url_otter = requested_individual

    show_all = st.session_state.view_show_all_otters
    eligible = (
        encounter_counts
        if show_all
        else encounter_counts[encounter_counts.index.isin(multiple_encounter_ids)]
    )
    individual_ids = eligible.index.astype(str).tolist()

    st.title("Otter: " + str(requested_individual if requested_individual else individual_ids[0]))
    if not individual_ids:
        st.info(
            "No labeled otters are present in multiple encounters yet. "
            "Turn on “Show all otters” to view single-encounter otters."
        )
        return

    if st.session_state.get("selected_otter") not in individual_ids:
        st.session_state.selected_otter = individual_ids[0]
        st.query_params["otter"] = individual_ids[0]
    with st.container(horizontal=True, vertical_alignment="center"):
        st.write("Otter")
        individual_id = st.selectbox(
            "Otter",
            individual_ids,
            key="selected_otter",
            format_func=lambda value: (
                f"{value} ({encounter_counts[value]} "
                f"{'encounter' if encounter_counts[value] == 1 else 'encounters'})"
            ),
            on_change=store_selected_otter_in_url,
            label_visibility="collapsed",
            width=200,
        )
        st.toggle(
            "Show all",
            key="view_show_all_otters",
            help=(
                "Off: only otters labeled in multiple encounters. "
                "On: every labeled otter."
            ),
        )
    annotations = manifest[manifest["individual_id"].astype(str) == individual_id]
    encounters = encounter_rows(manifest, individual_id)

    st.caption(f"{len(encounters)} encounters for {individual_id}")
    render_encounter_map(encounters)

    render_timeline(encounters, annotations, individual_id)


if __name__ == "__main__":
    main()
