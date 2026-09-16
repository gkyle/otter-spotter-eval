#!/usr/bin/env python3
"""Map page for viewing encounter locations across all labeled otters."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

WEB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB_DIR))
sys.path.insert(0, str(WEB_DIR.parent))
from common.manifest import load_manifest  # noqa: E402
from map_widget import otter_map  # noqa: E402
from location.geodistance import infer_encounter_locations  # noqa: E402
from view import (  # noqa: E402
    RESULTS_PATH,
    file_token,
    load_source_metadata,
    valid_location_mask,
)


GEOGRAPHY_OPTIONS = ("World", "South America", "Rupununi Region")
GEOGRAPHY_BOUNDS = {
    "World": None,
    "South America": (-56.0, -82.0, 13.0, -34.0),
    "Rupununi Region": (1.0, -61.5, 5.5, -58.0),
}


def mapped_observations(manifest: pd.DataFrame, multiple_only: bool) -> pd.DataFrame:
    """Return one map point per encounter, with all labeled otters at that point."""
    labeled = manifest.dropna(subset=["individual_id", "observation_id"]).copy()
    labeled["individual_id"] = labeled["individual_id"].astype(str)
    labeled["observation_id"] = labeled["observation_id"].astype(str)
    encounter_counts = labeled.groupby("individual_id")["observation_id"].nunique()
    if multiple_only:
        eligible_ids = set(encounter_counts[encounter_counts > 1].index)
        labeled = labeled[labeled["individual_id"].isin(eligible_ids)]

    labeled = infer_encounter_locations(labeled)

    located = labeled.loc[valid_location_mask(labeled)].copy()
    if located.empty:
        return pd.DataFrame(
            columns=[
                "observation_id",
                "latitude",
                "longitude",
                "title",
                "detail",
                "otter_ids",
            ]
        )
    located["latitude"] = pd.to_numeric(located["latitude"], errors="coerce")
    located["longitude"] = pd.to_numeric(located["longitude"], errors="coerce")

    metadata = load_source_metadata(file_token(RESULTS_PATH))
    observers_by_obs: dict[str, str] = {}
    if not metadata.empty and "observation_id" in metadata.columns and "observer" in metadata.columns:
        valid_meta = metadata.dropna(subset=["observation_id", "observer"])
        for _, row in valid_meta.iterrows():
            obs_id = str(row["observation_id"]).strip()
            obs_name = str(row["observer"]).strip()
            if obs_id and obs_name and obs_id not in observers_by_obs:
                observers_by_obs[obs_id] = obs_name

    rows = []
    for observation_id, group in located.groupby("observation_id", sort=False):
        otter_ids = sorted(group["individual_id"].unique())
        label = "otter" if len(otter_ids) == 1 else "otters"
        observer = str(observers_by_obs.get(observation_id, "")).strip()
        obs_text = f" · {observer}" if observer and observer.lower() not in ("", "unknown", "nan") else ""

        loc_src = str(group["location_source"].iloc[0]) if "location_source" in group.columns else ""
        inferred_text = " (Inferred Location)" if loc_src == "inferred_temporal_centroid" else ""

        rows.append(
            {
                "observation_id": observation_id,
                "latitude": float(group["latitude"].median()),
                "longitude": float(group["longitude"].median()),
                "title": f"{len(otter_ids)} {label}",
                "detail": f"{', '.join(otter_ids)}{obs_text} · Observation {observation_id}{inferred_text}",
                "otter_ids": otter_ids,
            }
        )
    return pd.DataFrame(rows)


def filter_points_by_geography(
    points: pd.DataFrame, geography: str
) -> pd.DataFrame:
    """Return map points within the selected approximate geographic bounds."""
    if geography not in GEOGRAPHY_BOUNDS:
        raise ValueError(f"Unknown geography: {geography}")
    bounds = GEOGRAPHY_BOUNDS[geography]
    if bounds is None:
        return points.copy()

    south, west, north, east = bounds
    latitude = pd.to_numeric(points["latitude"], errors="coerce")
    longitude = pd.to_numeric(points["longitude"], errors="coerce")
    return points.loc[
        latitude.between(south, north) & longitude.between(west, east)
    ].copy()


def main() -> None:
    st.title("Otter map")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.write("Region")
        geography = st.selectbox(
            "Geography",
            GEOGRAPHY_OPTIONS,
            index=GEOGRAPHY_OPTIONS.index("South America"),
            key="region",
            bind="query-params",
            label_visibility="collapsed",
            width=160,
        )
        show_all = st.toggle(
            "Show all",
            value=False,
            help=(
                "Off: only otters labeled in multiple encounters. "
                "On: every labeled otter."
            ),
        )
    manifest = load_manifest()
    points = mapped_observations(manifest, multiple_only=not show_all)
    points = filter_points_by_geography(points, geography)
    if points.empty:
        st.info(
            f"No labeled otters in {geography} have valid location data for "
            "this view."
        )
        return

    unique_otters = {
        individual_id
        for detail in points["detail"]
        for individual_id in detail.split(" · ", 1)[0].split(", ")
    }
    scope = "otters" if show_all else "otters with multiple encounters"
    st.caption(
        f"{len(unique_otters)} {scope} across {len(points)} encounters "
        f"in {geography}"
    )
    selected = otter_map(
        points=points[
            ["latitude", "longitude", "title", "detail", "otter_ids"]
        ].to_dict(orient="records"),
        minimum_area_square_miles=10,
        key="all-otter-map",
        default=None,
    )
    if selected and selected.get("nonce") != st.session_state.get("map_click_nonce"):
        st.session_state.map_click_nonce = selected["nonce"]
        st.switch_page(
            "view.py",
            query_params={"otter": selected["otter_id"]},
        )


if __name__ == "__main__":
    main()
