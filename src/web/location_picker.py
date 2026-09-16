"""Registration for the click-to-pick location Streamlit component."""

from pathlib import Path
from typing import Callable

import streamlit as st
import streamlit.components.v1 as components

from common.manifest import is_valid_location

location_picker = components.declare_component(
    "location_picker",
    path=Path(__file__).resolve().parent / "location_picker_component",
)


@st.dialog("Pick a location on the map")
def location_picker_dialog(
    dialog_key: str,
    latitude: float | None,
    longitude: float | None,
    on_confirm: Callable[[float, float], None],
) -> None:
    """Show the click-to-pick map in a popup and hand the choice to ``on_confirm``."""
    if not is_valid_location(latitude, longitude):
        latitude = None
        longitude = None
    with st.container(width=800):
        picked = location_picker(
            latitude=latitude, longitude=longitude, key=f"picker_{dialog_key}"
        )
    if picked is None and latitude is not None and longitude is not None:
        picked = {"latitude": latitude, "longitude": longitude}
    if picked is not None and not is_valid_location(
        picked.get("latitude"), picked.get("longitude")
    ):
        picked = None
    if picked is None:
        st.caption("Click the map to choose a point.")
    else:
        st.caption(f"Selected: {picked['latitude']:.6f}, {picked['longitude']:.6f}")

    button_columns = st.columns(2)
    if button_columns[0].button(
        "Use this location", type="primary", disabled=picked is None, width="stretch"
    ):
        on_confirm(picked["latitude"], picked["longitude"])
        st.rerun()
    if button_columns[1].button("Cancel", width="stretch"):
        st.rerun()
