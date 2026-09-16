"""Registration for the interactive Leaflet Streamlit component."""

from pathlib import Path

import streamlit.components.v1 as components

otter_map = components.declare_component(
    "otter_map",
    path=Path(__file__).resolve().parent / "map_component",
)
