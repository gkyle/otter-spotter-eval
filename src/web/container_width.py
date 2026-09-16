"""Registration for the responsive container-width Streamlit component."""

from pathlib import Path

import streamlit.components.v1 as components

container_width = components.declare_component(
    "container_width",
    path=Path(__file__).resolve().parent / "container_width_component",
)
