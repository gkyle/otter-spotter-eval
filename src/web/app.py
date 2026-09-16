#!/usr/bin/env python3
"""Single Streamlit entrypoint for the otter web tools.

Run with:
    uv run streamlit run src/web/app.py
"""

from pathlib import Path
import sys

import streamlit as st

WEB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB_DIR.parent))
from common.web import require_login  # noqa: E402

st.set_page_config(page_title="Otter Spotter", layout="wide")

require_login()

# Streamlit's default block-container padding ("8rem 1rem 10rem") wastes vertical space.
st.html(
    """
    <style>
    .block-container {
        max-width: 1024px !important;
        padding: 2rem 1rem 1rem !important;
    }
    [data-testid="stHeading"] h1 { font-size: 2em !important; }
    [data-testid="stToolbar"] .rc-overflow { padding: 0px 8px; }
    header[data-testid="stHeader"], [data-testid="stHeader"] {
        border-bottom: 1px solid #333 !important;
        height: 2rem !important;
        min-height: 2rem !important;
    }

    .block-container a:not([data-testid="stTopNavLink"]),
    [data-testid="stMarkdownContainer"] a,
    [data-testid="stCaptionContainer"] a,
    .stMarkdown a,
    a:not([data-testid="stTopNavLink"]) {
        color: #4b8b8b;
    }

    .block-container a:not([data-testid="stTopNavLink"]):hover,
    [data-testid="stMarkdownContainer"] a:hover,
    [data-testid="stCaptionContainer"] a:hover,
    .stMarkdown a:hover,
    a:not([data-testid="stTopNavLink"]):hover {
        color: #386767;
        text-decoration: underline;
    }
    </style>
    """
)


def home() -> None:
    """Render the root landing page without consuming a named URL route."""
    st.title("Otter Spotter")
    st.page_link(label_page, label="Open labeler", icon=":material/draw:")
    st.page_link(query_page, label="Open retrieval query", icon=":material/search:")
    st.page_link(view_page, label="Open encounter viewer", icon=":material/map:")
    st.page_link(encounter_page, label="Open encounters", icon=":material/map:")
    st.page_link(map_page, label="Open all-otter map", icon=":material/public:")
    st.page_link(upload_page, label="Upload new photos", icon=":material/upload:")


home_page = st.Page(home, title="Home", default=True, visibility="hidden")
label_page = st.Page(
    WEB_DIR / "label.py",
    title="Label",
    icon=":material/draw:",
    url_path="label",
)
query_page = st.Page(
    WEB_DIR / "query.py",
    title="Query",
    icon=":material/search:",
    url_path="query",
)
view_page = st.Page(
    WEB_DIR / "view.py",
    title="Otters",
    icon=":material/map:",
    url_path="view",
)
map_page = st.Page(
    WEB_DIR / "map.py",
    title="Map",
    icon=":material/public:",
    url_path="map",
)
encounter_page = st.Page(
    WEB_DIR / "encounter.py",
    title="Encounters",
    icon=":material/photo_library:",
    url_path="encounter",
)
upload_page = st.Page(
    WEB_DIR / "upload.py",
    title="Upload",
    icon=":material/upload:",
    url_path="upload",
)

navigation = st.navigation(
    [map_page, view_page, encounter_page, upload_page, query_page, label_page],
    position="top",
)
navigation.run()
