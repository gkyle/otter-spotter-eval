"""Shared utilities for the otter Streamlit pages."""

from __future__ import annotations

import ipaddress
import sys
from pathlib import Path
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.manifest import DATA_DIR, REPO_ROOT  # noqa: E402
from common.similarity_data import (  # noqa: E402
    load_similarity,
    load_similarity_breakdown,
)

QUERY_EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "miewid-msv3-finetuned"
QUERY_LOCATION_WEIGHT = 0.1
LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}


def _is_local_request() -> bool:
    """Treat localhost and private-LAN hosts as trusted for local dev."""
    host = st.context.headers.get("Host", "").rsplit(":", 1)[0].strip("[]")
    if not host or host in LOCAL_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


def require_login() -> None:
    """Gate the current page behind Google sign-in, except on the local LAN."""
    if _is_local_request():
        return

    try:
        auth_configured = "auth" in st.secrets
    except StreamlitSecretNotFoundError:
        auth_configured = False
    if not auth_configured:
        st.error(
            "Google sign-in isn't configured yet. Copy "
            "`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and "
            "fill in your Google OAuth credentials."
        )
        st.stop()

    if not st.user.is_logged_in:
        st.write("Sign in with an authorized Google account to continue.")
        if st.button("Log in with Google", icon=":material/login:"):
            st.login()
        st.stop()

    allowed_emails = set(st.secrets.get("access", {}).get("allowed_emails", []))
    if allowed_emails and st.user.email not in allowed_emails:
        st.title("Access denied")
        st.write(f"{st.user.email} is not authorized to use this app.")
        if st.button("Log out", icon=":material/logout:"):
            st.logout()
        st.stop()

    with st.sidebar:
        st.caption(f"Signed in as {st.user.email}")
        if st.button("Log out", icon=":material/logout:"):
            st.logout()


def similarity_cache_token(part_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """Return file metadata that invalidates cached query scores when rebuilt."""
    names = (
        "index.csv",
        "similarity.npy",
        "embeddings.npy",
        "locations.csv",
        "score_available.npy",
    )
    token = []
    for name in names:
        path = part_dir / name
        if path.exists():
            stat = path.stat()
            token.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(token)


@st.cache_data(show_spinner=False)
def cached_query_similarity(
    embeddings_dir: str,
    body_part: str,
    location_weight: float,
    cache_token: tuple[tuple[str, int, int], ...],
) -> tuple[np.ndarray, pd.DataFrame, str, int]:
    """Load query scores, with cache invalidation tied to analysis outputs."""
    del cache_token
    return load_similarity(Path(embeddings_dir), body_part, location_weight)


@st.cache_data(show_spinner=False)
def cached_query_similarity_breakdown(
    embeddings_dir: str,
    body_part: str,
    location_weight: float,
    cache_token: tuple[tuple[str, int, int], ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    str,
    int,
]:
    """Load score components, invalidating the cache with analysis outputs."""
    del cache_token
    return load_similarity_breakdown(
        Path(embeddings_dir), body_part, location_weight
    )


def current_query_index(index: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Overlay current manifest labels onto a possibly older embedding index."""
    current = (
        manifest.assign(annotation_id=manifest["annotation_id"].astype(str))
        .drop_duplicates("annotation_id")
        .set_index("annotation_id")
    )
    updated = index.copy()
    updated["annotation_id"] = updated["annotation_id"].astype(str)
    for column in (
        "individual_id",
        "observation_id",
        "body_part",
        "image_path",
        "latitude",
        "longitude",
        "location_source",
    ):
        if column in current.columns:
            values = updated["annotation_id"].map(current[column])
            if column in updated:
                updated[column] = values.fillna(updated[column])
            else:
                updated[column] = values
    return updated


def classify_single_match(
    otter_id: str,
    image_score: float,
    combined_score: float,
    distance_km: float | None,
    r5_count: int = 1,
    is_rank_1: bool = True,
) -> tuple[int, str, str]:
    """Classify confidence level for a candidate match.

    Returns: (confidence_level, blurb_text, circle_icon)
      - Level 1: 🟢 We think this is <otter-NNNN>
      - Level 2: 🟡 This might be <otter-NNNN>
      - Level 3: 🔴 We can't identify this otter
    """
    has_location = distance_km is not None
    near_location = has_location and distance_km < 60.0
    plausible_location = has_location and distance_km < 100.0

    # Level 1: "We think this is <otter-NNNN>" (Only applies to Rank 1)
    if is_rank_1 and (
        (near_location and (image_score >= 0.25 or combined_score >= 0.25))
        or (not has_location and image_score >= 0.35)
    ):
        return 1, f"We think this is {otter_id}", "🟢"

    # Level 2: "This might be <otter-NNNN>"
    if (
        (plausible_location and image_score >= 0.12)
        or image_score >= 0.25
        or (r5_count >= 2 and image_score >= 0.10)
    ):
        return 2, f"This might be {otter_id}", "🟡"

    # Level 3: "We can't identify this otter"
    return 3, "We can't identify this otter", "🔴"


def get_match_confidence(matches: list[dict]) -> tuple[int, str, str]:
    """Return (confidence_level, blurb_text, circle_icon) for top candidate in matches list."""
    if not matches:
        return 3, "We can't identify this otter", "🔴"

    top_match = matches[0]
    otter_id = top_match["individual_id"]
    top_5_ids = [m["individual_id"] for m in matches[:5]]
    r5_count = top_5_ids.count(otter_id)

    return classify_single_match(
        otter_id=otter_id,
        image_score=top_match["image_score"],
        combined_score=top_match["combined_score"],
        distance_km=top_match["distance_km"],
        r5_count=r5_count,
        is_rank_1=True,
    )

def label_url_for_encounter(
    observation_id: str, image_path: str | None = None
) -> str:
    """Build a URL pointing to the label page for an encounter and optional image."""
    obs_id = str(observation_id or "").strip()
    params = {}
    if obs_id and obs_id != "nan":
        params["encounter_id"] = obs_id
    if image_path:
        img_str = str(image_path).strip()
        if img_str and img_str != "nan":
            params["image"] = img_str
    return f"label?{urlencode(params)}" if params else "label"

def encounter_url_for_encounter(
    observation_id: str, image_path: str | None = None
) -> str:
    """Build a URL pointing to the label page for an encounter and optional image."""
    obs_id = str(observation_id or "").strip()
    params = {}
    if obs_id and obs_id != "nan":
        params["observation"] = obs_id
    if image_path:
        img_str = str(image_path).strip()
        if img_str and img_str != "nan":
            params["image"] = img_str
    return f"encounter?{urlencode(params)}" if params else "encounter"


def show_query_crop(
    container,
    row: pd.Series,
    caption: str,
    link_individual_id: str | None = None,
    circle_icon: str | None = None,
) -> None:
    """Show one cached crop, falling back to a warning when it is unavailable.

    When ``link_individual_id`` is given, the first line of ``caption`` is
    rendered as a hyperlink to that individual's query page instead of plain
    text. The encounter ID (if present in caption) is rendered as a link to the
    label page for that specific image and encounter.
    """
    crop_path = REPO_ROOT / str(row.crop_path)
    if not crop_path.is_file():
        container.warning(f"Missing crop: {row.crop_path}")
        return
    container.image(str(crop_path), width="stretch")

    obs_id = str(getattr(row, "observation_id", "") or "").strip()
    if obs_id and obs_id != "nan":
        img_path = getattr(row, "image_path", None)
        target_url = encounter_url_for_encounter(obs_id, img_path)
        link_html = f'<a href="{target_url}">{obs_id}</a>'
        if f"obs {obs_id}" in caption:
            caption = caption.replace(f"obs {obs_id}", f"obs {link_html}")
        elif f">{obs_id}<" not in caption and f"[{obs_id}]" not in caption and obs_id in caption:
            caption = caption.replace(obs_id, link_html)

    if link_individual_id is None:
        formatted_caption = "  \n".join(caption.splitlines())
        container.caption(formatted_caption, unsafe_allow_html=True)
        return
    first_line, _, rest = caption.partition("\n")
    label_text = f"{circle_icon} {first_line}" if circle_icon else first_line
    container.page_link(
        "query.py", label=label_text, query_params={"otter": link_individual_id}
    )
    if rest:
        formatted_rest = "  \n".join(rest.splitlines())
        container.caption(formatted_rest, unsafe_allow_html=True)
