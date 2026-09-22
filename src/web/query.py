#!/usr/bin/env python3
"""Streamlit page for inspecting individual retrieval results.

Run with:
    uv run streamlit run src/web/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.manifest import BODY_PARTS, load_manifest  # noqa: E402
from common.manifest import BODY_PARTS, is_valid_location, load_manifest  # noqa: E402
from common.max_encounter_distance import haversine_km  # noqa: E402

from common.web import (  # noqa: E402
    QUERY_EMBEDDINGS_DIR,
    QUERY_LOCATION_WEIGHT,
    cached_query_similarity_breakdown,
    classify_single_match,
    current_query_index,
    encounter_url_for_encounter,
    show_query_crop,
    similarity_cache_token,
)


def select_query_individual(individual_id: str) -> None:
    """Select an individual before Streamlit recreates the selectbox."""
    st.session_state["query_selected_otter"] = individual_id
    st.query_params["otter"] = individual_id


def store_query_selected_otter_in_url() -> None:
    """Persist the completed selector change without racing the widget rerun."""
    st.query_params["otter"] = st.session_state.query_selected_otter


def render_query_view(manifest: pd.DataFrame) -> None:
    """Render nearest cross-observation matches for an individual's crops."""
    st.title("Individual retrieval query")
    known_ids = sorted(manifest["individual_id"].dropna().astype(str).unique())
    if not known_ids:
        st.info("No labeled individuals are available yet.")
        return

    requested_individual = str(st.query_params.get("otter", ""))
    if requested_individual in known_ids and (
        requested_individual != st.session_state.get("query_applied_url_otter", "")
        # Navigating to another page clears this widget's state, so the
        # URL otter must be reapplied even if it matches the last-applied value.
        or "query_selected_otter" not in st.session_state
    ):
        st.session_state.query_selected_otter = requested_individual
    st.session_state.query_applied_url_otter = requested_individual

    if st.session_state.get("query_selected_otter") not in known_ids:
        st.session_state.query_selected_otter = known_ids[0]
        st.query_params["otter"] = known_ids[0]

    controls = st.columns([3, 2, 1, 2])
    individual_id = controls[0].selectbox(
        "Individual ID",
        known_ids,
        key="query_selected_otter",
        on_change=store_query_selected_otter_in_url,
    )
    default_part_index = (
        BODY_PARTS.index("throat_portrait") if "throat_portrait" in BODY_PARTS else 0
    )
    body_part = controls[1].selectbox(
        "Annotation class", BODY_PARTS, index=default_part_index
    )
    top_k = int(
        controls[2].number_input("Matches", min_value=1, max_value=10, value=5)
    )
    exclude_same_individual = controls[3].toggle(
        "Exclude same individual",
        value=False,
        help="Show only crops labeled as other individuals in the match results.",
    )
    # st.caption(
    #    f"Results use {1 - QUERY_LOCATION_WEIGHT:.0%} image similarity and "
    #    f"{QUERY_LOCATION_WEIGHT:.0%} location similarity when both images have "
    #    "coordinates; otherwise they use image similarity alone. Images from "
    #    "the query observation are always excluded."
    # )

    part_dir = QUERY_EMBEDDINGS_DIR / body_part
    try:
        (
            similarity,
            image_similarity,
            location_similarity,
            location_pair_available,
            index,
            source,
            locations_available,
        ) = cached_query_similarity_breakdown(
            str(QUERY_EMBEDDINGS_DIR),
            body_part,
            QUERY_LOCATION_WEIGHT,
            similarity_cache_token(part_dir),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Query scores are unavailable: {exc}")
        st.code("Rerun: src/identify/miewid/extract_embeddings.py")
        return

    index = current_query_index(index, manifest)
    available_ids = sorted(
        index.loc[
            index["body_part"].astype(str) == body_part,
            "individual_id",
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    if individual_id in available_ids:
        individual_position = available_ids.index(individual_id)
        previous_column, position_column, next_column = st.columns([1, 2, 1])
        previous_column.button(
            "← Previous individual",
            disabled=individual_position == 0,
            on_click=select_query_individual,
            args=(
                available_ids[individual_position - 1]
                if individual_position > 0
                else individual_id,
            ),
            width="stretch",
        )
        position_column.markdown(
            f"<p style='text-align: center'><strong>Individual "
            f"{individual_position + 1} of {len(available_ids)}</strong></p>",
            unsafe_allow_html=True,
        )
        next_column.button(
            "Next individual →",
            disabled=individual_position == len(available_ids) - 1,
            on_click=select_query_individual,
            args=(available_ids[individual_position + 1],)
            if individual_position + 1 < len(available_ids)
            else (individual_id,),
            width="stretch",
        )

    manifest_matches = manifest[
        (manifest["individual_id"].astype(str) == individual_id)
        & (manifest["body_part"].astype(str) == body_part)
    ]
    indexed_annotation_ids = set(index["annotation_id"].astype(str))
    missing_from_index = [
        annotation_id
        for annotation_id in manifest_matches["annotation_id"].astype(str)
        if annotation_id not in indexed_annotation_ids
    ]
    if missing_from_index:
        st.warning(
            f"{len(missing_from_index)} of this individual's "
            f"{len(manifest_matches)} {body_part} annotation(s) are not in the "
            "current analysis index. Rerun the analysis to include them."
        )
    query_rows = np.flatnonzero(
        (index["individual_id"].astype(str).to_numpy() == individual_id)
        & (index["body_part"].astype(str).to_numpy() == body_part)
    )
    unique_query_rows = []
    seen_images: set[str] = set()
    for row_index in query_rows:
        image_path = str(index.iloc[row_index].image_path)
        if image_path not in seen_images:
            seen_images.add(image_path)
            unique_query_rows.append(int(row_index))
    query_rows = np.asarray(unique_query_rows, dtype=np.int64)
    if not len(query_rows):
        if not len(manifest_matches):
            st.info(f"{individual_id} has no {body_part} annotations.")
        return

    st.caption(
        f"{len(query_rows)} query crop(s) · {source} · "
        f"{locations_available} crops with location data"
    )
    individuals = index["individual_id"].astype(str).to_numpy()
    observations = index["observation_id"].astype(str).to_numpy()

    for query_number, query_index in enumerate(query_rows, start=1):
        query = index.iloc[query_index]
        query_url = encounter_url_for_encounter(
            query.observation_id, getattr(query, "image_path", None)
        )
        st.subheader(
            f"Query {query_number}: {individual_id} · observation "
            f"[{query.observation_id}]({query_url})"
        )
        gallery_mask = observations != observations[query_index]
        if exclude_same_individual:
            gallery_mask &= individuals != individuals[query_index]
        gallery = np.flatnonzero(gallery_mask)
        order = gallery[
            np.argsort(-similarity[query_index, gallery], kind="stable")
        ][:top_k]
        columns = st.columns(top_k + 1)
        show_query_crop(
            columns[0],
            query,
            f"QUERY\n{individual_id}\nobs {query.observation_id}",
        )
        top_5_ids = [str(individuals[idx]) for idx in order[:5]]
        for rank_idx, (column, match_index) in enumerate(zip(columns[1:], order)):
            match = index.iloc[match_index]
            match_otter_id = str(individuals[match_index])
            score = float(similarity[query_index, match_index])
            image_score = float(image_similarity[query_index, match_index])
            has_loc_pair = False
            q_lat = q_lon = m_lat = m_lon = 0.0
            try:
                q_lat, q_lon = float(query.latitude), float(query.longitude)
                m_lat, m_lon = float(match.latitude), float(match.longitude)
                if is_valid_location(q_lat, q_lon) and is_valid_location(m_lat, m_lon):
                    has_loc_pair = True
            except (ValueError, TypeError):
                has_loc_pair = False

            if has_loc_pair:
                distance_km = haversine_km(
                    (q_lon, q_lat),
                    (m_lon, m_lat),
                )
                if location_pair_available[query_index, match_index] and not np.isnan(
                    location_similarity[query_index, match_index]
                ):
                    loc_val = float(location_similarity[query_index, match_index])
                else:
                    from location.geodistance import location_score as calc_loc_score

                    loc_val = float(calc_loc_score(np.array([distance_km]))[0])
                location_score = f"{loc_val:.3f} ({distance_km:,.1f} km)"
            else:
                distance_km = None
                location_score = "n/a"

            r5_count = top_5_ids.count(match_otter_id)
            level, blurb, circle_icon = classify_single_match(
                otter_id=match_otter_id,
                image_score=image_score,
                combined_score=score,
                distance_km=distance_km,
                r5_count=r5_count,
                is_rank_1=(rank_idx == 0),
            )

            show_query_crop(
                column,
                match,
                f"{match_otter_id}\n"
                f"obs {match.observation_id}\n"
                f"score: {score:.3f}\n"
                f"image: {image_score:.3f}\n"
                f"location: {location_score}",
                link_individual_id=match_otter_id,
                circle_icon=circle_icon,
            )
        st.divider()


def main() -> None:
    render_query_view(load_manifest())


if __name__ == "__main__":
    main()
