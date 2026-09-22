"""Load a re-identification backend's row index and pairwise similarities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from location.geodistance import (
    blend_similarity_matrix,
    load_location_cache,
    location_similarity_matrix,
)


def _load_visual_similarity(
    backend_dir: Path, body_part: str
) -> tuple[np.ndarray, pd.DataFrame, str, Path]:
    """Load the visual score matrix and its aligned crop index."""

    part_dir = backend_dir / body_part
    index_path = part_dir / "index.csv"
    similarity_path = part_dir / "similarity.npy"
    availability_path = part_dir / "score_available.npy"
    embeddings_path = part_dir / "embeddings.npy"
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    index = pd.read_csv(
        index_path, dtype={"individual_id": str, "observation_id": str}
    )
    if similarity_path.exists():
        similarity = np.load(similarity_path)
        source = "precomputed scores"
    elif embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        if embeddings.ndim != 2 or len(embeddings) != len(index):
            raise ValueError(f"{embeddings_path} is not aligned with {index_path}")
        similarity = embeddings @ embeddings.T
        source = "embedding cosine"
    else:
        raise FileNotFoundError(
            f"neither {similarity_path.name} nor {embeddings_path.name} exists "
            f"under {part_dir}"
        )

    if similarity.shape != (len(index), len(index)) or len(index) == 0:
        raise ValueError(f"similarity data under {part_dir} is not index-aligned")
    if not np.isfinite(similarity).all():
        raise ValueError(f"similarity data under {part_dir} contains non-finite values")
    return similarity, index, source, availability_path


def _score_availability(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    """Load and validate an optional pairwise visual-score availability mask."""
    if not path.exists():
        return None
    available = np.load(path)
    if available.shape != shape:
        raise ValueError(f"{path} is not aligned with visual similarity")
    return np.asarray(available, dtype=bool)


def load_similarity(
    backend_dir: Path,
    body_part: str,
    location_weight: float = 0.0,
) -> tuple[np.ndarray, pd.DataFrame, str, int]:
    """Load visual similarity and optionally blend in location similarity.

    A backend directory contains one subdirectory per body part and an
    ``index.csv`` aligned with either:

    * ``similarity.npy``: an N x N matrix where larger means more similar, or
    * ``embeddings.npy``: N L2-normalized feature vectors.

    Pairwise score backends should map scores onto the cosine-like [-1, 1]
    range before saving them so optional location blending has consistent
    semantics.
    """
    similarity, index, source, availability_path = _load_visual_similarity(
        backend_dir, body_part
    )

    locations_available = 0
    if location_weight:
        part_dir = backend_dir / body_part
        location_coords, available = load_location_cache(part_dir, len(index))
        score_available = _score_availability(
            availability_path, similarity.shape
        )
        similarity = blend_similarity_matrix(
            similarity,
            location_coords,
            available,
            location_weight,
            score_available,
        )
        locations_available = int(available.sum())
    return similarity, index, source, locations_available


def load_similarity_breakdown(
    backend_dir: Path,
    body_part: str,
    location_weight: float = 0.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    str,
    int,
]:
    """Load combined, image, and location scores with pair availability.

    Location scores are NaN for pairs where location did not participate in
    the combined score. The Boolean availability matrix makes that distinction
    explicit for callers rendering a score breakdown.
    """
    image_similarity, index, source, availability_path = _load_visual_similarity(
        backend_dir, body_part
    )
    location_similarity = np.full_like(image_similarity, np.nan)
    location_pair_available = np.zeros(image_similarity.shape, dtype=bool)
    locations_available = 0
    if not location_weight:
        return (
            image_similarity,
            image_similarity,
            location_similarity,
            location_pair_available,
            index,
            source,
            locations_available,
        )

    part_dir = backend_dir / body_part
    location_coords, available = load_location_cache(part_dir, len(index))
    locations_available = int(available.sum())
    location_pair_available = np.outer(available, available)
    score_available = _score_availability(
        availability_path, image_similarity.shape
    )
    if score_available is not None:
        location_pair_available &= score_available

    raw_location_similarity = location_similarity_matrix(location_coords, available)
    location_similarity[location_pair_available] = raw_location_similarity[
        location_pair_available
    ]
    combined = blend_similarity_matrix(
        image_similarity,
        location_coords,
        available,
        location_weight,
        score_available,
    )
    return (
        combined,
        image_similarity,
        location_similarity,
        location_pair_available,
        index,
        source,
        locations_available,
    )
