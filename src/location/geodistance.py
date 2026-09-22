"""Great-circle distance kernel used by the location pipeline.

Location similarity is derived directly from the great-circle distance between
two coordinates. Same-individual encounters cluster within a Giant River Otter's
home range, and most coordinates are deliberately obscured to a ~20 km grid, so
matches remain plausible out to roughly 60 km and become increasingly unlikely
beyond that. The kernel therefore gives full credit inside a plateau radius and
decays smoothly afterwards (see ``location_score``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common.manifest import REPO_ROOT, is_valid_location

RESULTS_CSV = REPO_ROOT / "data" / "results.csv"
LOCATIONS_FILE = "locations.csv"
EARTH_RADIUS_KM = 6371.0088
# Full credit within the plateau, half-Gaussian decay beyond. Calibrated to the
# same-individual encounter distances in data/eval/max_encounter_distances.csv:
# most confirmed pairs fall under ~60 km and the score is near zero by ~250 km.
LOCATION_PLATEAU_KM = 60.0
LOCATION_DECAY_KM = 90.0


def pairwise_haversine_km(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """Great-circle distances (km) between two (longitude, latitude) arrays.

    ``coords_a`` and ``coords_b`` are shaped (N, 2) and (M, 2) in degrees; the
    result is the (N, M) matrix of great-circle distances in kilometres.
    """
    a = np.radians(np.asarray(coords_a, dtype=np.float64))
    b = np.radians(np.asarray(coords_b, dtype=np.float64))
    if a.ndim != 2 or a.shape[1] != 2 or b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("coordinates must be shaped (N, 2) longitude/latitude")
    lon_a, lat_a = a[:, 0][:, None], a[:, 1][:, None]
    lon_b, lat_b = b[:, 0][None, :], b[:, 1][None, :]
    delta = (
        np.sin((lat_b - lat_a) / 2) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin((lon_b - lon_a) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(delta, 0.0, 1.0)))


def location_score(
    distance_km: np.ndarray,
    plateau_km: float = LOCATION_PLATEAU_KM,
    decay_km: float = LOCATION_DECAY_KM,
) -> np.ndarray:
    """Map great-circle distance to a [0, 1] location score (soft step).

    Scores 1.0 within ``plateau_km`` and decays as a half-Gaussian beyond it,
    so nearby crops are boosted while distant ones are increasingly penalized.
    """
    if plateau_km < 0 or decay_km <= 0:
        raise ValueError("plateau_km must be >= 0 and decay_km must be positive")
    distance = np.asarray(distance_km, dtype=np.float64)
    excess = np.maximum(0.0, distance - plateau_km)
    return np.exp(-((excess / decay_km) ** 2)).astype(np.float32)


def location_similarity_matrix(coords: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Return the (N, N) location-score matrix for row-aligned coordinates.

    Rows/columns whose coordinates are unavailable are left at 0; callers mask
    those pairs out before blending.
    """
    coords = np.asarray(coords, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    count = len(coords)
    scores = np.zeros((count, count), dtype=np.float32)
    present = np.where(available)[0]
    if present.size:
        distances = pairwise_haversine_km(coords[present], coords[present])
        scores[np.ix_(present, present)] = location_score(distances)
    return scores


def load_encounter_locations(
    path: Path = RESULTS_CSV,
) -> dict[str, tuple[float, float]]:
    """Return encounter_id -> (longitude, latitude) for encounters in results.csv."""
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype=str)
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame = frame.dropna(subset=["encounter_id", "latitude", "longitude"])
    frame = frame[
        frame["latitude"].between(-90, 90)
        & frame["longitude"].between(-180, 180)
        & ~((frame["latitude"] == 0) & (frame["longitude"] == 0))
    ]
    locations: dict[str, tuple[float, float]] = {}
    for row in frame.itertuples():
        enc = str(row.encounter_id)
        if enc not in locations:
            locations[enc] = (float(row.longitude), float(row.latitude))
    return locations


def write_location_cache(index: pd.DataFrame, out_dir: Path) -> tuple[int, int]:
    """Write the row-aligned coordinate cache from index coordinates."""
    required = {"latitude", "longitude", "location_source"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"crop index is missing columns: {', '.join(sorted(missing))}")
    latitude = pd.to_numeric(index["latitude"], errors="coerce")
    longitude = pd.to_numeric(index["longitude"], errors="coerce")
    available = (
        index.apply(
            lambda row: is_valid_location(row["latitude"], row["longitude"]),
            axis=1,
        )
        & index["location_source"].fillna("").astype(str).str.strip().ne("")
    ).to_numpy(dtype=bool)
    records = []
    for position, row in enumerate(index.itertuples()):
        present = bool(available[position])
        records.append(
            {
                "annotation_id": str(row.annotation_id),
                "observation_id": str(row.observation_id),
                "longitude": float(longitude.iloc[position]) if present else np.nan,
                "latitude": float(latitude.iloc[position]) if present else np.nan,
                "location_source": str(row.location_source) if present else "",
                "location_available": present,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_dir / LOCATIONS_FILE, index=False)
    return int(available.sum()), int((~available).sum())


def load_location_cache(out_dir: Path, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Load row-aligned (longitude, latitude) coordinates and availability.

    Coordinates for unavailable rows are NaN; the Boolean mask marks the rows
    that carry a usable location.
    """
    metadata_path = out_dir / LOCATIONS_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"location cache missing under {out_dir}; run the backend's extraction script"
        )
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != rows:
        raise ValueError(f"location cache under {out_dir} is not row-aligned")
    available = (
        metadata["location_available"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    if available.isna().any():
        raise ValueError(f"invalid availability values in {metadata_path}")
    coords = np.column_stack(
        (
            pd.to_numeric(metadata["longitude"], errors="coerce").to_numpy(),
            pd.to_numeric(metadata["latitude"], errors="coerce").to_numpy(),
        )
    ).astype(np.float64)
    return coords, available.to_numpy(dtype=bool)


def blend_similarity_matrix(
    visual: np.ndarray,
    location_coords: np.ndarray | None = None,
    location_available: np.ndarray | None = None,
    location_weight: float = 0.0,
    score_available: np.ndarray | None = None,
) -> np.ndarray:
    """Blend visual scores with the location kernel where both are available."""
    if not 0 <= location_weight <= 1:
        raise ValueError("location_weight must be between 0 and 1")
    visual = np.asarray(visual)
    if visual.ndim != 2 or visual.shape[0] != visual.shape[1]:
        raise ValueError("visual similarity must be a square matrix")
    if location_weight == 0:
        return visual
    if location_coords is None or location_available is None:
        raise ValueError("location coordinates and availability are required")
    if len(location_coords) != len(visual) or len(location_available) != len(visual):
        raise ValueError("location cache is not aligned with visual similarity")
    location = location_similarity_matrix(location_coords, location_available)
    pair_available = np.outer(location_available, location_available)
    if score_available is not None:
        score_available = np.asarray(score_available, dtype=bool)
        if score_available.shape != visual.shape:
            raise ValueError("score availability mask is not aligned with similarity")
        pair_available &= score_available
    combined = visual.copy()
    combined[pair_available] = (1 - location_weight) * visual[
        pair_available
    ] + location_weight * location[pair_available]
    return combined


def infer_encounter_locations(
    df: pd.DataFrame,
    obs_col: str = "observation_id",
    otter_col: str = "individual_id",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    date_col: str = "observed_at",
    source_col: str = "location_source",
) -> pd.DataFrame:
    """Infer missing encounter locations at runtime using date-weighted centroids of otter locations.

    Returns a copy of df with missing latitude/longitude filled for unlocated encounters
    where an otter in that encounter has other located encounters.
    """
    if df.empty or obs_col not in df.columns or otter_col not in df.columns:
        return df

    res = df.copy()
    lat = (
        pd.to_numeric(res[lat_col], errors="coerce")
        if lat_col in res.columns
        else pd.Series(dtype=float)
    )
    lon = (
        pd.to_numeric(res[lon_col], errors="coerce")
        if lon_col in res.columns
        else pd.Series(dtype=float)
    )

    has_loc = lat.notna() & lon.notna() & ((lat != 0) | (lon != 0))
    res["_has_loc"] = has_loc

    located = res[res["_has_loc"]].copy()
    if located.empty:
        res.drop(columns=["_has_loc"], inplace=True, errors="ignore")
        return res

    located["_lat_num"] = pd.to_numeric(located[lat_col], errors="coerce")
    located["_lon_num"] = pd.to_numeric(located[lon_col], errors="coerce")

    if date_col in located.columns:
        located["_date_parsed"] = pd.to_datetime(
            located[date_col], errors="coerce", utc=True
        )
    else:
        located["_date_parsed"] = pd.NaT

    otter_loc_map: dict[str, list[dict[str, object]]] = {}
    for (otter, obs_id), group in located.groupby([otter_col, obs_col]):
        otter_str = str(otter).strip()
        if not otter_str or otter_str.lower() in ("nan", "none", ""):
            continue
        first_row = group.iloc[0]
        otter_loc_map.setdefault(otter_str, []).append(
            {
                "obs_id": str(obs_id),
                "lat": float(first_row["_lat_num"]),
                "lon": float(first_row["_lon_num"]),
                "date": first_row["_date_parsed"],
            }
        )

    if date_col in res.columns:
        res["_date_parsed"] = pd.to_datetime(res[date_col], errors="coerce", utc=True)
    else:
        res["_date_parsed"] = pd.NaT

    unlocated_obs = res[~res["_has_loc"]][obs_col].unique()

    for obs_id in unlocated_obs:
        obs_mask = res[obs_col] == obs_id
        obs_rows = res[obs_mask]
        otters = [
            str(o).strip()
            for o in obs_rows[otter_col].unique()
            if str(o).strip() and str(o).strip().lower() not in ("nan", "none", "")
        ]

        enc_date = obs_rows["_date_parsed"].iloc[0] if not obs_rows.empty else pd.NaT

        candidate_locs: list[dict[str, object]] = []
        for o in otters:
            candidate_locs.extend(otter_loc_map.get(o, []))

        if not candidate_locs:
            continue

        weights = []
        for loc in candidate_locs:
            loc_date = loc["date"]
            if pd.notna(enc_date) and pd.notna(loc_date):
                days_diff = abs((enc_date - loc_date).total_seconds()) / 86400.0
                w = 1.0 / (days_diff + 1.0)
            else:
                w = 1.0
            weights.append(w)

        w_arr = np.array(weights, dtype=np.float64)
        w_sum = w_arr.sum()
        if w_sum <= 0:
            continue
        w_arr /= w_sum

        inf_lat = round(
            float(sum(w * loc["lat"] for w, loc in zip(w_arr, candidate_locs))), 6
        )
        inf_lon = round(
            float(sum(w * loc["lon"] for w, loc in zip(w_arr, candidate_locs))), 6
        )

        if pd.api.types.is_numeric_dtype(res[lat_col]):
            res.loc[obs_mask, lat_col] = inf_lat
            res.loc[obs_mask, lon_col] = inf_lon
        else:
            res.loc[obs_mask, lat_col] = str(inf_lat)
            res.loc[obs_mask, lon_col] = str(inf_lon)
        if source_col in res.columns:
            res.loc[obs_mask, source_col] = "inferred_temporal_centroid"

    # Ensure any remaining (0, 0) coordinates are cleared to unlocated (NaN / empty string)
    still_zero = (pd.to_numeric(res[lat_col], errors="coerce") == 0) & (
        pd.to_numeric(res[lon_col], errors="coerce") == 0
    )
    if still_zero.any():
        if pd.api.types.is_numeric_dtype(res[lat_col]):
            res.loc[still_zero, [lat_col, lon_col]] = np.nan
        else:
            res.loc[still_zero, [lat_col, lon_col]] = ""
        if source_col in res.columns:
            res.loc[still_zero, source_col] = ""

    res.drop(columns=["_has_loc", "_date_parsed"], inplace=True, errors="ignore")
    return res
