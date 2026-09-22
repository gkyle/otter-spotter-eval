#!/usr/bin/env python3
"""Report the farthest pair of encounters for each labeled individual.

Coordinates come from the annotation manifest. Repeated annotations are
deduplicated, and encounters containing multiple image coordinates use their
median coordinate.

Usage:
    uv run src/max_encounter_distance.py
    uv run src/max_encounter_distance.py --include-singletons
    uv run src/max_encounter_distance.py --output data/eval/distances.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.manifest import MANIFEST_PATH, REPO_ROOT, load_manifest  # noqa: E402
from common.manifest import (  # noqa: E402
    MANIFEST_PATH,
    REPO_ROOT,
    is_valid_location,
    load_manifest,
)

EARTH_RADIUS_KM = 6371.0088


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/max_encounter_distances.csv"),
    )
    parser.add_argument(
        "--include-singletons",
        action="store_true",
        help="Include individuals with fewer than two located encounters.",
    )
    return parser.parse_args()


def haversine_km(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    """Great-circle distance between (longitude, latitude) coordinates."""
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(value)))


def individual_distances(
    manifest: pd.DataFrame,
    locations: dict[str, tuple[float, float]],
    include_singletons: bool = False,
) -> pd.DataFrame:
    """Return one geographic-diameter row per eligible individual."""
    rows = []
    grouped = manifest.groupby("individual_id", sort=True)
    for individual_id, annotations in grouped:
        encounters = sorted(
            {
                str(value)
                for value in annotations["observation_id"].dropna()
                if str(value)
            }
        )
        located = [(encounter, locations[encounter]) for encounter in encounters
                   if encounter in locations]
        missing = [encounter for encounter in encounters if encounter not in locations]
        if len(located) < 2 and not include_singletons:
            continue

        farthest_distance = math.nan
        farthest_a = ""
        farthest_b = ""
        for (encounter_a, coordinate_a), (encounter_b, coordinate_b) in combinations(
            located, 2
        ):
            distance = haversine_km(coordinate_a, coordinate_b)
            if math.isnan(farthest_distance) or distance > farthest_distance:
                farthest_distance = distance
                farthest_a = encounter_a
                farthest_b = encounter_b

        rows.append(
            {
                "individual_id": str(individual_id),
                "encounter_count": len(encounters),
                "located_encounter_count": len(located),
                "missing_location_count": len(missing),
                "max_distance_km": farthest_distance,
                "encounter_a": farthest_a,
                "encounter_b": farthest_b,
                "missing_encounters": ";".join(missing),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["max_distance_km", "individual_id"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def manifest_locations(manifest: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Return encounter coordinates derived from manifest annotations."""
    frame = manifest[["observation_id", "latitude", "longitude"]].copy()
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame = frame.dropna(subset=["observation_id", "latitude", "longitude"])
    frame = frame[
        frame.apply(
            lambda row: is_valid_location(row["latitude"], row["longitude"]),
            axis=1,
        )
    ]
    grouped = frame.groupby("observation_id")[["longitude", "latitude"]].median()
    return {
        str(observation_id): (float(row.longitude), float(row.latitude))
        for observation_id, row in grouped.iterrows()
    }


def display_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    locations = manifest_locations(manifest)
    result = individual_distances(
        manifest, locations, include_singletons=args.include_singletons
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    if result.empty:
        print("No individuals have enough located encounters to compare.")
    else:
        printable = result[
            [
                "individual_id",
                "encounter_count",
                "located_encounter_count",
                "max_distance_km",
                "encounter_a",
                "encounter_b",
            ]
        ].copy()
        printable["max_distance_km"] = printable["max_distance_km"].map(
            lambda value: "n/a" if pd.isna(value) else f"{value:.2f}"
        )
        print(printable.to_string(index=False))
        missing = int((result["missing_location_count"] > 0).sum())
        print(
            f"\n{len(result)} individual(s); "
            f"{missing} have at least one encounter without coordinates."
        )
    print(f"Wrote {display_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
