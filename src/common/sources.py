"""Common data sources helper module for otter-spotter.

Provides runtime access to data/results.csv and review-result helpers.
Decoupled from offline data source scrapers, fetchers, and translators.
"""

from __future__ import annotations

import functools
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_CSV = DATA_DIR / "results.csv"
RESULTS_PATH = RESULTS_CSV
RESULTS_REVIEW_CSV = DATA_DIR / "results_review.csv"

COMMON_COLUMNS = [
    "source",
    "encounter_id",
    "photo_id",
    "image_path",
    "image_url",
    "page_url",
    "observed_at",
    "observer",
    "latitude",
    "longitude",
    "location_source",
    "locality",
    "license",
    "quality_grade",
]


def results_file_token() -> tuple[int, int]:
    """Return cache invalidation metadata for data/results.csv."""
    if not RESULTS_CSV.is_file():
        return 0, 0
    stat = RESULTS_CSV.stat()
    return stat.st_mtime_ns, stat.st_size


@functools.lru_cache(maxsize=1)
def _cached_results(mtime_ns: int, size: int) -> pd.DataFrame:
    del mtime_ns, size
    if not RESULTS_CSV.is_file():
        return pd.DataFrame(columns=COMMON_COLUMNS)
    return pd.read_csv(RESULTS_CSV, dtype=str)


def load_sources_results() -> pd.DataFrame:
    """Load the consolidated results CSV with automatic invalidation on file changes."""
    token = results_file_token()
    return _cached_results(token[0], token[1]).copy()


def save_sources_results(df: pd.DataFrame, target_path: Path = RESULTS_CSV) -> Path:
    """Atomically save consolidated results and invalidate runtime caches."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp = target_path.with_name(f".{target_path.name}.tmp")
    df.to_csv(temp, index=False)
    temp.replace(target_path)
    _cached_results.cache_clear()
    return target_path


def review_file_token() -> tuple[int, int]:
    """Return cache invalidation metadata for data/results_review.csv."""
    if not RESULTS_REVIEW_CSV.is_file():
        return 0, 0
    stat = RESULTS_REVIEW_CSV.stat()
    return stat.st_mtime_ns, stat.st_size


def load_review_results(target_path: Path = RESULTS_REVIEW_CSV) -> pd.DataFrame:
    """Load newly uploaded staged records from data/results_review.csv."""
    if not target_path.is_file():
        return pd.DataFrame(columns=COMMON_COLUMNS)
    try:
        df = pd.read_csv(target_path, dtype=str)
        for col in COMMON_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[COMMON_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=COMMON_COLUMNS)


def save_review_results(df: pd.DataFrame, target_path: Path = RESULTS_REVIEW_CSV) -> Path:
    """Save review results atomically."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp = target_path.with_name(f".{target_path.name}.tmp")
    save_df = df.copy()
    for col in COMMON_COLUMNS:
        if col not in save_df.columns:
            save_df[col] = ""
    save_df[COMMON_COLUMNS].to_csv(temp, index=False)
    temp.replace(target_path)
    return target_path


def append_review_records(
    new_records: list[dict[str, str]], target_path: Path = RESULTS_REVIEW_CSV
) -> Path:
    """Append new review rows to results_review.csv, avoiding duplicate image_path entries."""
    df_existing = load_review_results(target_path=target_path)
    df_new = pd.DataFrame(new_records, columns=COMMON_COLUMNS)
    if not df_existing.empty and not df_new.empty:
        existing_paths = set(df_existing["image_path"].dropna())
        df_new = df_new[~df_new["image_path"].isin(existing_paths)]
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    return save_review_results(df_combined, target_path=target_path)


def remove_review_record(image_path: str, target_path: Path = RESULTS_REVIEW_CSV) -> None:
    """Remove a committed review row from results_review.csv by its image_path."""
    if not target_path.is_file():
        return
    df = load_review_results(target_path=target_path)
    if df.empty:
        return
    m = df["image_path"].astype(str) != str(image_path)
    save_review_results(df[m], target_path=target_path)


def update_source_records(
    source: str, new_records: list[dict[str, str]], target_path: Path = RESULTS_CSV
) -> Path:
    """Update records for a specific source in results.csv while preserving other sources."""
    if target_path.is_file():
        df_existing = pd.read_csv(target_path, dtype=str)
        df_other = df_existing[df_existing["source"] != source]
    else:
        df_other = pd.DataFrame(columns=COMMON_COLUMNS)

    df_new = pd.DataFrame(new_records, columns=COMMON_COLUMNS)
    df_combined = pd.concat([df_other, df_new], ignore_index=True)
    return save_sources_results(df_combined, target_path=target_path)
