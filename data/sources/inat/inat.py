#!/usr/bin/env python3
"""iNaturalist data source translation and helper module."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

SOURCE_NAME = "inat"


def _normalize_val(val: object) -> str:
    if pd.isna(val) or val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "null") else s


def translate_to_results(repo_root: Path | None = None) -> list[dict[str, str]]:
    """Translate iNaturalist observation & photo metadata into unified results rows."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    source_dir = repo_root / "data" / "sources" / "inat"
    inat_results_path = source_dir / "inaturalist_results.csv"
    inat_photos_path = source_dir / "inaturalist_photos.csv"
    inat_images_dir = source_dir / "images"

    # Map disk files: (obs_id, photo_id) -> rel_path
    inat_disk_files: dict[tuple[str, str], str] = {}
    if inat_images_dir.is_dir():
        for p in inat_images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                obs_id = p.parent.name
                photo_id = p.stem
                rel_path = str(p.relative_to(repo_root))
                inat_disk_files[(obs_id, photo_id)] = rel_path

    # Load inat observation metadata
    inat_obs_meta: dict[str, dict[str, str]] = {}
    if inat_results_path.is_file():
        df_obs = pd.read_csv(inat_results_path, dtype=str)
        for _, r in df_obs.iterrows():
            obs_id = _normalize_val(r.get("observation_id"))
            if not obs_id:
                continue
            obs_at = _normalize_val(r.get("time_observed_at")) or _normalize_val(r.get("observed_on"))
            observer = _normalize_val(r.get("observer_name")) or _normalize_val(r.get("observer_login"))
            inat_obs_meta[obs_id] = {
                "observed_at": obs_at,
                "observer": observer,
                "latitude": _normalize_val(r.get("latitude")),
                "longitude": _normalize_val(r.get("longitude")),
                "locality": _normalize_val(r.get("place_guess")),
                "page_url": _normalize_val(r.get("uri")) or f"https://www.inaturalist.org/observations/{obs_id}",
                "quality_grade": _normalize_val(r.get("quality_grade")),
                "license": _normalize_val(r.get("observation_license_code")),
            }

    rows: list[dict[str, str]] = []
    recorded_photos: set[tuple[str, str]] = set()

    # Load inat photos
    if inat_photos_path.is_file():
        df_photos = pd.read_csv(inat_photos_path, dtype=str)
        for _, pr in df_photos.iterrows():
            obs_id = _normalize_val(pr.get("observation_id"))
            photo_id = _normalize_val(pr.get("photo_id"))
            if not obs_id or not photo_id:
                continue
            recorded_photos.add((obs_id, photo_id))
            meta = inat_obs_meta.get(obs_id, {})
            image_path = inat_disk_files.get((obs_id, photo_id), "")
            photo_url = _normalize_val(pr.get("original_url")) or _normalize_val(pr.get("url"))
            license_code = _normalize_val(pr.get("license_code")) or meta.get("license", "")

            rows.append({
                "source": SOURCE_NAME,
                "encounter_id": f"inat_{obs_id}",
                "photo_id": photo_id,
                "image_path": image_path,
                "image_url": photo_url,
                "page_url": meta.get("page_url", f"https://www.inaturalist.org/observations/{obs_id}"),
                "observed_at": meta.get("observed_at", ""),
                "observer": meta.get("observer", ""),
                "latitude": meta.get("latitude", ""),
                "longitude": meta.get("longitude", ""),
                "location_source": SOURCE_NAME,
                "locality": meta.get("locality", ""),
                "license": license_code,
                "quality_grade": meta.get("quality_grade", ""),
            })

    # Include disk files not present in inaturalist_photos.csv
    for (obs_id, photo_id), image_path in inat_disk_files.items():
        if (obs_id, photo_id) not in recorded_photos:
            meta = inat_obs_meta.get(obs_id, {})
            rows.append({
                "source": SOURCE_NAME,
                "encounter_id": f"inat_{obs_id}",
                "photo_id": photo_id,
                "image_path": image_path,
                "image_url": "",
                "page_url": meta.get("page_url", f"https://www.inaturalist.org/observations/{obs_id}"),
                "observed_at": meta.get("observed_at", ""),
                "observer": meta.get("observer", ""),
                "latitude": meta.get("latitude", ""),
                "longitude": meta.get("longitude", ""),
                "location_source": SOURCE_NAME,
                "locality": meta.get("locality", ""),
                "license": meta.get("license", ""),
                "quality_grade": meta.get("quality_grade", ""),
            })

    # Include observations with no photos yet
    obs_with_photos = {r["encounter_id"].removeprefix("inat_") for r in rows if r["source"] == SOURCE_NAME}
    for obs_id, meta in inat_obs_meta.items():
        if obs_id not in obs_with_photos:
            rows.append({
                "source": SOURCE_NAME,
                "encounter_id": f"inat_{obs_id}",
                "photo_id": "",
                "image_path": "",
                "image_url": "",
                "page_url": meta.get("page_url", f"https://www.inaturalist.org/observations/{obs_id}"),
                "observed_at": meta.get("observed_at", ""),
                "observer": meta.get("observer", ""),
                "latitude": meta.get("latitude", ""),
                "longitude": meta.get("longitude", ""),
                "location_source": SOURCE_NAME,
                "locality": meta.get("locality", ""),
                "license": meta.get("license", ""),
                "quality_grade": meta.get("quality_grade", ""),
            })

    return rows


def update_results_csv(repo_root: Path | None = None) -> Path:
    """Update inat records in data/results.csv while preserving other sources."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    records = translate_to_results(repo_root)
    import sys
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from common.sources import update_source_records
    return update_source_records(SOURCE_NAME, records, target_path=repo_root / "data" / "results.csv")


if __name__ == "__main__":
    records = translate_to_results()
    print(f"iNaturalist: translated {len(records)} records.")

