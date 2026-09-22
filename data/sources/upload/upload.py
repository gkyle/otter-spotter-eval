#!/usr/bin/env python3
"""Upload data source translation and helper module."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

SOURCE_NAME = "upload"


def _normalize_val(val: object) -> str:
    if pd.isna(val) or val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "null") else s


def translate_to_results(repo_root: Path | None = None) -> list[dict[str, str]]:
    """Translate Upload metadata into unified results rows."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    source_dir = repo_root / "data" / "sources" / "upload"
    upload_results_path = source_dir / "upload_results.csv"

    rows: list[dict[str, str]] = []
    if not upload_results_path.is_file():
        # Fall back to building upload index dynamically if upload_results.csv is missing
        try:
            import sys
            src_dir = repo_root / "src"
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            from common.upload import build_upload_index
            df_up = build_upload_index()
        except Exception:
            df_up = pd.DataFrame()
    else:
        df_up = pd.read_csv(upload_results_path, dtype=str)

    if not df_up.empty:
        for _, ur in df_up.iterrows():
            img_p_str = _normalize_val(ur.get("image_path"))
            if not img_p_str:
                continue
            enc = _normalize_val(ur.get("encounter_id"))
            if enc.startswith("upload-") or enc.startswith("upload_"):
                encounter_id = enc
            elif enc:
                encounter_id = f"upload-{enc}"
            else:
                encounter_id = f"upload-{Path(img_p_str).parent.name}"

            pid = Path(img_p_str).stem

            lat_val = _normalize_val(ur.get("latitude"))
            lon_val = _normalize_val(ur.get("longitude"))
            loc_src = _normalize_val(ur.get("location_source"))
            if not loc_src and lat_val and lon_val:
                loc_src = "upload_exif"
            elif not (lat_val and lon_val):
                loc_src = ""

            rows.append({
                "source": SOURCE_NAME,
                "encounter_id": encounter_id,
                "photo_id": pid,
                "image_path": img_p_str,
                "image_url": "",
                "page_url": "",
                "observed_at": _normalize_val(ur.get("captured_at")),
                "observer": _normalize_val(ur.get("owner")),
                "latitude": lat_val,
                "longitude": lon_val,
                "location_source": loc_src,
                "locality": "",
                "license": "",
                "quality_grade": "",
            })

    return rows


def update_results_csv(repo_root: Path | None = None) -> Path:
    """Update upload records in data/results.csv while preserving other sources."""
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
    print(f"Upload: translated {len(records)} records.")

