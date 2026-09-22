#!/usr/bin/env python3
"""
wikimedia.py

Core library functions for searching Wikimedia Commons, fetching image metadata,
extracting licensing details, page URLs, location descriptions, and downloading images.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path(__file__).parent
DEFAULT_QUERY = '"giant river otter"'
USER_AGENT = "OtterSpotter/1.0 (https://github.com/gkyle/otter-spotter; admin@otterspotter.org)"

# Known locations dictionary for text mining
KNOWN_LOCATIONS = {
    "karanambu": "Karanambu, Guyana",
    "lake sandoval": "Lago Sandoval, Peru",
    "lago sandoval": "Lago Sandoval, Peru",
    "sandoval": "Lago Sandoval, Peru",
    "tambopata": "Reserva Nacional Tambopata, Peru",
    "pantanal": "Pantanal, Brazil",
    "mato grosso": "Mato Grosso, Brazil",
    "rio mutum": "Rio Mutum, Mato Grosso, Brazil",
    "mutum river": "Rio Mutum, Mato Grosso, Brazil",
    "encontro das águas": "Parque Estadual Encontro das Águas, Brazil",
    "encontro das aguas": "Parque Estadual Encontro das Águas, Brazil",
    "cantão": "Parque Estadual do Cantão, Tocantins, Brazil",
    "cantao": "Parque Estadual do Cantão, Tocantins, Brazil",
    "chester zoo": "Chester Zoo, UK",
    "cali zoo": "Zoológico de Cali, Colombia",
    "zoologico de cali": "Zoológico de Cali, Colombia",
    "los angeles zoo": "Los Angeles Zoo, CA",
    "philadelphia zoo": "Philadelphia Zoo, PA",
    "duisburg zoo": "Zoo Duisburg, Germany",
    "madre de dios": "Madre de Dios, Peru",
    "iquitos": "Iquitos, Peru",
    "puerto maldonado": "Puerto Maldonado, Peru",
    "leticia": "Leticia, Colombia",
    "amazonas": "Amazonas",
    "orinoco": "Orinoco River",
    "essequibo": "Essequibo River, Guyana",
    "rupununi": "Rupununi, Guyana",
    "iwokrama": "Iwokrama Forest, Guyana",
    "manu national park": "Parque Nacional Manu, Peru",
    "parque nacional manu": "Parque Nacional Manu, Peru",
    "manu": "Parque Nacional Manu, Peru",
    "reserva amazonica": "Reserva Amazonica, Peru",
    "dallas zoo": "Dallas Zoo, TX",
    "miami zoo": "Zoo Miami, FL",
}

STOP_WORDS = set([
    "giant", "river", "otter", "otters", "ariranha", "lontra", "pteronura",
    "brasiliensis", "photo", "image", "picture", "canon", "nikon", "sony",
    "totals", "taken", "wildlife", "nature", "animal", "animals", "mammal"
])


class HTMLTagStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text: list[str] = []

    def handle_data(self, d: str):
        self.text.append(d)

    def get_data(self) -> str:
        return " ".join(self.text)


def clean_html_text(raw_text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if not raw_text:
        return ""
    stripper = HTMLTagStripper()
    stripper.feed(str(raw_text))
    text_content = stripper.get_data()
    text_content = re.sub(r"[\r\n\t]+", " ", text_content)
    text_content = re.sub(r"\s+", " ", text_content)
    return text_content.strip()


def query_slug(query: str) -> str:
    """Convert search query string into a safe file slug."""
    clean = re.sub(r'[^a-zA-Z0-9]+', '-', query.strip().lower()).strip('-')
    return clean or "search"


def extract_location_clues(title: str, description: str) -> tuple[str, str]:
    """
    Mine candidate location clue and resolved location name from text.
    Returns (location_clue, resolved_location_name).
    """
    full_text = clean_html_text(f"{title}. {description}")
    full_text_lower = full_text.lower()

    # Direct match in known locations (sorted by key length descending)
    sorted_known = sorted(KNOWN_LOCATIONS.items(), key=lambda x: len(x[0]), reverse=True)
    for key, mapped_val in sorted_known:
        if key in full_text_lower:
            return (key.title(), mapped_val)

    # Pattern extraction for Parks, Lakes, Rivers, Zoos
    patterns = [
        (r'\b(?:Parque|Park|Reserva|Reserve)\s+(?:Estadual|Nacional|National)?\s*[A-ZÀ-ÿ][a-zÀ-ÿ]+(?:\s+[A-ZÀ-ÿ][a-zÀ-ÿ]+)?\b', 'Park/Reserve'),
        (r'\b(?:Lake|Lago|Laguna|Cocha)\s+[A-ZÀ-ÿ][a-zÀ-ÿ]+\b', 'Lake'),
        (r'\b(?:Rio|Río|River)\s+[A-ZÀ-ÿ][a-zÀ-ÿ]+\b', 'River'),
        (r'\b[A-ZÀ-ÿ][a-zÀ-ÿ]+\s+(?:Zoo|Tierpark|Park|Reserve)\b', 'Zoo/Park'),
    ]
    for pattern, _category in patterns:
        match = re.search(pattern, full_text)
        if match:
            candidate = match.group(0).strip()
            if not any(sw in candidate.lower() for sw in ["giant", "river", "otter"]):
                return (candidate, candidate)

    return ("", "")


def search_wikimedia(query: str, limit: int | None = None) -> list[dict[str, Any]]:
    """
    Search Wikimedia Commons via Action API (w/api.php).
    Retrieves search results, handling pagination.
    """
    results: list[dict[str, Any]] = []
    continue_params: dict[str, str] = {}
    
    while True:
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",  # File namespace
            "gsrlimit": "50",
            "prop": "imageinfo|info",
            "inprop": "url",
            "iiprop": "url|size|mime|timestamp|user|extmetadata",
            "iiurlwidth": "1280",
        }
        params.update(continue_params)
        
        url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"Error querying Wikimedia API: {exc}")
            break

        pages = data.get("query", {}).get("pages", {})
        for pageid, page in pages.items():
            results.append(page)
            if limit is not None and len(results) >= limit:
                return results[:limit]
                
        if "continue" in data:
            continue_params = data["continue"]
            time.sleep(0.3)  # Politeness delay
        else:
            break

    return results


def compact_wikimedia_photo(page: dict[str, Any]) -> dict[str, str]:
    """
    Extract and normalize standard metadata fields from a Wikimedia Commons API page object.
    """
    page_id = str(page.get("pageid") or "").strip()
    title = str(page.get("title") or "").strip()
    page_url = str(page.get("fullurl") or "").strip()
    
    if not page_url and title:
        safe_title = urllib.parse.quote(title.replace(" ", "_"), safe=":/")
        page_url = f"https://commons.wikimedia.org/wiki/{safe_title}"

    ii_list = page.get("imageinfo", [])
    ii = ii_list[0] if isinstance(ii_list, list) and ii_list else {}
    
    image_url = str(ii.get("thumburl") or ii.get("url") or "").strip().split("?")[0]
    full_image_url = str(ii.get("url") or "").strip().split("?")[0]
    
    width = str(ii.get("width") or "")
    height = str(ii.get("height") or "")
    size_bytes = str(ii.get("size") or "")
    mime = str(ii.get("mime") or "")
    user = str(ii.get("user") or "")
    timestamp = str(ii.get("timestamp") or "")

    meta = ii.get("extmetadata", {})
    if not isinstance(meta, dict):
        meta = {}

    desc_html = meta.get("ImageDescription", {}).get("value", "") if isinstance(meta.get("ImageDescription"), dict) else ""
    description = clean_html_text(desc_html)

    lic_name = ""
    if isinstance(meta.get("LicenseShortName"), dict):
        lic_name = str(meta["LicenseShortName"].get("value") or "")
    if not lic_name and isinstance(meta.get("License"), dict):
        lic_name = str(meta["License"].get("value") or "")

    lic_url = ""
    if isinstance(meta.get("LicenseUrl"), dict):
        lic_url = str(meta["LicenseUrl"].get("value") or "")

    artist_html = meta.get("Artist", {}).get("value", "") if isinstance(meta.get("Artist"), dict) else ""
    artist = clean_html_text(artist_html) or user

    lat = ""
    if isinstance(meta.get("GPSLatitude"), dict):
        lat = str(meta["GPSLatitude"].get("value") or "")

    lon = ""
    if isinstance(meta.get("GPSLongitude"), dict):
        lon = str(meta["GPSLongitude"].get("value") or "")

    location_clue, resolved_location = extract_location_clues(title, description)

    return {
        "id": page_id,
        "title": title,
        "page_url": page_url,
        "image_url": image_url or full_image_url,
        "full_image_url": full_image_url,
        "license": lic_name,
        "license_url": lic_url,
        "author": artist,
        "timestamp": timestamp,
        "width": width,
        "height": height,
        "size": size_bytes,
        "mime": mime,
        "description": description,
        "latitude": lat,
        "longitude": lon,
        "location_clue": location_clue,
        "resolved_location_name": resolved_location,
    }


def download_image(url: str, photo_id: str, output_dir: Path, delay: float = 0.8) -> tuple[Path, bool]:
    """
    Download image file atomically to output_dir/<photo_id>.<ext>.
    Returns (file_path, created_new_file).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clean_url = url.split("?")[0]
    
    # Extract extension from URL
    parsed_path = urllib.parse.urlparse(clean_url).path
    ext = Path(parsed_path).suffix.lower() or ".jpg"
    if ext not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
        ext = ".jpg"

    file_path = output_dir / f"{photo_id}{ext}"
    if file_path.exists() and file_path.stat().st_size > 0:
        return file_path, False

    # Retry loop with backoff for rate limiting (HTTP 429)
    max_retries = 4
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(clean_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                content = resp.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5.0
                print(f"Rate limited (429). Waiting {wait_time}s before retry...", flush=True)
                time.sleep(wait_time)
            else:
                raise

    with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    tmp_path.replace(file_path)
    if delay > 0:
        time.sleep(delay)
    return file_path, True


def existing_image(image_dir: Path, photo_id: str) -> Path | None:
    """Find existing downloaded image matching photo_id regardless of extension."""
    if not image_dir.is_dir():
        return None
    for path in image_dir.glob(f"{photo_id}.*"):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def atomic_json(path: Path, data: Any) -> None:
    """Write data as JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV file rows as dictionaries."""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    """Write CSV rows atomically."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", newline="", delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(tmp.name)

    tmp_path.replace(path)


SOURCE_NAME = "wikimedia"


def translate_to_results(repo_root: Path | None = None) -> list[dict[str, str]]:
    """Translate Wikimedia metadata into unified results rows."""
    import pandas as pd

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    source_dir = repo_root / "data" / "sources" / "wikimedia"
    wm_results_path = source_dir / "wikimedia_results.csv"
    wm_images_dir = source_dir / "images"

    def _normalize_val(val: object) -> str:
        if pd.isna(val) or val is None:
            return ""
        s = str(val).strip()
        return "" if s.lower() in ("nan", "none", "null") else s

    rows: list[dict[str, str]] = []
    if wm_results_path.is_file():
        df_wm = pd.read_csv(wm_results_path, dtype=str)
        for _, wr in df_wm.iterrows():
            pid = _normalize_val(wr.get("id"))
            if not pid:
                continue
            encounter_id = f"wikimedia_{pid}"
            img_rel = ""
            for ext in (".jpg", ".jpeg", ".png"):
                cand = wm_images_dir / f"{pid}{ext}"
                if cand.is_file():
                    img_rel = str(cand.relative_to(repo_root))
                    break

            loc_name = _normalize_val(wr.get("resolved_location_name")) or _normalize_val(wr.get("location_clue"))
            photo_url = _normalize_val(wr.get("full_image_url")) or _normalize_val(wr.get("image_url"))

            rows.append({
                "source": SOURCE_NAME,
                "encounter_id": encounter_id,
                "photo_id": pid,
                "image_path": img_rel,
                "image_url": photo_url,
                "page_url": _normalize_val(wr.get("page_url")),
                "observed_at": _normalize_val(wr.get("timestamp")),
                "observer": _normalize_val(wr.get("author")),
                "latitude": _normalize_val(wr.get("latitude")),
                "longitude": _normalize_val(wr.get("longitude")),
                "location_source": SOURCE_NAME,
                "locality": loc_name,
                "license": _normalize_val(wr.get("license")),
                "quality_grade": "",
            })

    return rows


def update_results_csv(repo_root: Path | None = None) -> Path:
    """Update wikimedia records in data/results.csv while preserving other sources."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    records = translate_to_results(repo_root)
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from common.sources import update_source_records
    return update_source_records(SOURCE_NAME, records, target_path=repo_root / "data" / "results.csv")


