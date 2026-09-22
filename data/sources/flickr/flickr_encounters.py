"""Stable compact encounter identifiers for Flickr observations."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

SHORT_ID = re.compile(r"^flickr_[0-9a-f]{8}$")


def short_flickr_encounter(value: str) -> str:
    """Map a canonical Flickr encounter description to a compact stable ID."""
    if SHORT_ID.fullmatch(value):
        return value
    if value.startswith("fl-") and len(value) == 11:
        return f"flickr_{value[3:]}"
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=4).hexdigest()
    return f"flickr_{digest}"


def flickr_encounter(owner: str, date_taken: str, photo_id: str) -> str:
    """Group one owner's photos taken during the same date and hour."""
    safe_owner = re.sub(r"[^A-Za-z0-9@_-]+", "_", owner.strip()).strip("_")
    try:
        taken = datetime.fromisoformat(date_taken.strip())
    except (TypeError, ValueError):
        taken = None
    if safe_owner and taken is not None:
        return short_flickr_encounter(f"flickr-{safe_owner}-{taken:%Y%m%d-%H}")
    return short_flickr_encounter(f"flickr-photo-{photo_id}")
