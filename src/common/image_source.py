"""Resolve otter source photos from the local image collection."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.manifest import REPO_ROOT  # noqa: E402

def resolve_image_source(image_path: str) -> str:
    """Return the local filesystem path for a repo-relative image path."""
    return str(REPO_ROOT / image_path)


def image_available(image_path: str) -> bool:
    """Whether the image exists on local disk."""
    return Path(resolve_image_source(image_path)).is_file()


def open_image(image_path: str) -> Image.Image:
    """Open a local source photo, ensuring upright orientation."""
    return ImageOps.exif_transpose(Image.open(resolve_image_source(image_path)))
