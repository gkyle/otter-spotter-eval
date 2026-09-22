#!/usr/bin/env python3
"""Phase 2: crop labeled bounding boxes out of the original pool images.

Reads the annotation manifest and writes one crop per bounding box to
``data/crops/<body_part>/<annotation_id>.jpg``, keeping each crop target in a
separate folder. Also writes ``data/crops/crops.csv`` -- a self-contained index
pairing every crop with its labels -- which the Phase 3 embedding step consumes.

Crops are saved at native resolution (the model's preprocessing resizes to
440x440 at inference time); this avoids baking upscaling artifacts into the
cache and keeps crops useful for visual inspection.

Usage:
    uv run src/identify/crop.py
    uv run src/identify/crop.py --body-part throat --margin 0.1 --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.manifest import (  # noqa: E402
    BODY_PARTS,
    DATA_DIR,
    MANIFEST_PATH,
    REPO_ROOT,
    filter_manifest_by_license,
    load_manifest,
)

CROPS_DIR = DATA_DIR / "crops"
CROPS_INDEX = CROPS_DIR / "crops.csv"
JPEG_QUALITY = 95


def crop_with_margin(
    image: Image.Image, x: int, y: int, w: int, h: int, margin: float
) -> Image.Image:
    """Crop the box, optionally padded by ``margin`` (fraction of side), clamped."""
    img_w, img_h = image.size
    mx = int(round(w * margin))
    my = int(round(h * margin))
    left = max(0, x - mx)
    top = max(0, y - my)
    right = min(img_w, x + w + mx)
    bottom = min(img_h, y + h + my)
    return image.crop((left, top, right, bottom))


def clear_crops(output_dir: Path, body_part: str | None = None) -> int:
    """Remove managed crop folders in the requested regeneration scope."""
    parts = [body_part] if body_part else BODY_PARTS
    removed = 0
    for part in parts:
        part_dir = output_dir / part
        if part_dir.exists():
            removed += sum(path.is_file() for path in part_dir.rglob("*"))
            shutil.rmtree(part_dir)
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                        help="annotation labels (default: data/labels.csv)")
    parser.add_argument("--output-dir", type=Path, default=CROPS_DIR,
                        help="crops output directory (default: data/crops)")
    parser.add_argument("--body-part", choices=BODY_PARTS,
                        help="only crop this target (default: all)")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="pad each box by this fraction of its size (default: 0)")
    parser.add_argument(
        "--filter-by-license",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="filter manifest to retain only open-licensed images (default: True)",
    )
    parser.add_argument("--force", action="store_true",
                        help="delete existing crops in scope, then regenerate them")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.filter_by_license:
        orig_len = len(manifest)
        manifest = filter_manifest_by_license(manifest)
        print(f"Filtered crops manifest by license: {len(manifest)}/{orig_len} annotations retained.")
    if args.body_part:
        manifest = manifest[manifest["body_part"] == args.body_part]
    crops_index = args.output_dir / CROPS_INDEX.name

    if args.force:
        removed = clear_crops(args.output_dir, args.body_part)
        print(f"Force cleanup: removed {removed} existing crop image(s).")

    if manifest.empty:
        # Do not leave a stale index pointing at crops removed by --force.
        if args.force:
            empty_index = manifest.copy()
            empty_index["crop_path"] = pd.Series(dtype=str)
            crops_index.parent.mkdir(parents=True, exist_ok=True)
            empty_index.to_csv(crops_index, index=False)
        print("No annotations to crop.")
        return 0

    created = skipped = failed = 0
    index_rows: list[dict] = []
    # Group by source image so each original is opened only once.
    for image_path, group in manifest.groupby("image_path"):
        source = REPO_ROOT / str(image_path)
        image = None
        for _, row in group.iterrows():
            body_part = str(row["body_part"])
            annotation_id = str(row["annotation_id"])
            crop_path = args.output_dir / body_part / f"{annotation_id}.jpg"

            if crop_path.exists() and not args.force:
                skipped += 1
            else:
                try:
                    if image is None:
                        with Image.open(source) as source_image:
                            image = source_image.convert("RGB")
                    x, y, w, h = (int(row["bbox_x"]), int(row["bbox_y"]),
                                  int(row["bbox_w"]), int(row["bbox_h"]))
                    if w <= 0 or h <= 0:
                        raise ValueError(f"degenerate bbox {(x, y, w, h)}")
                    crop = crop_with_margin(image, x, y, w, h, args.margin)
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(crop_path, "JPEG", quality=JPEG_QUALITY)
                    created += 1
                except (OSError, ValueError) as exc:
                    failed += 1
                    print(f"  failed {annotation_id} ({image_path}): {exc}",
                          file=sys.stderr)
                    continue

            record = row.to_dict()
            record["crop_path"] = str(crop_path.resolve().relative_to(REPO_ROOT))
            index_rows.append(record)
        if image is not None:
            image.close()

    index = pd.DataFrame(index_rows)
    crops_index.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(crops_index, index=False)

    print(f"Done: {created} cropped, {skipped} already present, {failed} failed.")
    if not index.empty:
        per_part = index.groupby("body_part").size().to_dict()
        try:
            display_index = crops_index.resolve().relative_to(REPO_ROOT)
        except ValueError:
            display_index = crops_index
        print(f"Index: {len(index)} crops -> {display_index} "
              f"({per_part})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
