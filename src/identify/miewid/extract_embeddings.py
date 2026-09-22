#!/usr/bin/env python3
"""Phase 3: extract MiewID embeddings for every labeled crop.

Reads the Phase 2 crop index (``data/crops/crops.csv``) and, for each crop
target (``whole_body``, ``throat``, ``throat_portrait``), writes an L2-normalized
embedding cache:

    data/embeddings/miewid-msv3/<body_part>/embeddings.npy  # (N, 2152)
    data/embeddings/miewid-msv3/<body_part>/index.csv
    data/embeddings/miewid-msv3/<body_part>/locations.csv

Extraction is incremental: crops already present in the cache are reused, only
new crops are run through the model, and crops removed from the manifest are
dropped. Use ``--force`` to recompute everything.

Usage:
    uv run src/identify/miewid/extract_embeddings.py
    uv run src/identify/miewid/extract_embeddings.py --body-part throat --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy  # noqa: F401
import sklearn  # noqa: F401
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import BODY_PARTS, DATA_DIR, REPO_ROOT  # noqa: E402
from location.geodistance import write_location_cache  # noqa: E402
from model import EMBEDDING_DIM, embed_images, get_device, load_model  # noqa: E402

CROPS_INDEX = DATA_DIR / "crops" / "crops.csv"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "miewid-msv3-finetuned"
INDEX_COLUMNS = [
    "annotation_id",
    "individual_id",
    "observation_id",
    "latitude",
    "longitude",
    "location_source",
    "body_part",
    "crop_path",
    "split",
]


def load_cache(out_dir: Path) -> dict[str, np.ndarray]:
    """Map annotation_id -> cached embedding vector, if a valid cache exists."""
    index_path = out_dir / "index.csv"
    npy_path = out_dir / "embeddings.npy"
    if not (index_path.exists() and npy_path.exists()):
        return {}
    index = pd.read_csv(index_path, dtype={"annotation_id": str})
    embeddings = np.load(npy_path)
    if len(index) != len(embeddings):
        return {}  # cache is inconsistent; recompute from scratch
    return {aid: embeddings[i] for i, aid in enumerate(index["annotation_id"].astype(str))}


def process_body_part(
    body_part: str,
    crops: pd.DataFrame,
    model,
    device,
    out_dir: Path,
    batch_size: int,
    force: bool,
) -> tuple[int, int]:
    """Embed all crops for one body part; returns (newly_computed, total)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = crops.reset_index(drop=True)
    cache = {} if force else load_cache(out_dir)

    pending = [row for _, row in crops.iterrows()
               if str(row["annotation_id"]) not in cache]

    computed = 0
    for start in range(0, len(pending), batch_size):
        images: list[Image.Image] = []
        valid: list[pd.Series] = []
        for row in pending[start : start + batch_size]:
            try:
                with Image.open(REPO_ROOT / str(row["crop_path"])) as source:
                    images.append(source.convert("RGB"))
                valid.append(row)
            except OSError as exc:
                print(f"  skip {row['annotation_id']}: {exc}", file=sys.stderr)
        try:
            new_embeddings = embed_images(
                model, images, device, batch_size=batch_size
            )
            for row, vector in zip(valid, new_embeddings):
                cache[str(row["annotation_id"])] = vector
                computed += 1
        finally:
            for image in images:
                image.close()

    # Assemble output in crop-index order, keeping only crops still present.
    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    for _, row in crops.iterrows():
        aid = str(row["annotation_id"])
        if aid in cache:
            rows.append({col: row.get(col, "") for col in INDEX_COLUMNS})
            vectors.append(cache[aid])

    embeddings = (
        np.vstack(vectors) if vectors else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    )
    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
    output_index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    output_index.to_csv(out_dir / "index.csv", index=False)
    return computed, len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops-index", type=Path, default=CROPS_INDEX,
                        help="Phase 2 crop index (default: data/crops/crops.csv)")
    parser.add_argument("--output-dir", type=Path, default=EMBEDDINGS_DIR,
                        help="embedding output root (default: data/embeddings/miewid-msv3-finetuned)")
    parser.add_argument("--body-part", choices=BODY_PARTS,
                        help="only extract this target (default: all present)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    parser.add_argument("--force", action="store_true",
                        help="recompute all embeddings, ignoring the cache")
    parser.add_argument("--checkpoint", type=Path,
                        help="fine-tuned model checkpoint produced by train.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.crops_index.exists():
        print(f"error: {args.crops_index} not found; run src/identify/crop.py first.",
              file=sys.stderr)
        return 2

    crops = pd.read_csv(args.crops_index, dtype={"annotation_id": str,
                                                 "individual_id": str,
                                                 "observation_id": str})
    if crops.empty:
        print("No crops to embed.")
        return 0

    parts = [args.body_part] if args.body_part else sorted(crops["body_part"].unique())
    if args.force:
        # Clear every selected cache up front. If native image/CUDA code later
        # terminates the process, a non-force retry can safely reuse only body
        # parts that this run actually completed; untouched stale parts cannot
        # be mistaken for fresh results.
        for body_part in parts:
            part_dir = args.output_dir / body_part
            for filename in ("embeddings.npy", "index.csv"):
                path = part_dir / filename
                if path.exists():
                    path.unlink()
    device = get_device(prefer_gpu=not args.cpu)
    print(f"device: {device}")
    print("loading MiewID-msv3 ...")
    model = load_model(device, args.checkpoint)

    for body_part in parts:
        subset = crops[crops["body_part"] == body_part]
        if subset.empty:
            continue
        out_dir = args.output_dir / body_part
        computed, total = process_body_part(
            body_part, subset, model, device, out_dir, args.batch_size, False
        )
        try:
            available, missing = write_location_cache(
                pd.read_csv(out_dir / "index.csv", dtype=str), out_dir
            )
        except ValueError as exc:
            print(f"error loading locations: {exc}", file=sys.stderr)
            return 2
        resolved_out_dir = out_dir.resolve()
        try:
            display_out_dir = resolved_out_dir.relative_to(REPO_ROOT)
        except ValueError:
            display_out_dir = resolved_out_dir
        print(f"{body_part:11s}: {total} embeddings "
              f"({computed} new) -> {display_out_dir}")
        print(f"{'':11s}  locations: {available} available, {missing} missing")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
