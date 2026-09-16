#!/usr/bin/env python3
"""Convert the annotation manifest into an encounter-safe YOLO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import ExifTags, Image, ImageOps

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.manifest import filter_manifest_by_license  # noqa: E402

CLASSES = {"whole_body": 0, "throat": 1, "throat_portrait": 2}
MIN_BOX_PX = 4
REQUIRED_COLUMNS = {
    "image_path",
    "observation_id",
    "body_part",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "img_width",
    "img_height",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/labels.csv"))
    parser.add_argument(
        "--negatives",
        type=Path,
        default=Path("data/labels_negative.csv"),
        help="Rejected generated boxes to export as empty-label hard-negative crops.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/detector"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--filter-by-license",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter training images to retain only those with open licenses (default: True).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy source images instead of making relative symbolic links.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing generated dataset.",
    )
    return parser.parse_args()


def validate_manifest(frame: pd.DataFrame, repo_root: Path | None = None) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    unknown = set(frame["body_part"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown body_part values: {sorted(unknown)}")
    if frame[list(REQUIRED_COLUMNS - {"image_path", "body_part"})].isna().any().any():
        raise ValueError("Required annotation fields contain empty values")

    bad_boxes = (
        (frame["bbox_w"] <= 0)
        | (frame["bbox_h"] <= 0)
        | (frame["bbox_x"] < 0)
        | (frame["bbox_y"] < 0)
        | (frame["bbox_x"] + frame["bbox_w"] > frame["img_width"] + 1)
        | (frame["bbox_y"] + frame["bbox_h"] > frame["img_height"] + 1)
    )
    if bad_boxes.any():
        raise ValueError(f"{int(bad_boxes.sum())} boxes are invalid or outside their image")


def filter_missing_images(
    frame: pd.DataFrame, repo_root: Path, description: str = ""
) -> pd.DataFrame:
    """Filter out rows where the referenced image file does not exist on disk."""
    existing_paths = {
        path
        for path in frame["image_path"].dropna().unique()
        if (repo_root / path).is_file()
    }
    missing_mask = ~frame["image_path"].isin(existing_paths)
    missing_count = int(missing_mask.sum())
    if missing_count > 0:
        missing_images = frame["image_path"].nunique() - len(existing_paths)
        prefix = f"{description} " if description else ""
        print(
            f"Ignored {missing_count} {prefix}labels for "
            f"{missing_images} missing images."
        )
        return frame[~missing_mask].copy()
    return frame


def split_observations(
    image_frame: pd.DataFrame,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, str]:
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("Train and validation fractions must be between zero and one")
    if train_fraction + val_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one")

    counts = (
        image_frame.groupby("observation_id", sort=False)["image_path"]
        .nunique()
        .to_dict()
    )
    observations = list(counts)
    random.Random(seed).shuffle(observations)
    # Place large encounters first, with the seed resolving equal-size ties.
    observations.sort(key=lambda observation: counts[observation], reverse=True)

    total = sum(counts.values())
    targets = {
        "train": total * train_fraction,
        "val": total * val_fraction,
        "test": total * (1 - train_fraction - val_fraction),
    }
    assigned = {split: 0 for split in targets}
    result: dict[str, str] = {}
    for observation in observations:
        split = max(
            targets,
            key=lambda name: (targets[name] - assigned[name]) / max(targets[name], 1),
        )
        result[str(observation)] = split
        assigned[split] += counts[observation]
    return result


def generated_name(image_path: str) -> str:
    source = Path(image_path)
    digest = hashlib.sha1(image_path.encode("utf-8")).hexdigest()[:12]
    return f"{source.stem}-{digest}{source.suffix.lower()}"


def link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def manifest_oriented_image(
    source: Path, declared_size: tuple[int, int]
) -> tuple[Image.Image, str | None]:
    """Load an image in manifest coordinates and describe any normalization."""
    with Image.open(source) as image:
        if image.size == declared_size:
            return image.copy(), None
        orientation = image.getexif().get(ExifTags.Base.Orientation)
        if orientation not in (None, 1):
            transposed = ImageOps.exif_transpose(image)
            if transposed.size != declared_size:
                raise ValueError(
                    f"Dimension mismatch for {source}: manifest={declared_size}, "
                    f"EXIF-oriented file={transposed.size}"
                )
            return transposed.copy(), f"EXIF orientation {orientation}"
        if image.size[::-1] == declared_size:
            # A known upload batch was rewritten without its orientation
            # metadata after being labeled. Its manifest coordinates match a
            # counterclockwise quarter-turn of the stored pixel matrix.
            return (
                image.transpose(Image.Transpose.ROTATE_90).copy(),
                "counterclockwise 90-degree dimension recovery",
            )
        raise ValueError(
            f"Dimension mismatch for {source}: manifest={declared_size}, "
            f"file={image.size}"
        )


def save_normalized_image(image: Image.Image, destination: Path) -> None:
    """Save an orientation-normalized dataset copy using the source suffix."""
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        image.convert("RGB").save(destination, quality=95)
    else:
        image.save(destination)


def yolo_line(row: pd.Series) -> str:
    width = float(row["img_width"])
    height = float(row["img_height"])
    center_x = (float(row["bbox_x"]) + float(row["bbox_w"]) / 2) / width
    center_y = (float(row["bbox_y"]) + float(row["bbox_h"]) / 2) / height
    box_width = float(row["bbox_w"]) / width
    box_height = float(row["bbox_h"]) / height
    return (
        f"{CLASSES[row['body_part']]} {center_x:.8f} {center_y:.8f} "
        f"{box_width:.8f} {box_height:.8f}"
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.exists() and (repo_root / "data" / "manifest.csv").exists():
        manifest_path = (repo_root / "data" / "manifest.csv").resolve()
    negatives_path = (repo_root / args.negatives).resolve()
    if not negatives_path.exists() and (repo_root / "data" / "detector_negatives.csv").exists():
        negatives_path = (repo_root / "data" / "detector_negatives.csv").resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    frame = pd.read_csv(manifest_path, dtype={"observation_id": str})
    validate_manifest(frame, repo_root)
    frame = filter_missing_images(frame, repo_root)

    if args.filter_by_license:
        orig_count = len(frame)
        frame = filter_manifest_by_license(frame, repo_root)
        print(f"Filtered manifest by license: {len(frame)}/{orig_count} annotations retained ({orig_count - len(frame)} restricted images excluded).")
    negative_columns = {
        "negative_id",
        "image_path",
        "observation_id",
        "body_part",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "img_width",
        "img_height",
    }
    if negatives_path.exists():
        negatives = pd.read_csv(negatives_path, dtype={"observation_id": str})
        missing_negative_columns = negative_columns - set(negatives)
        if missing_negative_columns:
            raise ValueError(
                "Detector negatives are missing columns: "
                f"{sorted(missing_negative_columns)}"
            )
        unknown_parts = set(negatives["body_part"]) - set(CLASSES)
        if unknown_parts:
            raise ValueError(
                f"Unknown negative body_part values: {sorted(unknown_parts)}"
            )
        negatives = filter_missing_images(negatives, repo_root, description="negative")
    else:
        negatives = pd.DataFrame(columns=sorted(negative_columns))

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} exists; pass --force to replace it")
        shutil.rmtree(output_dir)

    image_frame = frame.drop_duplicates("image_path").copy()
    split_items = image_frame[["image_path", "observation_id"]].copy()
    if not negatives.empty:
        negative_split_items = negatives[["negative_id", "observation_id"]].rename(
            columns={"negative_id": "image_path"}
        )
        negative_split_items["image_path"] = (
            "negative:" + negative_split_items["image_path"].astype(str)
        )
        split_items = pd.concat(
            [split_items, negative_split_items], ignore_index=True
        )
    observation_splits = split_observations(
        split_items, args.seed, args.train_fraction, args.val_fraction
    )
    image_frame["split"] = image_frame["observation_id"].map(observation_splits)

    index_rows: list[dict[str, object]] = []
    for image_path, annotations in frame.groupby("image_path", sort=True):
        observation_id = str(annotations.iloc[0]["observation_id"])
        split = observation_splits[observation_id]
        name = generated_name(image_path)
        source = (repo_root / image_path).resolve()
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        declared_sizes = set(
            zip(annotations["img_width"].astype(int), annotations["img_height"].astype(int))
        )
        if len(declared_sizes) != 1:
            raise ValueError(
                f"Conflicting manifest dimensions for {image_path}: {declared_sizes}"
            )
        declared_size = next(iter(declared_sizes))
        image, orientation_adjustment = manifest_oriented_image(
            source, declared_size
        )

        dataset_image = image_dir / name
        try:
            if orientation_adjustment is None:
                link_or_copy(source, dataset_image, args.copy)
            else:
                save_normalized_image(image, dataset_image)
                print(f"Normalized {image_path}: {orientation_adjustment}")
        finally:
            image.close()
        label_path = label_dir / f"{Path(name).stem}.txt"
        label_path.write_text(
            "\n".join(yolo_line(row) for _, row in annotations.iterrows()) + "\n",
            encoding="utf-8",
        )
        index_rows.append(
            {
                "source_image": image_path,
                "dataset_image": str((image_dir / name).relative_to(output_dir)),
                "observation_id": observation_id,
                "split": split,
                "annotation_count": len(annotations),
                "kind": "positive",
                "orientation_normalized": orientation_adjustment is not None,
            }
        )

    for negative in negatives.itertuples(index=False):
        split = observation_splits[str(negative.observation_id)]
        source = (repo_root / str(negative.image_path)).resolve()
        if not source.is_file():
            continue
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        name = f"negative-{negative.negative_id}.jpg"

        image, orientation_adjustment = manifest_oriented_image(
            source,
            (int(negative.img_width), int(negative.img_height)),
        )
        try:
            x1 = max(0, int(negative.bbox_x))
            y1 = max(0, int(negative.bbox_y))
            x2 = min(image.width, x1 + int(negative.bbox_w))
            y2 = min(image.height, y1 + int(negative.bbox_h))
            if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                raise ValueError(
                    f"Negative {negative.negative_id} has an invalid crop box"
                )
            crop = image.crop((x1, y1, x2, y2)).convert("RGB")
            crop.save(image_dir / name, format="JPEG", quality=95)
        finally:
            image.close()
        # An empty label explicitly marks the entire derived crop as background.
        (label_dir / f"{Path(name).stem}.txt").write_text("", encoding="utf-8")
        index_rows.append(
            {
                "source_image": str(negative.image_path),
                "dataset_image": str((image_dir / name).relative_to(output_dir)),
                "observation_id": str(negative.observation_id),
                "split": split,
                "annotation_count": 0,
                "kind": "hard_negative",
                "orientation_normalized": orientation_adjustment is not None,
            }
        )

    index = pd.DataFrame(index_rows)
    index.to_csv(output_dir / "index.csv", index=False)
    dataset_yaml = (
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: whole_body\n"
        "  1: throat\n"
        "  2: throat_portrait\n"
    )
    (output_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

    summary = {
        "seed": args.seed,
        "images": int(len(index)),
        "annotations": int(len(frame)),
        "negative_examples": int(len(negatives)),
        "orientation_normalized_images": int(
            index["orientation_normalized"].sum()
        ),
        "observations": int(frame["observation_id"].nunique()),
        "classes": frame["body_part"].value_counts().sort_index().to_dict(),
        "splits": {},
    }
    for split, group in index.groupby("split"):
        positive_group = group[group["kind"] == "positive"]
        split_paths = set(positive_group["source_image"])
        split_annotations = frame[frame["image_path"].isin(split_paths)]
        summary["splits"][split] = {
            "images": int(len(group)),
            "positive_images": int(len(positive_group)),
            "negative_examples": int((group["kind"] == "hard_negative").sum()),
            "observations": int(group["observation_id"].nunique()),
            "annotations": int(len(split_annotations)),
            "classes": split_annotations["body_part"]
            .value_counts()
            .sort_index()
            .to_dict(),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_dir / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
