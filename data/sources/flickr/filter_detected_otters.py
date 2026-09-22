#!/usr/bin/env python3
"""Detect otter crop classes in Flickr images and build a portrait contact sheet.

Writes one incremental CSV row per image with counts, maximum scores, and all
boxes. The contact sheet contains every image with a ``throat_portrait`` box,
sorted by its highest portrait score.

Usage:
    uv run python data/sources/flickr/filter_detected_otters.py --device 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = REPO_ROOT / "data/models/otter-detector/yolo26n-640-final/weights/best.pt"
DEFAULT_RESULTS = SOURCE_DIR / "flickr_results_filtered.csv"
CLASSES = ("whole_body", "throat", "throat_portrait")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
CSV_COLUMNS = [
    "id", "whole_body_count", "whole_body_max_score",
    "throat_count", "throat_max_score", "throat_portrait_count",
    "throat_portrait_max_score", "detections_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=SOURCE_DIR / "images")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=SOURCE_DIR / "detector")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--thumbnail-width", type=int, default=260)
    parser.add_argument("--thumbnail-height", type=int, default=210)
    parser.add_argument("--limit", type=int, help="process at most N images")
    parser.add_argument("--force", action="store_true", help="ignore cached rows")
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def load_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def result_row(result, image: Path) -> dict[str, Any]:
    found: dict[str, list[float]] = {name: [] for name in CLASSES}
    detections = []
    if result.boxes is not None:
        for xyxy, score, class_id in zip(
            result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
            result.boxes.cls.int().cpu().tolist(),
        ):
            name = str(result.names[class_id])
            detections.append({
                "class_id": int(class_id), "class_name": name,
                "confidence": float(score),
                "bbox_xyxy": [float(value) for value in xyxy],
            })
            if name in found:
                found[name].append(float(score))
    row: dict[str, Any] = {
        "id": image.stem,
        "detections_json": json.dumps(detections, separators=(",", ":")),
    }
    for name, scores in found.items():
        row[f"{name}_count"] = len(scores)
        row[f"{name}_max_score"] = max(scores, default="")
    return row


def error_row(image: Path, error: Exception) -> dict[str, Any]:
    row: dict[str, Any] = {column: "" for column in CSV_COLUMNS}
    row.update({
        "id": image.stem, "detections_json": "[]", "_error": str(error),
    })
    for name in CLASSES:
        row[f"{name}_count"] = 0
    return row


def predict(model, paths: list[Path], args: argparse.Namespace):
    kwargs: dict[str, Any] = {
        "source": [str(path) for path in paths], "imgsz": args.imgsz,
        "conf": args.confidence, "batch": args.batch, "stream": True,
        "verbose": False,
    }
    if args.device:
        kwargs["device"] = args.device
    return list(model.predict(**kwargs))


def predict_chunk(model, paths: list[Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Predict a batch, falling back to individual files if one is corrupt."""
    try:
        by_path = {Path(result.path).resolve(): result for result in predict(model, paths, args)}
        return [result_row(by_path[path.resolve()], path) for path in paths]
    except (OSError, RuntimeError, ValueError, KeyError) as batch_error:
        if len(paths) == 1:
            return [error_row(paths[0], batch_error)]
        rows = []
        for path in paths:
            try:
                rows.append(result_row(predict(model, [path], args)[0], path))
            except (OSError, RuntimeError, ValueError, IndexError) as exc:
                rows.append(error_row(path, exc))
        return rows


def decoded_detections(row: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return json.loads(str(row.get("detections_json", "[]")))
    except json.JSONDecodeError:
        return []


def portrait_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows
                if int(row.get("throat_portrait_count") or 0) > 0]
    return sorted(selected,
                  key=lambda row: float(row.get("throat_portrait_max_score") or 0),
                  reverse=True)


def make_composite(rows: list[dict[str, Any]], images: dict[str, Path],
                   output: Path, columns: int,
                   thumb_w: int, thumb_h: int) -> None:
    label_h, max_height = 38, 60_000
    cell_h = thumb_h + label_h
    if not rows:
        canvas = Image.new("RGB", (800, 160), "white")
        ImageDraw.Draw(canvas).text((20, 65),
                                    "No throat_portrait detections above threshold.",
                                    fill="black")
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, quality=92)
        return
    columns = max(columns, math.ceil(len(rows) * cell_h / max_height))
    canvas = Image.new("RGB", (columns * thumb_w,
                               math.ceil(len(rows) / columns) * cell_h), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for position, row in enumerate(rows):
        left, top = (position % columns) * thumb_w, (position // columns) * cell_h
        path = images.get(str(row["id"]))
        if path is None:
            draw.text((left + 5, top + 5), "missing", fill="black", font=font)
            continue
        try:
            with Image.open(path) as source:
                source = ImageOps.exif_transpose(source).convert("RGB")
                original_w, original_h = source.size
                thumbnail = ImageOps.contain(source, (thumb_w, thumb_h))
        except OSError:
            draw.text((left + 5, top + 5), "unreadable", fill="black", font=font)
            continue
        image_left = left + (thumb_w - thumbnail.width) // 2
        image_top = top + (thumb_h - thumbnail.height) // 2
        canvas.paste(thumbnail, (image_left, image_top))
        scale = min(thumb_w / original_w, thumb_h / original_h)
        for detection in decoded_detections(row):
            if detection.get("class_name") == "throat_portrait":
                x1, y1, x2, y2 = detection["bbox_xyxy"]
                draw.rectangle((image_left + x1 * scale, image_top + y1 * scale,
                                image_left + x2 * scale, image_top + y2 * scale),
                               outline="#39ff14", width=3)
        score = float(row["throat_portrait_max_score"])
        draw.text((left + 4, top + thumb_h + 10),
                  f"{path.stem}  throat_portrait={score:.3f}", fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, optimize=True)


def main() -> int:
    args = parse_args()
    if not args.images.is_dir() or not args.model.is_file():
        print("error: images directory or detector model not found", file=sys.stderr)
        return 2
    if not 0 < args.confidence <= 1 or args.batch < 1 or args.columns < 1:
        print("error: invalid confidence, batch, or columns", file=sys.stderr)
        return 2
    from ultralytics import YOLO

    paths = list_images(args.images)
    if args.limit is not None:
        paths = paths[:args.limit]
    csv_path = args.results
    cache = {} if args.force else load_cache(csv_path)
    rows_by_id, pending = {}, []
    for path in paths:
        cached = cache.get(path.stem)
        if cached:
            rows_by_id[path.stem] = cached
        else:
            pending.append(path)
    print(f"Images: {len(paths)} total, {len(paths)-len(pending)} cached, "
          f"{len(pending)} to process")
    model = YOLO(str(args.model)) if pending else None
    for start in range(0, len(pending), args.batch):
        chunk = pending[start:start + args.batch]
        for row in predict_chunk(model, chunk, args):
            rows_by_id[str(row["id"])] = row
        completed = [rows_by_id[path.stem] for path in paths
                     if path.stem in rows_by_id]
        atomic_csv(csv_path, completed)
        print(f"Processed {min(start+len(chunk), len(pending))}/{len(pending)}", flush=True)
    rows = [rows_by_id[path.stem] for path in paths]
    atomic_csv(csv_path, rows)
    portraits = portrait_rows(rows)
    composite = args.output_dir / "throat_portrait_composite.jpg"
    make_composite(portraits, {path.stem: path for path in paths}, composite, args.columns,
                   args.thumbnail_width, args.thumbnail_height)
    failures = sum(bool(row.get("_error")) for row in rows)
    print(f"Throat portrait candidates: {len(portraits)} -> {composite}")
    print(f"Detection rows: {len(rows)} -> {csv_path} ({failures} failed)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
