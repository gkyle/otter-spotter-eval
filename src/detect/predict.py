#!/usr/bin/env python3
"""Run the otter detector and emit app-friendly JSON bounding boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/detector/predictions"))
    return parser.parse_args()


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    kwargs = {
        "source": str(args.source),
        "imgsz": args.imgsz,
        "conf": args.confidence,
        "verbose": False,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    results = YOLO(str(args.model)).predict(**kwargs)
    output = []
    for result in results:
        boxes = result.boxes
        detections = []
        if boxes is not None:
            for xyxy, confidence, class_id in zip(
                boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.cls.int().cpu().tolist(),
            ):
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": confidence,
                        "bbox_xyxy": {
                            "x1": xyxy[0],
                            "y1": xyxy[1],
                            "x2": xyxy[2],
                            "y2": xyxy[3],
                        },
                    }
                )
        output.append(
            {
                "image_path": str(result.path),
                "image_width": int(result.orig_shape[1]),
                "image_height": int(result.orig_shape[0]),
                "detections": detections,
            }
        )
        if args.save_preview:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            result.save(filename=str(args.output_dir / Path(result.path).name))
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
