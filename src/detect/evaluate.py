#!/usr/bin/env python3
"""Evaluate an otter detector on the held-out encounter test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/detector/dataset.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    return parser.parse_args()


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    metrics = YOLO(str(args.model)).val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
    )
    names = metrics.names
    report = {
        "split": args.split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {
            names[index]: {
                "precision": float(metrics.box.p[index]),
                "recall": float(metrics.box.r[index]),
                "map50": float(metrics.box.ap50[index]),
                "map50_95": float(metrics.box.maps[index]),
            }
            for index in range(len(metrics.box.maps))
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
