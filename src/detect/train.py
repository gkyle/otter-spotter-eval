#!/usr/bin/env python3
"""Fine-tune a YOLO detector for otter crop-target boxes."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/detector/dataset.yaml"))
    parser.add_argument("--model", default="data/models/yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=float, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--degrees",
        type=float,
        default=90.0,
        help="maximum random rotation in degrees (default: 90)",
    )
    parser.add_argument("--project", type=Path, default=Path("data/models/otter-detector"))
    parser.add_argument("--name", default="yolo26n-640")
    parser.add_argument(
        "--filter-by-license",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Indicate whether dataset was/should be filtered by open licenses (default: True).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    if args.resume:
        model = YOLO(args.model)
        model.train(resume=True)
        return 0

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        degrees=args.degrees,
        deterministic=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        plots=True,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
