#!/usr/bin/env python3
"""Export the trained detector for server, iOS, or Android inference."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--format",
        choices=("onnx", "coreml", "litert"),
        default="onnx",
        help="ONNX is portable; CoreML targets iOS; LiteRT targets Android.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include non-maximum suppression in the exported graph.",
    )
    return parser.parse_args()


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    exported = YOLO(str(args.model)).export(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        nms=args.nms,
    )
    print(exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
