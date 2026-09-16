#!/usr/bin/env bash
set -Eeuo pipefail

trap 'status=$?; echo "Detector training aborted at line ${LINENO} (exit ${status})." >&2; exit "${status}"' ERR

# Rebuild the encounter-safe YOLO dataset and fine-tune the otter detector.
# Extra arguments are appended to train.py, so they override values below:
#   scripts/train_detector.sh --epochs 150 --imgsz 960 --name yolo26n-960

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

uv run src/detect/build_dataset.py \
  --manifest data/labels.csv \
  --negatives data/labels_negative.csv \
  --output-dir data/detector \
  --seed 42 \
  --train-fraction 0.8 \
  --val-fraction 0.1 \
  --force \
  --no-filter-by-license \
  "$@"

uv run src/detect/train.py \
  --data data/detector/dataset.yaml \
  --model data/models/yolo26n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch -1 \
  --device 0 \
  --workers 4 \
  --patience 25 \
  --seed 42 \
  --project data/models/otter-detector \
  --no-filter-by-license \
  --name yolo26n-640 \
  "$@"

latest_dir="$(find data/models/otter-detector -maxdepth 1 -mindepth 1 -type d -name 'yolo26n-640*' | sort -V | tail -n 1)"
if [ -n "${latest_dir}" ]; then
  ln -sfn "$(basename "${latest_dir}")" data/models/otter-detector/yolo26n-640-final
  echo "Updated symlink data/models/otter-detector/yolo26n-640-final -> $(basename "${latest_dir}")"
fi
