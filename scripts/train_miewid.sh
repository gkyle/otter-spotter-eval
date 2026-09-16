#!/usr/bin/env bash
set -euo pipefail

# If cropped images do not exist, generate them.
if [ ! -f data/crops/crops.csv ]; then
    uv run src/identify/crop.py
fi

uv run src/identify/miewid/train.py \
  --body-part throat_portrait \
  --epochs 25 \
  --freeze-epochs 2 \
  --batch-size 8 \
  --no-filter-by-license \
  "$@"
