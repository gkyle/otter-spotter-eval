# Otter annotation detector

This package trains one object detector with three classes:

- `whole_body`
- `throat`
- `throat_portrait` — forward-facing with most of the throat visible

The annotations are bounding boxes. Boxes may overlap; a throat is normally nested inside its otter's whole-body box. `throat_portrait` is a subset of `throat` used 
for forward-facing view.

## Build the dataset

Convert `data/labels.csv`:

```bash
uv run src/detect/build_dataset.py
```

The converter validates every image and box, then creates deterministic
80/10/10 train/validation/test splits. All images from one `observation_id`
remain in the same split to avoid encounter leakage. Images are symlinked by
default, so the generated dataset does not duplicate the originals. Use
`--copy` for a self-contained dataset and `--force` to replace an old build.

Only labeled images are included. Do not add downloaded-but-unreviewed images
as empty/negative examples: an unlabeled otter in one of those images would
teach the detector that an otter is background.

The labeling tool's **Negative** action records rejected generated boxes in
`data/labels_negative.csv`. During dataset generation, each rejected region
is exported as a separate crop with an empty label. This supplies a safe
background example without incorrectly treating the rest of an image—which may
contain real otters—as background. Negative crops remain in the same
encounter-level split as every other image derived from their observation.

## Train and evaluate

The reproducible training entry point rebuilds the dataset and spells out all
parameters used for the baseline run:

```bash
scripts/train_detector.sh
```

The equivalent direct commands are:

```bash
uv run src/detect/train.py

uv run src/detect/evaluate.py \
  data/models/otter-detector/yolo26n-640/weights/best.pt
```

Training starts with the pretrained YOLO26 nano weights.

## Predict

Return JSON boxes for an uploaded image:

```bash
uv run src/detect/predict.py \
  data/models/otter-detector/yolo26n-640/weights/best.pt image.jpg
```
