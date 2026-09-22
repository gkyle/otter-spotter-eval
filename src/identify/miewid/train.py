#!/usr/bin/env python3
"""Fine-tune MiewID-msv3 on encounter-grouped otter crops.

The classifier is used only as an ArcFace training objective. Checkpoints save
the embedding model separately, so they remain compatible with the existing
embedding and retrieval pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import DATA_DIR, REPO_ROOT, filter_manifest_by_license  # noqa: E402
from model import IMAGE_SIZE, MODEL_TAG, get_device, load_model  # noqa: E402

DEFAULT_CROPS = DATA_DIR / "crops" / "crops.csv"
DEFAULT_OUTPUT = DATA_DIR / "models" / "miewid-msv3-otters"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def encounter_split(frame: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    """Assign whole encounters to validation without orphaning an identity.

    A candidate encounter is selected only when every identity it contains has
    another encounter left for training. This prevents observation leakage and
    makes every validation query have a training-gallery match.
    """
    frame = frame.copy()
    by_id = frame.groupby("individual_id")["observation_id"].agg(lambda x: set(x))
    remaining = {key: set(value) for key, value in by_id.items()}
    by_obs = frame.groupby("observation_id")["individual_id"].agg(lambda x: set(x))
    candidates = list(by_obs.index)
    random.Random(seed).shuffle(candidates)
    target = round(frame["observation_id"].nunique() * val_fraction)
    selected: set[str] = set()
    for observation in candidates:
        identities = by_obs[observation]
        if all(len(remaining[individual]) > 1 for individual in identities):
            selected.add(observation)
            for individual in identities:
                remaining[individual].discard(observation)
            if len(selected) >= target:
                break
    frame["split"] = frame["observation_id"].map(
        lambda value: "val" if value in selected else "train"
    )
    return frame


class CropDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, labels: dict[str, int], augment: bool):
        self.frame = frame.reset_index(drop=True)
        steps: list[object] = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if augment:
            # Giant River Otter throat markings are asymmetric. A horizontal
            # flip invents a pattern that does not occur on the individual and
            # removes useful left/right identity information.
            steps += [
                transforms.RandomApply([transforms.ColorJitter(.2, .2, .15, .05)], p=.7),
                transforms.RandomAffine(8, translate=(.04, .04), scale=(.9, 1.1)),
            ]
        steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(steps)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(REPO_ROOT / row["crop_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[row["individual_id"]]


class ArcFace(nn.Module):
    def __init__(self, dimensions: int, classes: int, scale: float = 30, margin: float = .35):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, dimensions))
        nn.init.xavier_uniform_(self.weight)
        self.scale, self.margin = scale, margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cosine.gather(1, labels[:, None])) + self.margin)
        logits = cosine.scatter(1, labels[:, None], target)
        return logits * self.scale


@torch.no_grad()
def recall_at_one(model: nn.Module, gallery: DataLoader, queries: DataLoader, device: torch.device) -> float:
    model.eval()
    def collect(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        vectors, labels = [], []
        for images, target in loader:
            vectors.append(F.normalize(model(images.to(device))).cpu())
            labels.append(target)
        return torch.cat(vectors), torch.cat(labels)
    gallery_x, gallery_y = collect(gallery)
    query_x, query_y = collect(queries)
    if not len(query_y):
        return float("nan")
    matches = gallery_y[(query_x @ gallery_x.T).argmax(1)] == query_y
    return matches.float().mean().item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops-index", type=Path, default=DEFAULT_CROPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--body-part", choices=["whole_body", "throat", "throat_portrait", "all"], default="throat")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--filter-by-license",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="filter crops to retain only open-licensed images (default: True)",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate data and print the split without loading the model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    frame = pd.read_csv(args.crops_index, dtype=str).dropna(subset=["individual_id", "observation_id", "crop_path"])
    if args.filter_by_license:
        orig_count = len(frame)
        frame = filter_manifest_by_license(frame)
        print(f"Filtered training crops by license: {len(frame)}/{orig_count} crops retained.")
    if args.body_part != "all":
        frame = frame[frame.body_part == args.body_part]
    missing = [path for path in frame.crop_path if not (REPO_ROOT / path).is_file()]
    if frame.empty or missing:
        raise SystemExit("no usable crops" if frame.empty else f"{len(missing)} crop files are missing")
    frame = encounter_split(frame, args.val_fraction, args.seed)
    train = frame[frame.split == "train"].copy()
    val = frame[frame.split == "val"].copy()
    labels = {value: index for index, value in enumerate(sorted(train.individual_id.unique()))}
    val = val[val.individual_id.isin(labels)]
    overlap = set(train.observation_id) & set(val.observation_id)
    print(f"train: {len(train)} crops, {train.individual_id.nunique()} identities, {train.observation_id.nunique()} encounters")
    print(f"val:   {len(val)} crops, {val.individual_id.nunique()} identities, {val.observation_id.nunique()} encounters")
    print(f"encounter overlap: {len(overlap)}")
    if args.dry_run:
        return 0
    if val.empty:
        raise SystemExit("validation split is empty; choose a body part with multi-encounter identities")

    device = get_device(not args.cpu)
    if device.type != "cuda" and not args.cpu:
        raise SystemExit("CUDA is unavailable; pass --cpu explicitly (training will be slow)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "split.csv", index=False)
    (args.output_dir / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    model = load_model(device)
    dimensions = model.bn.num_features
    head = ArcFace(dimensions, len(labels)).to(device)
    train_loader = DataLoader(CropDataset(train, labels, True), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
    gallery_loader = DataLoader(CropDataset(train, labels, False), batch_size=args.batch_size, num_workers=args.workers)
    val_loader = DataLoader(CropDataset(val, labels, False), batch_size=args.batch_size, num_workers=args.workers)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.learning_rate},
        {"params": head.parameters(), "lr": args.head_learning_rate},
    ], weight_decay=args.weight_decay, foreach=False)
    best = -math.inf
    for epoch in range(args.epochs):
        frozen = epoch < args.freeze_epochs
        for parameter in model.backbone.parameters():
            parameter.requires_grad = not frozen
        model.train(); head.train(); total = 0.0
        for images, target in train_loader:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(model(images), target), target)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(target)
        score = recall_at_one(model, gallery_loader, val_loader, device)
        print(f"epoch {epoch + 1:02d}/{args.epochs}: loss={total / len(train):.4f} val_recall@1={score:.4f}")
        payload = {"model": model.state_dict(), "epoch": epoch + 1, "val_recall_at_1": score, "base_model": MODEL_TAG, "body_part": args.body_part}
        torch.save(payload, args.output_dir / "last.pt")
        if score > best:
            best = score
            torch.save(payload, args.output_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
