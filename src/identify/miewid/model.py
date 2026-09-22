#!/usr/bin/env python3
"""MiewID-msv3 loading and embedding helpers.

Centralizes the model tag, preprocessing, and batched inference so the
extraction, evaluation, and visualization steps stay consistent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModel

MODEL_TAG = "conservationxlabs/miewid-msv3"
IMAGE_SIZE = 440
EMBEDDING_DIM = 2152

# Same preprocessing the model card specifies (ImageNet stats, 440x440).
preprocess = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_device(prefer_gpu: bool = True) -> torch.device:
    return torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")


def load_model(
    device: torch.device,
    checkpoint: str | Path | None = None,
) -> AutoModel:
    """Load MiewID-msv3 in eval mode on ``device``.

    Loading emits benign "meta parameter ... no-op" warnings from the timm
    backbone init; the msv3 checkpoint weights are still applied correctly.
    """
    model = AutoModel.from_pretrained(MODEL_TAG, trust_remote_code=True)
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("model", payload)
        model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def embed_images(
    model: AutoModel,
    images: list[Image.Image],
    device: torch.device,
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Return an (N, EMBEDDING_DIM) float32 array of embeddings for ``images``."""
    chunks: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        tensor = torch.stack([preprocess(img.convert("RGB")) for img in batch])
        output = model(tensor.to(device))
        chunks.append(output.float().cpu().numpy())

    if not chunks:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)
    return embeddings
