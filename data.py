"""Waterbirds dataset loading + 4-subgroup bookkeeping.

Why this matters:
    The whole point of the project is that ERM hides its failure on the
    minority subgroups. We therefore need to *see* the four subgroups
    (landbird-on-land, landbird-on-water, waterbird-on-land,
    waterbird-on-water) at every step: in dataloaders, in evaluation,
    and in saliency visualization.

WILDS conventions for Waterbirds:
    dataset[i]  -> (image_tensor, y, metadata_array)
    metadata[:, 0]  = background (0=land, 1=water)
    metadata[:, 1]  = bird label (0=landbird, 1=waterbird)

A "group" is the pair (y, background), giving 4 groups indexed 0..3.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from wilds import get_dataset

from erm.config import (
    BATCH_SIZE,
    DATA_ROOT,
    GROUP_NAMES,
    NUM_GROUPS,
    NUM_WORKERS,
)


# ----------------------------------------------------------------------
# Subgroup utilities
# ----------------------------------------------------------------------
def group_id_from_metadata(metadata_row: np.ndarray) -> int:
    """Convert (background, bird_type) into a group index 0..3.

    Group encoding (matches GROUP_NAMES in config):
        0: landbird (0)  on land (0)
        1: landbird (0)  on water (1)
        2: waterbird (1) on land (0)
        3: waterbird (1) on water (1)
    """
    background = int(metadata_row[0])
    bird = int(metadata_row[1])
    return bird * 2 + background


def group_counts(dataset) -> dict[str, int]:
    """Count how many examples each group has in a WILDS split."""
    counts = {name: 0 for name in GROUP_NAMES}
    for i in range(len(dataset)):
        _, _, metadata = dataset[i]
        gid = group_id_from_metadata(metadata)
        counts[GROUP_NAMES[gid]] += 1
    return counts


# ----------------------------------------------------------------------
# Transforms (ImageNet stats — required for pretrained ResNet)
# ----------------------------------------------------------------------
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_waterbirds(split: str):
    """Return a WILDS Waterbirds dataset for the given split.

    split in {"train", "val", "test"}.
    """
    dataset = get_dataset(dataset="waterbirds", root_dir=str(DATA_ROOT), download=True)
    return dataset.get_subset(split, transform=(train_transform() if split == "train"
                                                else eval_transform()))


def make_loader(subset, shuffle: bool) -> DataLoader:
    """Wrap a WILDS subset in a standard PyTorch DataLoader."""
    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )


# ----------------------------------------------------------------------
# Evaluation helper: build (group -> list of (pred, label))
# ----------------------------------------------------------------------
def collect_group_predictions(model, loader, device) -> dict[int, list[tuple[int, int]]]:
    """Run `model` over `loader` and bucket predictions by subgroup.

    Returns a dict mapping group_id -> list of (pred, true_label) tuples.
    """
    model.eval()
    buckets: dict[int, list[tuple[int, int]]] = {g: [] for g in range(NUM_GROUPS)}

    with torch.no_grad():
        for batch in loader:
            # WILDS batches may be (x, y, metadata) or (x, y) depending on version.
            if len(batch) == 3:
                x, y, metadata = batch
            else:
                x, y = batch
                metadata = None

            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_np = y.cpu().numpy()

            if metadata is None:
                # Fall back: assign everything to a single group if metadata missing.
                # This should NOT happen with WILDS Waterbirds.
                for p, t in zip(preds, y_np):
                    buckets[0].append((int(p), int(t)))
            else:
                meta_np = metadata.numpy()
                for j in range(len(preds)):
                    gid = group_id_from_metadata(meta_np[j])
                    buckets[gid].append((int(preds[j]), int(y_np[j])))

    return buckets


def accuracy_per_group(buckets) -> dict[str, float]:
    """Convert prediction buckets into per-group accuracy dict."""
    out = {}
    for gid, items in buckets.items():
        if len(items) == 0:
            out[GROUP_NAMES[gid]] = float("nan")
        else:
            correct = sum(1 for p, t in items if p == t)
            out[GROUP_NAMES[gid]] = correct / len(items)
    return out
