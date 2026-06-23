"""Empirical Risk Minimization (ERM) training loop.

WHAT THIS IS
============
ERM = train by minimizing the *average* cross-entropy loss over all
training examples, with every example contributing equally to the
gradient. No group weighting, no re-sampling, no special objectives.

In code, this is just:
    loss = CrossEntropy(logits, labels)
    loss.backward()
    optimizer.step()

There is nothing exotic here — and that is the point. ERM is the
*default* in deep learning, and on biased data it is exactly what
produces the worst-group failure we want to demonstrate.

WHAT WE LOG
===========
We track BOTH average validation accuracy AND worst-group validation
accuracy every epoch. We select the best checkpoint using
**worst-group val accuracy** (not average), because:
    - Selecting on average accuracy rewards shortcut learning.
    - The WILDS protocol requires group-robust model selection.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from erm.config import (
    LEARNING_RATE,
    MODEL_DIR,
    NUM_EPOCHS,
    SEED,
    WEIGHT_DECAY,
)
from erm.data import (
    accuracy_per_group,
    collect_group_predictions,
    load_waterbirds,
    make_loader,
)
from erm.model import build_model


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Train log container
# ----------------------------------------------------------------------
@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_avg_acc: float
    val_worst_group_acc: float
    val_per_group: dict[str, float]


# ----------------------------------------------------------------------
# Main training routine
# ----------------------------------------------------------------------
def train_erm(device: str = "cuda") -> dict:
    """Train a ResNet-18 with ERM on Waterbirds.

    Returns a dict with training history and the path to the best
    checkpoint (selected by worst-group val accuracy).
    """
    set_seed(SEED)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Data
    train_subset = load_waterbirds("train")
    val_subset = load_waterbirds("val")
    train_loader = make_loader(train_subset, shuffle=True)
    val_loader = make_loader(val_subset, shuffle=False)
    print(f"[data] train={len(train_subset)}  val={len(val_subset)}")

    # Model, loss, optimizer
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()  # <-- THIS is ERM. Plain mean loss.
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history: list[EpochLog] = []
    best_worst = -1.0
    best_path = MODEL_DIR / "erm_best.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        # ------------------ TRAIN ------------------
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch:02d} train"):
            x, y, _ = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)  # ERM: equal weight per example
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_train_loss = running_loss / max(n_batches, 1)

        # ------------------ EVAL ------------------
        buckets = collect_group_predictions(model, val_loader, device)
        per_group = accuracy_per_group(buckets)
        valid = [v for v in per_group.values() if not np.isnan(v)]
        val_avg = float(np.mean(valid))
        val_worst = float(min(valid))

        log = EpochLog(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_avg_acc=val_avg,
            val_worst_group_acc=val_worst,
            val_per_group=per_group,
        )
        history.append(log)

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_avg={val_avg:.4f}  "
            f"val_worst_group={val_worst:.4f}"
        )

        # ------------------ CHECKPOINT (best by worst-group) ------------------
        if val_worst > best_worst:
            best_worst = val_worst
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_worst_group_acc": val_worst,
                    "val_avg_acc": val_avg,
                    "val_per_group": per_group,
                },
                best_path,
            )

    # Persist training history
    history_path = MODEL_DIR / "erm_history.json"
    history_path.write_text(json.dumps([log.__dict__ for log in history], indent=2))
    print(f"[done] best worst-group val acc = {best_worst:.4f}")
    print(f"[done] checkpoint -> {best_path}")
    print(f"[done] history   -> {history_path}")

    return {
        "best_checkpoint": str(best_path),
        "best_worst_group_val_acc": best_worst,
        "history_path": str(history_path),
    }


if __name__ == "__main__":
    train_erm()
