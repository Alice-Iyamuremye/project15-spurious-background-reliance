"""Group DRO training loop."""
from __future__ import annotations

import json, random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from erm.config import LEARNING_RATE, MODEL_DIR, NUM_EPOCHS, NUM_GROUPS, SEED, WEIGHT_DECAY
from erm.data import (accuracy_per_group, collect_group_predictions, load_waterbirds, make_loader)
from erm.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class GroupDROLoss:
    """Distributionally Robust Loss with exponential weight updates."""
    def __init__(self, num_groups: int, robust_step_size: float = 0.01):
        self.num_groups = num_groups
        self.robust_step_size = robust_step_size
        self.q = torch.ones(num_groups) / num_groups

    def __call__(self, logits, labels, group_ids):
        device = logits.device
        if self.q.device != device:
            self.q = self.q.to(device)
        per_example_loss = F.cross_entropy(logits, labels, reduction="none")
        batch_loss = torch.zeros(self.num_groups, device=device)
        batch_count = torch.zeros(self.num_groups, device=device)
        for gid in range(self.num_groups):
            mask = (group_ids == gid)
            if mask.any():
                batch_loss[gid] = per_example_loss[mask].sum()
                batch_count[gid] = mask.sum().float()
        group_mean_loss = torch.where(
            batch_count > 0, batch_loss / batch_count.clamp(min=1), torch.zeros_like(batch_loss)
        )
        with torch.no_grad():
            new_q = self.q * torch.exp(self.robust_step_size * group_mean_loss)
            self.q = new_q / new_q.sum()
        return (self.q * group_mean_loss).sum()


@dataclass
class EpochLog:
    epoch: int; train_loss: float; val_avg_acc: float
    val_worst_group_acc: float; val_per_group: dict


def train_groupdro(device: str = "cuda") -> dict:
    set_seed(SEED)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    train_subset = load_waterbirds("train")
    val_subset = load_waterbirds("val")
    train_loader = make_loader(train_subset, shuffle=True)
    val_loader = make_loader(val_subset, shuffle=False)
    print(f"[data] train={len(train_subset)}  val={len(val_subset)}")

    model = build_model().to(device)
    criterion = GroupDROLoss(num_groups=NUM_GROUPS, robust_step_size=0.01)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = []
    best_worst = -1.0
    best_path = MODEL_DIR / "groupdro_best.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch:02d} train"):
            x, y, metadata = batch
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            metadata_np = metadata.numpy()
            group_ids = torch.tensor(
                metadata_np[:, 1].astype(int) * 2 + metadata_np[:, 0].astype(int),
                dtype=torch.long, device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y, group_ids)
            loss.backward(); optimizer.step()
            running_loss += loss.item(); n_batches += 1
        scheduler.step()
        avg_train_loss = running_loss / max(n_batches, 1)

        buckets = collect_group_predictions(model, val_loader, device)
        per_group = accuracy_per_group(buckets)
        valid = [v for v in per_group.values() if not np.isnan(v)]
        val_avg = float(np.mean(valid))
        val_worst = float(min(valid))

        history.append(EpochLog(epoch, avg_train_loss, val_avg, val_worst, per_group))
        q_str = "[" + ", ".join(f"{w:.3f}" for w in criterion.q.cpu().numpy()) + "]"
        print(f"[epoch {epoch:02d}] train_loss={avg_train_loss:.4f}  val_avg={val_avg:.4f}  val_worst_group={val_worst:.4f}  q={q_str}")

        if val_worst > best_worst:
            best_worst = val_worst
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_worst_group_acc": val_worst, "val_avg_acc": val_avg,
                        "val_per_group": per_group}, best_path)

    history_path = MODEL_DIR / "groupdro_history.json"
    history_path.write_text(json.dumps([log.__dict__ for log in history], indent=2))
    print(f"[done] best worst-group val acc = {best_worst:.4f}")
    return {"best_checkpoint": str(best_path), "best_worst_group_val_acc": best_worst}


if __name__ == "__main__":
    train_groupdro()
