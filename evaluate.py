"""Subgroup-aware evaluation on the Waterbirds test set.

We compute:
    - Overall accuracy
    - Per-subgroup accuracy for all 4 (bird x background) groups
    - Worst-group accuracy   <-- THIS is the headline number
    - A confusion matrix per minority subgroup to localize the failure

This file can also load any checkpoint (e.g., an ERM or a Group DRO model)
and report the same metrics, so the comparison later is apples-to-apples.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from erm.config import GROUP_NAMES, NUM_GROUPS, RESULT_DIR
from erm.data import (
    accuracy_per_group,
    collect_group_predictions,
    load_waterbirds,
    make_loader,
)
from erm.model import build_model


def load_checkpoint(path: Path) -> torch.nn.Module:
    """Load a saved checkpoint into a fresh ResNet-18."""
    model = build_model()
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return model


def evaluate_on_test(
    checkpoint: Path,
    device: str = "cuda",
    save_report: bool = True,
) -> dict:
    """Run a checkpoint on the test set and produce a subgroup report."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint).to(device)

    test_subset = load_waterbirds("test")
    test_loader = make_loader(test_subset, shuffle=False)
    print(f"[test] n={len(test_subset)}")

    buckets = collect_group_predictions(model, test_loader, device)
    per_group = accuracy_per_group(buckets)

    valid = [v for v in per_group.values() if not np.isnan(v)]
    overall = float(np.mean(valid))
    worst_group_name = min(per_group, key=per_group.get)
    worst_group_acc = per_group[worst_group_name]

    # Per-group confusion summary
    confusion = {}
    for gid in range(NUM_GROUPS):
        items = buckets[gid]
        if not items:
            continue
        # tp = correctly classified; fp-style = wrong predictions
        correct = sum(1 for p, t in items if p == t)
        confusion[GROUP_NAMES[gid]] = {
            "n": len(items),
            "correct": correct,
            "errors": len(items) - correct,
        }

    print("\n=== Test Subgroup Accuracy ===")
    for name, acc in per_group.items():
        flag = "  <-- WORST" if name == worst_group_name else ""
        print(f"  {name:<22s}  {acc:.4f}{flag}")
    print(f"  {'AVERAGE':<22s}  {overall:.4f}")
    print(f"  {'WORST GROUP':<22s}  {worst_group_acc:.4f}  ({worst_group_name})")

    report = {
        "checkpoint": str(checkpoint),
        "overall_accuracy": overall,
        "worst_group_name": worst_group_name,
        "worst_group_accuracy": worst_group_acc,
        "per_group_accuracy": per_group,
        "per_group_confusion": confusion,
    }

    if save_report:
        out = RESULT_DIR / f"{checkpoint.stem}_test_report.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"[report] -> {out}")

    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    evaluate_on_test(args.checkpoint, device=args.device)
