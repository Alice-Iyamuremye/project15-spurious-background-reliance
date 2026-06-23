"""End-to-end driver for the ERM baseline.

Usage:
    python -m erm.run_erm              # full pipeline
    python -m erm.run_erm --skip-train # re-evaluate an existing checkpoint
"""
from __future__ import annotations

import argparse

from erm.config import MODEL_DIR
from erm.evaluate import evaluate_on_test
from erm.saliency import generate_saliency_for_checkpoint
from erm.train_erm import train_erm


def main(skip_train: bool = False, device: str = "cuda") -> None:
    if not skip_train:
        info = train_erm(device=device)
        ckpt = info["best_checkpoint"]
    else:
        ckpt = str(MODEL_DIR / "erm_best.pt")

    # 1) Subgroup evaluation on the test set
    evaluate_on_test(ckpt, device=device)

    # 2) Saliency maps on one example per subgroup
    generate_saliency_for_checkpoint(ckpt, device=device, per_group=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    main(skip_train=args.skip_train, device=args.device)
