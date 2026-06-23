"""Grad-CAM saliency maps for visual diagnosis of the ERM shortcut.

A perfect way to "prove" the model is cheating is to show that its
attention lies on the BACKGROUND (water / land) rather than on the
BIRD. Grad-CAM produces a heatmap that highlights which image regions
push a given prediction.

We use Grad-CAM on the last convolutional layer (layer4 of ResNet-50).
For each chosen example we save a side-by-side figure:
    [original image]  [Grad-CAM heatmap]  [overlay]

We sample examples from each of the 4 subgroups so the story is clear.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from erm.config import GROUP_NAMES, SALiency_DIR
from erm.data import group_id_from_metadata, load_waterbirds
from erm.evaluate import load_checkpoint
from erm.model import get_gradcam_target_layer


# ----------------------------------------------------------------------
# Grad-CAM implementation (minimal, self-contained)
# ----------------------------------------------------------------------
class GradCAM:
    """Minimal Grad-CAM: gradients of a target class w.r.t. a conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx):
        """Return a HxW heatmap in [0,1] for the given class."""
        self.model.eval()
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        self.model.zero_grad()
        score.backward()

        # Global-average-pool the gradients to get channel weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1].
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ----------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------
def denormalize(img_tensor):
    """Undo the ImageNet normalization so the image is displayable."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


def overlay_heatmap(rgb_img, heatmap):
    """Blend a heatmap onto an RGB image, resizing heatmap to match image."""
    import matplotlib.cm as cm

    # Resize heatmap (typically 7x7) to match the image (typically 224x224).
    if heatmap.shape != rgb_img.shape[:2]:
        h_tensor = torch.tensor(heatmap).float().unsqueeze(0).unsqueeze(0)
        h_resized = F.interpolate(
            h_tensor,
            size=(rgb_img.shape[0], rgb_img.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        heatmap = h_resized.squeeze().numpy()

    colored = cm.jet(heatmap)[:, :, :3]  # drop alpha
    overlay = 0.5 * rgb_img + 0.5 * colored
    return np.clip(overlay, 0, 1)


def _find_example_indices(dataset, target_group, n=1):
    """Find the first `n` indices whose group == target_group."""
    found = []
    for i in range(len(dataset)):
        _, _, metadata = dataset[i]
        if group_id_from_metadata(metadata) == target_group:
            found.append(i)
            if len(found) >= n:
                break
    return found


def generate_saliency_for_checkpoint(checkpoint, device="cuda", per_group=1):
    """Save a grid of saliency images covering all 4 subgroups."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint).to(device)
    target_layer = get_gradcam_target_layer(model)
    cam = GradCAM(model, target_layer)

    test_subset = load_waterbirds("test")

    n_groups = len(GROUP_NAMES)
    n_cols = n_groups * 3  # original | heatmap | overlay per group

    # CRITICAL: squeeze=False guarantees axes is always 2D,
    # regardless of nrows/ncols.
    fig, axes = plt.subplots(
        nrows=per_group,
        ncols=n_cols,
        figsize=(3 * n_groups, 3 * per_group),
        squeeze=False,
    )

    for col, gid in enumerate(range(n_groups)):
        idxs = _find_example_indices(test_subset, gid, n=per_group)
        if not idxs:
            print(f"[warn] no examples found for group {gid} ({GROUP_NAMES[gid]})")
            continue
        for row, idx in enumerate(idxs):
            if row >= per_group:
                break  # safety guard
            x, y, metadata = test_subset[idx]
            label = int(y)
            x_batch = x.unsqueeze(0).to(device)
            heatmap = cam(x_batch, label)

            rgb = denormalize(x).permute(1, 2, 0).cpu().numpy()
            overlay = overlay_heatmap(rgb, heatmap)

            ax_orig = axes[row, col * 3 + 0]
            ax_cam = axes[row, col * 3 + 1]
            ax_over = axes[row, col * 3 + 2]

            ax_orig.imshow(rgb)
            ax_orig.set_title(f"{GROUP_NAMES[gid]}\nlabel={label}")
            ax_cam.imshow(heatmap, cmap="jet")
            ax_cam.set_title("Grad-CAM")
            ax_over.imshow(overlay)
            ax_over.set_title("overlay")
            for ax in (ax_orig, ax_cam, ax_over):
                ax.axis("off")

    fig.suptitle(f"Saliency maps \u2014 {checkpoint.stem}", fontsize=14)
    fig.tight_layout()
    out_path = SALiency_DIR / f"{checkpoint.stem}_saliency.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saliency] saved -> {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--per-group", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    generate_saliency_for_checkpoint(
        args.checkpoint, device=args.device, per_group=args.per_group
    )
