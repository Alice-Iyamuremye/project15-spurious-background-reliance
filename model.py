"""ResNet-18 backbone with a 2-class head for Waterbirds.

We use ImageNet-pretrained weights because:
    1. They give a much better starting point than random init.
    2. The WILDS Waterbirds baseline protocol uses pretrained ResNets.
    3. With only ~4.8k training images, training from scratch would
       underfit dramatically.

The final fully-connected layer is replaced with a 2-output layer
matching our binary task (landbird vs waterbird).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from erm.config import MODEL_NAME, NUM_CLASSES, PRETRAINED


def build_model() -> nn.Module:
    """Return a ResNet-18 (pretrained) with NUM_CLASSES outputs."""
    if MODEL_NAME == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
        model = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")

    # Replace the 1000-class ImageNet head with our binary head.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the last conv layer for Grad-CAM.

    For ResNet-18, the standard choice is `layer4`. This is the deepest
    feature map before global average pooling and the FC head, so it
    carries the most spatially-resolved "what the model saw" signal.
    """
    return model.layer4
