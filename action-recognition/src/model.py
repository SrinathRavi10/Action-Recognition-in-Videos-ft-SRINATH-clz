"""
Model factory: a Kinetics-400-pretrained R(2+1)D-18 backbone with the
final classifier replaced for our class subset.

Why R(2+1)D-18:
  - Factorizes 3D convolutions into a 2D spatial + 1D temporal conv,
    which is far lighter than a full 3D CNN of similar depth.
  - Ships with Kinetics-400 pretrained weights in torchvision, so we get
    strong motion features for free via transfer learning.
  - Comfortably fits an 8-frame/16-frame, 112x112 clip batch on a single
    16GB GPU (or Colab's T4/L4), unlike heavier options such as SlowFast.
"""

import torch.nn as nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze the last residual block so the model can adapt motion
        # features to our classes, not just the final layer
        for param in model.layer4.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # always trainable
    return model
