"""
torch_model.py
--------------
PyTorch version of the same detector concept as the NumPy model: a small
CNN backbone with three heads (objectness, bounding box, class). This is
a direct, faster, GPU-capable upgrade from the from-scratch NumPy model --
same idea, standard framework.

Requires: pip install torch torchvision
"""

import torch
import torch.nn as nn


class AnimalDetectorCNN(nn.Module):
    def __init__(self, n_classes=4, img_size=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # /2
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # /4
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # /8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        flat_dim = 128 * 4 * 4
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.ReLU(), nn.Dropout(0.3),
        )
        self.objectness_head = nn.Linear(256, 1)
        self.box_head = nn.Linear(256, 4)
        self.class_head = nn.Linear(256, n_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.flatten(1)
        feat = self.fc(feat)
        objectness_logit = self.objectness_head(feat).squeeze(-1)
        box = torch.sigmoid(self.box_head(feat))
        class_logits = self.class_head(feat)
        return objectness_logit, box, class_logits
