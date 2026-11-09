"""
train_thermal_wildlife.py
--------------------------
Generates thermal infrared dataset & trains PyTorch Thermal Wildlife Model.
Saves trained weight checkpoint to animal_detector/weights/thermal_animal_detector.pt.
"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

SRC_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SRC_DIR.parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = WEIGHTS_DIR / "thermal_animal_detector.pt"

WILDLIFE_CLASSES = ["deer", "lion", "tiger", "boar", "wolf", "leopard", "dog", "cat", "horse", "cow", "bird"]
CLASS_TO_ID = {c: i for i, c in enumerate(WILDLIFE_CLASSES)}
ID_TO_CLASS = {i: c for i, c in enumerate(WILDLIFE_CLASSES)}

COLORMAP_CHOICES = [cv2.COLORMAP_INFERNO, cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_BONE, cv2.COLORMAP_HOT]


def generate_synthetic_thermal_sample(img_size=128):
    """Generates a synthetic thermal infrared animal image and bounding box label."""
    # Dark/cool thermal background
    bg_intensity = random.randint(10, 60)
    bg_noise = np.random.randint(0, 30, (img_size, img_size), dtype=np.uint8)
    gray_img = np.clip(bg_noise + bg_intensity, 0, 255).astype(np.uint8)

    has_animal = random.choice([True, True, True, False])
    class_name = "no_animal"
    class_id = len(WILDLIFE_CLASSES)
    bbox = [0.0, 0.0, 0.0, 0.0]

    if has_animal:
        class_name = random.choice(WILDLIFE_CLASSES)
        class_id = CLASS_TO_ID[class_name]

        # Heat intensity for living animal (bright thermal body)
        heat_val = random.randint(180, 255)
        pil_gray = Image.fromarray(gray_img)
        draw = ImageDraw.Draw(pil_gray)

        s = random.randint(16, 36)
        cx = random.randint(s + 5, img_size - s - 5)
        cy = random.randint(s + 5, img_size - s - 5)

        # Draw warm quadruped heat signature
        x1, y1 = max(0, cx - s), max(0, cy - int(s * 0.6))
        x2, y2 = min(img_size, cx + s), min(img_size, cy + int(s * 0.6))

        draw.ellipse([x1, y1, x2, y2], fill=heat_val)
        # Head heat
        hx = min(img_size - 2, x2 + int(s * 0.3))
        hy = max(2, y1 - int(s * 0.2))
        draw.ellipse([hx - int(s * 0.4), hy - int(s * 0.4), hx + int(s * 0.4), hy + int(s * 0.4)], fill=heat_val)

        # Legs
        draw.line([cx - int(s * 0.5), cy, cx - int(s * 0.5), cy + int(s * 0.8)], fill=heat_val, width=3)
        draw.line([cx + int(s * 0.3), cy, cx + int(s * 0.3), cy + int(s * 0.8)], fill=heat_val, width=3)

        gray_img = np.array(pil_gray)

        bx1 = max(0, cx - s - 2)
        by1 = max(0, hy - int(s * 0.4) - 2)
        bx2 = min(img_size, hx + int(s * 0.4) + 2)
        by2 = min(img_size, cy + int(s * 0.8) + 2)

        bbox = [bx1 / img_size, by1 / img_size, bx2 / img_size, by2 / img_size]

    # Apply colormap (Inferno, Jet, Turbo, Bone, Hot, or Grayscale IR)
    apply_col = random.choice([True, True, False])
    if apply_col:
        cmap = random.choice(COLORMAP_CHOICES)
        bgr = cv2.applyColorMap(gray_img, cmap)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(rgb), class_id, bbox, has_animal


class SyntheticThermalDataset(Dataset):
    def __init__(self, n_samples=600, img_size=128):
        self.n_samples = n_samples
        self.img_size = img_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        pil_img, class_id, bbox, has_animal = generate_synthetic_thermal_sample(self.img_size)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        tensor_img = torch.tensor(arr).permute(2, 0, 1)

        return tensor_img, {
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "has_animal": torch.tensor(1.0 if has_animal else 0.0, dtype=torch.float32),
        }


class ThermalWildlifeCNN(nn.Module):
    def __init__(self, n_classes=len(WILDLIFE_CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.obj_head = nn.Linear(128, 1)
        self.box_head = nn.Linear(128, 4)
        self.cls_head = nn.Linear(128, n_classes)

    def forward(self, x):
        feat = self.features(x)
        flat = self.classifier(feat)
        obj = self.obj_head(flat).squeeze(-1)
        box = torch.sigmoid(self.box_head(flat))
        cls = self.cls_head(flat)
        return obj, box, cls


def train_model(epochs=12, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Thermal Wildlife Model on {device}...")

    ds_train = SyntheticThermalDataset(n_samples=800, img_size=128)
    ds_val = SyntheticThermalDataset(n_samples=200, img_size=128)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    model = ThermalWildlifeCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, target in loader_train:
            x = x.to(device)
            has_anim = target["has_animal"].to(device)
            gt_box = target["bbox"].to(device)
            gt_cls = target["class_id"].to(device)

            optimizer.zero_grad()
            obj_logits, box_preds, cls_logits = model(x)

            l_obj = bce_loss(obj_logits, has_anim)
            mask = has_anim.bool()

            if mask.any():
                l_box = mse_loss(box_preds[mask], gt_box[mask])
                valid_cls_mask = gt_cls[mask] < len(WILDLIFE_CLASSES)
                if valid_cls_mask.any():
                    l_cls = ce_loss(cls_logits[mask][valid_cls_mask], gt_cls[mask][valid_cls_mask])
                else:
                    l_cls = torch.tensor(0.0, device=device)
            else:
                l_box = torch.tensor(0.0, device=device)
                l_cls = torch.tensor(0.0, device=device)

            loss = l_obj + 2.0 * l_box + l_cls
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:2d}/{epochs} - Loss: {total_loss / len(loader_train):.4f}")

    torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
    print(f"[SUCCESS] Thermal Wildlife Model trained & saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model(epochs=12)
