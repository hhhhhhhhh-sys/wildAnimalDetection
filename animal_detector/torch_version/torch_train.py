"""
torch_train.py
---------------
Trains torch_model.AnimalDetectorCNN on the dataset described by an
annotations.csv (same schema as the NumPy pipeline).

Run:
    python3 torch_train.py --data_root ../dataset --epochs 30

Requires: pip install torch torchvision
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_dataset import AnimalDetectionDataset, CLASSES
from torch_model import AnimalDetectorCNN


def compute_loss(objectness_logit, box_pred, class_logits, target, box_loss_scale=2.0):
    has_animal = target["has_animal"]
    bce = nn.functional.binary_cross_entropy_with_logits(objectness_logit, has_animal)

    mask = has_animal.bool()
    if mask.any():
        box_loss = nn.functional.mse_loss(box_pred[mask], target["box"][mask])
        cls_loss = nn.functional.cross_entropy(class_logits[mask], target["class_id"][mask])
    else:
        box_loss = torch.tensor(0.0, device=objectness_logit.device)
        cls_loss = torch.tensor(0.0, device=objectness_logit.device)

    total = bce + box_loss_scale * box_loss + cls_loss
    return total, bce.item(), box_loss.item(), cls_loss.item()


def train(data_root, epochs=30, batch_size=32, lr=1e-3, img_size=128, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = AnimalDetectionDataset(os.path.join(data_root, "train"), img_size=img_size)
    val_ds = AnimalDetectionDataset(os.path.join(data_root, "val"), img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = AnimalDetectorCNN(n_classes=len(CLASSES), img_size=img_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, target in train_loader:
            x = x.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            optimizer.zero_grad()
            obj_logit, box_pred, cls_logits = model(x)
            loss, bce, box_l, cls_l = compute_loss(obj_logit, box_pred, cls_logits, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct_obj, correct_cls, n_obj, n_cls = 0, 0, 0, 0
        with torch.no_grad():
            for x, target in val_loader:
                x = x.to(device)
                target = {k: v.to(device) for k, v in target.items()}
                obj_logit, box_pred, cls_logits = model(x)
                obj_pred = (torch.sigmoid(obj_logit) > 0.5).float()
                correct_obj += (obj_pred == target["has_animal"]).sum().item()
                n_obj += x.size(0)
                mask = target["has_animal"].bool()
                if mask.any():
                    pred_cls = cls_logits[mask].argmax(dim=1)
                    correct_cls += (pred_cls == target["class_id"][mask]).sum().item()
                    n_cls += mask.sum().item()

        print(f"epoch {epoch:3d}/{epochs}  loss={running_loss/len(train_loader):.4f}  "
              f"val_obj_acc={correct_obj/n_obj:.3f}  val_cls_acc={(correct_cls/max(1,n_cls)):.3f}")

    os.makedirs("../weights", exist_ok=True)
    torch.save(model.state_dict(), "../weights/animal_detector_torch.pt")
    print("Saved -> ../weights/animal_detector_torch.pt")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=128)
    args = parser.parse_args()
    train(args.data_root, args.epochs, args.batch_size, args.lr, args.img_size)
