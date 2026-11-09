"""
train.py
--------
Trains the NumPy CNN (model_numpy.AnimalDetectorCNN) on the synthetic
animal-detection dataset.

Loss = BCE(objectness) + has_animal * [ MSE(bbox) + CrossEntropy(class) ]

We backprop this combined loss manually into the raw network output,
then let the network's own .backward() propagate it through every layer.
"""

import os
import time
import numpy as np

from model_numpy import AnimalDetectorCNN
from data_utils import load_split, IMG_SIZE, N_CLASSES

SRC_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(SRC_DIR, "..", "dataset")
WEIGHTS_DIR = os.path.join(SRC_DIR, "..", "weights")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_loss_and_grad(raw, obj_t, box_t, cls_t):
    """
    raw: (N,9) network output (pre-activation)
    obj_t: (N,) 0/1
    box_t: (N,4) normalized target box (only meaningful where obj_t==1)
    cls_t: (N,4) one-hot (only meaningful where obj_t==1)
    Returns scalar loss and d(raw) gradient, shape (N,9)
    """
    n = raw.shape[0]
    obj_logit = raw[:, 0]
    box_logit = raw[:, 1:5]
    cls_logit = raw[:, 5:]

    obj_pred = sigmoid(obj_logit)
    box_pred = sigmoid(box_logit)
    ex = np.exp(cls_logit - cls_logit.max(axis=1, keepdims=True))
    cls_pred = ex / ex.sum(axis=1, keepdims=True)

    eps = 1e-7
    obj_loss = -(obj_t * np.log(obj_pred + eps) + (1 - obj_t) * np.log(1 - obj_pred + eps))

    mask = obj_t[:, None]  # only penalize box/class when an animal is present
    n_pos = max(1.0, obj_t.sum())  # normalize box/class loss by #positives, not full batch
    box_loss = ((box_pred - box_t) ** 2) * mask
    cls_loss = -np.sum(cls_t * np.log(cls_pred + eps), axis=1, keepdims=True) * mask

    total_loss = obj_loss.mean() + box_loss.sum() / n_pos + cls_loss.sum() / n_pos

    d_obj = 2.0 * (obj_pred - obj_t) / n
    d_box = 2 * (box_pred - box_t) * mask / n_pos
    d_cls = (cls_pred - cls_t) * mask / n_pos

    d_raw = np.concatenate([d_obj[:, None], d_box, d_cls], axis=1)
    return total_loss, d_raw


def accuracy(model, X, obj_t, box_t, cls_t):
    obj_pred, box_pred, cls_pred = model.predict(X)
    obj_acc = ((obj_pred > 0.5).astype(np.float32) == obj_t).mean()
    has = obj_t == 1
    if has.sum() > 0:
        cls_acc = (cls_pred[has].argmax(axis=1) == cls_t[has].argmax(axis=1)).mean()
        iou = mean_iou(box_pred[has] * IMG_SIZE, box_t[has] * IMG_SIZE)
    else:
        cls_acc, iou = 0.0, 0.0
    return obj_acc, cls_acc, iou


def mean_iou(pred_boxes, true_boxes):
    ious = []
    for p, t in zip(pred_boxes, true_boxes):
        xa, ya = max(p[0], t[0]), max(p[1], t[1])
        xb, yb = min(p[2], t[2]), min(p[3], t[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_p = max(0, p[2] - p[0]) * max(0, p[3] - p[1])
        area_t = max(0, t[2] - t[0]) * max(0, t[3] - t[1])
        union = area_p + area_t - inter
        ious.append(inter / union if union > 0 else 0)
    return float(np.mean(ious))


def train(epochs=45, batch_size=32, lr=0.002, seed=0):
    np.random.seed(seed)
    print("Loading dataset...")
    Xtr, obj_tr, box_tr, cls_tr = load_split(os.path.join(DATASET_DIR, "train"))
    Xva, obj_va, box_va, cls_va = load_split(os.path.join(DATASET_DIR, "val"))
    print(f"train={len(Xtr)} val={len(Xva)}")

    model = AnimalDetectorCNN(img_size=IMG_SIZE, n_classes=N_CLASSES)
    n = len(Xtr)
    history = []

    best_score = -1
    best_state_path = os.path.join(WEIGHTS_DIR, "animal_detector.npz")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        cur_lr = lr * (0.97 ** epoch)  # gentle decay for a cleaner late-stage convergence
        idx = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            b_idx = idx[start:start + batch_size]
            xb = Xtr[b_idx]
            raw = model.forward(xb)
            loss, d_raw = compute_loss_and_grad(raw, obj_tr[b_idx], box_tr[b_idx], cls_tr[b_idx])
            model.backward(d_raw, cur_lr)
            epoch_loss += loss
            n_batches += 1

        val_obj_acc, val_cls_acc, val_iou = accuracy(model, Xva, obj_va, box_va, cls_va)
        dt = time.time() - t0
        print(f"epoch {epoch:2d}/{epochs}  loss={epoch_loss/n_batches:.4f}  "
              f"val_obj_acc={val_obj_acc:.3f}  val_cls_acc={val_cls_acc:.3f}  "
              f"val_box_iou={val_iou:.3f}  ({dt:.1f}s)")
        history.append((epoch, epoch_loss / n_batches, val_obj_acc, val_cls_acc, val_iou))

        score = val_obj_acc + val_cls_acc + val_iou  # simple combined metric for checkpointing
        if score > best_score:
            best_score = score
            model.save(best_state_path)

    print(f"Saved best trained weights -> {best_state_path}")

    # save training history for a report / plot
    with open(os.path.join(WEIGHTS_DIR, "training_log.csv"), "w") as f:
        f.write("epoch,loss,val_obj_acc,val_cls_acc,val_box_iou\n")
        for row in history:
            f.write(",".join(str(x) for x in row) + "\n")

    return model, history


if __name__ == "__main__":
    train()
