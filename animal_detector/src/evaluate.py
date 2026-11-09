"""
evaluate.py
-----------
Runs the trained model over the held-out TEST split and reports:
  - objectness accuracy, precision, recall (animal vs no_animal)
  - classification accuracy on true positives
  - mean IoU on true positives
This is the "official" report-worthy number to show a professor,
since it's on data the model never saw during training or validation.
"""

import os
import numpy as np
from model_numpy import AnimalDetectorCNN
from data_utils import load_split, IMG_SIZE
from train import mean_iou

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "animal_detector.npz")


def evaluate():
    X, obj_t, box_t, cls_t = load_split(os.path.join(DATASET_DIR, "test"))
    model = AnimalDetectorCNN.load(WEIGHTS_PATH)
    obj_pred, box_pred, cls_pred = model.predict(X)

    pred_pos = obj_pred > 0.5
    true_pos = obj_t == 1

    tp = np.sum(pred_pos & true_pos)
    fp = np.sum(pred_pos & ~true_pos)
    fn = np.sum(~pred_pos & true_pos)
    tn = np.sum(~pred_pos & ~true_pos)

    acc = (tp + tn) / len(X)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    has = true_pos
    cls_acc = (cls_pred[has].argmax(axis=1) == cls_t[has].argmax(axis=1)).mean()
    iou = mean_iou(box_pred[has] * IMG_SIZE, box_t[has] * IMG_SIZE)

    print(f"Test set size: {len(X)}")
    print(f"Objectness -> accuracy={acc:.3f} precision={precision:.3f} recall={recall:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Classification accuracy (on true positives): {cls_acc:.3f}")
    print(f"Mean IoU (on true positives): {iou:.3f}")


if __name__ == "__main__":
    evaluate()
