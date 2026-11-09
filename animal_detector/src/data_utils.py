"""
data_utils.py
-------------
Loads the synthetic dataset (images + CSV annotations) into NumPy arrays
ready for training / evaluation.
"""

import os
import csv
import numpy as np
from PIL import Image

IMG_SIZE = 64
N_CLASSES = 4  # dog, cat, bird, rabbit  (no_animal is handled via objectness=0)


def load_split(split_dir):
    ann_path = os.path.join(split_dir, "annotations.csv")
    img_dir = os.path.join(split_dir, "images")
    rows = list(csv.DictReader(open(ann_path)))

    n = len(rows)
    X = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    obj = np.zeros(n, dtype=np.float32)
    box = np.zeros((n, 4), dtype=np.float32)       # normalized xmin,ymin,xmax,ymax
    cls_onehot = np.zeros((n, N_CLASSES), dtype=np.float32)

    for i, r in enumerate(rows):
        img = Image.open(os.path.join(img_dir, r["filename"])).convert("RGB")
        X[i] = np.asarray(img, dtype=np.float32) / 255.0
        has_animal = int(r["has_animal"])
        obj[i] = has_animal
        if has_animal:
            box[i] = [
                float(r["xmin"]) / IMG_SIZE,
                float(r["ymin"]) / IMG_SIZE,
                float(r["xmax"]) / IMG_SIZE,
                float(r["ymax"]) / IMG_SIZE,
            ]
            cls_onehot[i, int(r["class_id"])] = 1.0

    return X, obj, box, cls_onehot
