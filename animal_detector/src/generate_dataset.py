"""
generate_dataset.py
--------------------
Builds a synthetic "animal detection" dataset from scratch (no downloads needed).

Each image contains either:
  - one drawn animal silhouette (dog / cat / bird / rabbit) at a random
    position, size, rotation-ish variation, and color, on a random textured
    background   -> label = class name + bounding box
  - just background texture / random distractor shapes, no animal
    -> label = "no_animal", box = (0,0,0,0)

This gives a real, reproducible dataset with pixel-accurate ground-truth
boxes, which is what makes training an actual detector possible without
needing to download real photos.

Output:
  dataset/<split>/images/*.png
  dataset/<split>/annotations.csv   (filename, class_id, class_name, has_animal, xmin, ymin, xmax, ymax)
"""

import os
import csv
import random
import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE = 64  # keep small -> fast to train in pure numpy
CLASSES = ["dog", "cat", "bird", "rabbit"]  # class_id 0..3, 4 = no_animal
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def random_bg(size):
    """Random textured / solid background so the model can't cheat on color alone."""
    mode = random.choice(["solid", "noisy", "gradient"])
    img = Image.new("RGB", (size, size))
    if mode == "solid":
        color = tuple(random.randint(150, 255) for _ in range(3))
        img = Image.new("RGB", (size, size), color)
    elif mode == "noisy":
        arr = np.random.randint(140, 255, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
    else:
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        c1 = np.random.randint(140, 255, 3)
        c2 = np.random.randint(140, 255, 3)
        for y in range(size):
            t = y / size
            arr[y, :, :] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


def draw_dog(draw, cx, cy, s, color):
    body = [cx - s, cy - s * 0.5, cx + s * 0.6, cy + s * 0.5]
    draw.ellipse(body, fill=color)
    head_r = s * 0.45
    hx, hy = cx + s * 0.7, cy - s * 0.2
    draw.ellipse([hx - head_r, hy - head_r, hx + head_r, hy + head_r], fill=color)
    ear = s * 0.3
    draw.polygon([(hx - head_r * 0.6, hy - head_r), (hx - head_r, hy - head_r - ear), (hx - head_r * 0.2, hy - head_r * 0.6)], fill=color)
    draw.polygon([(hx + head_r * 0.6, hy - head_r), (hx + head_r, hy - head_r - ear), (hx + head_r * 0.2, hy - head_r * 0.6)], fill=color)
    for lx in (cx - s * 0.6, cx, cx + s * 0.3):
        draw.line([lx, cy + s * 0.4, lx, cy + s * 0.9], fill=color, width=max(1, int(s * 0.12)))
    all_x = [body[0], body[2], hx - head_r, hx + head_r, cx - s * 0.6, cx + s * 0.3]
    all_y = [body[1], body[3], hy - head_r - ear, cy + s * 0.9]
    return min(all_x), min(all_y), max(all_x), max(all_y)


def draw_cat(draw, cx, cy, s, color):
    body = [cx - s * 0.9, cy - s * 0.4, cx + s * 0.6, cy + s * 0.5]
    draw.ellipse(body, fill=color)
    head_r = s * 0.4
    hx, hy = cx + s * 0.55, cy - s * 0.35
    draw.ellipse([hx - head_r, hy - head_r, hx + head_r, hy + head_r], fill=color)
    ear = s * 0.35
    draw.polygon([(hx - head_r * 0.7, hy - head_r * 0.7), (hx - head_r * 0.9, hy - head_r - ear), (hx - head_r * 0.1, hy - head_r * 0.5)], fill=color)
    draw.polygon([(hx + head_r * 0.7, hy - head_r * 0.7), (hx + head_r * 0.9, hy - head_r - ear), (hx + head_r * 0.1, hy - head_r * 0.5)], fill=color)
    tail_x0, tail_y0 = cx - s * 0.9, cy
    tail_x1, tail_y1 = cx - s * 1.4, cy - s * 0.6
    draw.line([tail_x0, tail_y0, tail_x1, tail_y1], fill=color, width=max(1, int(s * 0.15)))
    all_x = [body[0], body[2], hx - head_r, hx + head_r, tail_x1]
    all_y = [body[1], body[3], hy - head_r - ear, tail_y1]
    return min(all_x), min(all_y), max(all_x), max(all_y)


def draw_bird(draw, cx, cy, s, color):
    body = [cx - s * 0.6, cy - s * 0.45, cx + s * 0.6, cy + s * 0.45]
    draw.ellipse(body, fill=color)
    head_r = s * 0.35
    hx, hy = cx + s * 0.55, cy - s * 0.5
    draw.ellipse([hx - head_r, hy - head_r, hx + head_r, hy + head_r], fill=color)
    beak = [(hx + head_r * 0.8, hy), (hx + head_r * 1.6, hy + s * 0.08), (hx + head_r * 0.8, hy + s * 0.25)]
    draw.polygon(beak, fill=(230, 160, 30))
    wing = [(cx - s * 0.1, cy - s * 0.1), (cx - s * 0.6, cy + s * 0.5), (cx + s * 0.3, cy + s * 0.15)]
    draw.polygon(wing, fill=color)
    all_x = [body[0], body[2], hx - head_r, hx + head_r * 1.6]
    all_y = [body[1], body[3], hy - head_r, cy + s * 0.5]
    return min(all_x), min(all_y), max(all_x), max(all_y)


def draw_rabbit(draw, cx, cy, s, color):
    body = [cx - s * 0.6, cy - s * 0.3, cx + s * 0.7, cy + s * 0.6]
    draw.ellipse(body, fill=color)
    head_r = s * 0.4
    hx, hy = cx + s * 0.75, cy - s * 0.1
    draw.ellipse([hx - head_r, hy - head_r, hx + head_r, hy + head_r], fill=color)
    ear_h = s * 0.9
    draw.ellipse([hx - head_r * 0.7, hy - head_r - ear_h, hx - head_r * 0.1, hy - head_r * 0.3], fill=color)
    draw.ellipse([hx + head_r * 0.1, hy - head_r - ear_h, hx + head_r * 0.7, hy - head_r * 0.3], fill=color)
    all_x = [body[0], body[2], hx - head_r * 0.7, hx + head_r * 0.7]
    all_y = [body[1], body[3], hy - head_r - ear_h, cy + s * 0.6]
    return min(all_x), min(all_y), max(all_x), max(all_y)


DRAW_FN = {"dog": draw_dog, "cat": draw_cat, "bird": draw_bird, "rabbit": draw_rabbit}


def random_distractor(draw, size):
    """A random non-animal shape to make 'no_animal' images non-trivial."""
    for _ in range(random.randint(1, 3)):
        shape = random.choice(["rect", "line", "circle"])
        color = tuple(random.randint(0, 200) for _ in range(3))
        x0, y0 = random.randint(0, size), random.randint(0, size)
        x1, y1 = random.randint(0, size), random.randint(0, size)
        if shape == "rect":
            xa, xb = sorted([x0, x1])
            ya, yb = sorted([y0, y1])
            if xa == xb:
                xb += 1
            if ya == yb:
                yb += 1
            draw.rectangle([xa, ya, xb, yb], outline=color, width=2)
        elif shape == "line":
            draw.line([x0, y0, x1, y1], fill=color, width=2)
        else:
            r = random.randint(3, 10)
            draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], outline=color, width=2)


def make_sample(size=IMG_SIZE, force_no_animal=False):
    img = random_bg(size)
    draw = ImageDraw.Draw(img)

    has_animal = 0 if force_no_animal else (random.random() > 0.15)  # ~85% contain an animal
    if not has_animal:
        random_distractor(draw, size)
        return img, "no_animal", 4, 0, (0, 0, 0, 0)

    cls = random.choice(CLASSES)
    s = random.uniform(size * 0.18, size * 0.32)
    cx = random.uniform(s * 1.3, size - s * 1.3)
    cy = random.uniform(s * 1.3, size - s * 1.3)
    color = tuple(random.randint(20, 180) for _ in range(3))

    xmin, ymin, xmax, ymax = DRAW_FN[cls](draw, cx, cy, s, color)
    # occasional light clutter even when an animal is present -> forces model to localize, not just detect "any shape"
    if random.random() < 0.3:
        random_distractor(draw, size)

    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(size, xmax), min(size, ymax)
    return img, cls, CLASS_TO_ID[cls], 1, (xmin, ymin, xmax, ymax)


def build_split(split_name, n_samples, out_root):
    split_dir = os.path.join(out_root, split_name)
    img_dir = os.path.join(split_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for i in range(n_samples):
        img, cls_name, cls_id, has_animal, box = make_sample()
        fname = f"{split_name}_{i:05d}.png"
        img.save(os.path.join(img_dir, fname))
        rows.append([fname, cls_id, cls_name, has_animal, *[round(v, 2) for v in box]])

    with open(os.path.join(split_dir, "annotations.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class_id", "class_name", "has_animal", "xmin", "ymin", "xmax", "ymax"])
        writer.writerows(rows)
    print(f"{split_name}: {n_samples} images -> {split_dir}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset")
    build_split("train", 900, OUT_ROOT)
    build_split("val", 150, OUT_ROOT)
    build_split("test", 60, OUT_ROOT)
    print("Classes:", CLASSES, "(+ no_animal = id 4)")
