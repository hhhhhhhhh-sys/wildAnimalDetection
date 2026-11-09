"""
pretrained_quickstart.py
-------------------------
THIS is the recommended path for detecting animals in REAL photographs.

Instead of training from zero on a tiny custom dataset, this loads a
Faster R-CNN model already pretrained by torchvision on the COCO dataset
(millions of real, diverse photos). COCO's 80 categories already include
several common animals: bird, cat, dog, horse, sheep, cow, elephant, bear,
zebra, giraffe. This script filters detections down to just those classes,
draws boxes, and prints an alert -- with NO training required.

First run downloads the pretrained weights automatically (needs internet
on your machine, which the sandbox that produced this code doesn't have --
that's why I could not run/test this file myself, but the API is standard
and stable in torchvision).

Usage:
    from pretrained_quickstart import detect_animal
    result = detect_animal("real_photo.jpg", "real_photo_boxed.jpg")

Requires: pip install torch torchvision
"""

import torch
import torchvision
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw

# COCO category names in torchvision's index order (index 0 = background / unused slots kept as 'N/A')
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]
ANIMAL_CLASSES = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}

_model = None


def _get_model(device):
    global _model
    if _model is None:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        _model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        _model.eval().to(device)
    return _model


def detect_animal(image_path, out_path=None, conf_threshold=0.6, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _get_model(device)

    img = Image.open(image_path).convert("RGB")
    x = to_tensor(img).to(device)

    with torch.no_grad():
        pred = model([x])[0]

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    detections = []

    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < conf_threshold:
            continue
        name = COCO_INSTANCE_CATEGORY_NAMES[label]
        if name not in ANIMAL_CLASSES:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        label_text = f"{name} {score*100:.0f}%"
        draw.rectangle([x1, max(0, y1 - 16), x1 + 8 * len(label_text), max(0, y1)], fill=(255, 0, 0))
        draw.text((x1 + 2, max(0, y1 - 15)), label_text, fill=(255, 255, 255))
        detections.append({"class": name, "confidence": float(score), "box": (x1, y1, x2, y2)})

    animal_detected = len(detections) > 0
    if animal_detected:
        names = ", ".join(sorted(set(d["class"] for d in detections)))
        alert = f"ALERT: animal(s) detected -> {names}"
    else:
        alert = "No animal detected"

    if out_path is None:
        import os
        base, _ = os.path.splitext(image_path)
        out_path = base + "_detected.png"
    annotated.save(out_path)

    print(alert)
    return {"animal_detected": animal_detected, "detections": detections, "alert": alert, "out_path": out_path}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 pretrained_quickstart.py <image_path> [output_path]")
        sys.exit(1)
    img_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    print(detect_animal(img_path, out))
