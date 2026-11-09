"""
torch_inference.py
-------------------
Same behavior as src/inference.py but using the trained PyTorch model.
No UI -- plain function you call from your own code:

    from torch_inference import detect_animal
    result = detect_animal("photo.jpg", "photo_boxed.jpg", weights_path="../weights/animal_detector_torch.pt")

Requires: pip install torch torchvision
"""

import sys
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from torch_model import AnimalDetectorCNN
from torch_dataset import CLASSES

CONF_THRESHOLD = 0.5


def detect_animal(image_path, out_path=None, weights_path="../weights/animal_detector_torch.pt",
                   img_size=128, conf_threshold=CONF_THRESHOLD, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = AnimalDetectorCNN(n_classes=len(CLASSES), img_size=img_size).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    orig = Image.open(image_path).convert("RGB")
    ow, oh = orig.size
    resized = orig.resize((img_size, img_size))
    x = TF.to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        obj_logit, box_pred, cls_logits = model(x)
        confidence = torch.sigmoid(obj_logit).item()
        class_probs = torch.softmax(cls_logits, dim=1)[0]
        class_id = int(class_probs.argmax())
        class_name = CLASSES[class_id]
        class_conf = float(class_probs[class_id])
        box = box_pred[0].cpu().numpy()

    animal_detected = confidence >= conf_threshold
    box_orig = (box[0] * ow, box[1] * oh, box[2] * ow, box[3] * oh)

    annotated = orig.copy()
    if animal_detected:
        draw = ImageDraw.Draw(annotated)
        line_w = max(2, int(min(ow, oh) * 0.01))
        draw.rectangle(box_orig, outline=(255, 0, 0), width=line_w)
        label = f"{class_name} {class_conf*100:.0f}%"
        draw.rectangle([box_orig[0], max(0, box_orig[1] - 14), box_orig[0] + 8 * len(label), max(0, box_orig[1])],
                        fill=(255, 0, 0))
        draw.text((box_orig[0] + 2, max(0, box_orig[1] - 13)), label, fill=(255, 255, 255))
        alert = f"ALERT: {class_name} detected! (confidence {confidence*100:.1f}%)"
    else:
        alert = f"No animal detected (confidence {confidence*100:.1f}%)"

    if out_path is None:
        import os
        base, _ = os.path.splitext(image_path)
        out_path = base + "_detected.png"
    annotated.save(out_path)

    result = {
        "animal_detected": animal_detected,
        "class": class_name if animal_detected else None,
        "confidence": round(confidence, 4),
        "box": tuple(round(float(v), 1) for v in box_orig) if animal_detected else None,
        "alert": alert,
        "out_path": out_path,
    }
    print(result["alert"])
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 torch_inference.py <image_path> [output_path]")
        sys.exit(1)
    img_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    print(detect_animal(img_path, out))
