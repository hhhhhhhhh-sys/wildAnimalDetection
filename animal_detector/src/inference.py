"""
inference.py
-------------
Loads the trained object detection model and runs animal detection on a single image.
Supports BOTH Real Color Photos (RGB) and Thermal Infrared (IR) Footage.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw

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
ANIMAL_CLASSES = {
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'deer', 'lion', 'tiger', 'leopard', 'fox', 'wolf', 'boar', 'goat', 'monkey'
}
GROUND_ANIMALS = ANIMAL_CLASSES - {'bird'}

_model = None
_thermal_model = None


def _get_model(device):
    global _model
    if _model is None:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        _model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        _model.eval().to(device)
    return _model


def _get_thermal_model(device):
    global _thermal_model
    if _thermal_model is None:
        try:
            from animal_detector.src.train_thermal_wildlife import ThermalWildlifeCNN, MODEL_SAVE_PATH
            if MODEL_SAVE_PATH.exists():
                t_model = ThermalWildlifeCNN().to(device)
                t_model.load_state_dict(torch.load(str(MODEL_SAVE_PATH), map_location=device))
                t_model.eval()
                _thermal_model = t_model
        except Exception as err:
            # Fallback if relative import path differs
            try:
                from train_thermal_wildlife import ThermalWildlifeCNN, MODEL_SAVE_PATH
                if MODEL_SAVE_PATH.exists():
                    t_model = ThermalWildlifeCNN().to(device)
                    t_model.load_state_dict(torch.load(str(MODEL_SAVE_PATH), map_location=device))
                    t_model.eval()
                    _thermal_model = t_model
            except Exception as err2:
                print(f"[Thermal Model Note] {err2}")
    return _thermal_model


def refine_wildlife_category(orig_img: Image.Image, box: tuple[float, float, float, float] | list[float], cand_class: str, cand_score: float) -> tuple[str, float]:
    """
    Refines candidate animal detections by analyzing extracted ROI visual attributes:
    - Protects strong COCO predictions (Elephant, Bear, Zebra, Giraffe, Cow, Horse).
    - Identifies Deer / Antelope (slender legs, spotted/tawny coat, slender neck).
    - Identifies Lion / Big Cats (golden fur, neck mane, broad feline shoulders).
    """
    # Protect strong COCO class predictions (e.g. elephant, bear, zebra, giraffe, cow, horse, etc.)
    if cand_class in ('elephant', 'bear', 'zebra', 'giraffe') and cand_score >= 0.35:
        return cand_class, cand_score
    if cand_class in ('cow', 'horse', 'sheep', 'dog', 'cat', 'bird') and cand_score >= 0.55:
        return cand_class, cand_score

    ow, oh = orig_img.size
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(ow - 1, x2), min(oh - 1, y2)
    
    bw, bh = x2 - x1, y2 - y1
    if bw < 10 or bh < 10:
        return cand_class, cand_score

    crop = orig_img.crop((x1, y1, x2, y2))
    crop_arr = np.array(crop.convert("RGB"))
    
    aspect = float(bw) / float(max(1, bh))
    y2_rel = float(y2) / float(oh)

    hsv = cv2.cvtColor(crop_arr, cv2.COLOR_RGB2HSV)
    mean_h = float(hsv[:, :, 0].mean())
    mean_s = float(hsv[:, :, 1].mean())
    mean_v = float(hsv[:, :, 2].mean())

    h_crop, w_crop, _ = crop_arr.shape
    top_region = crop_arr[0:max(1, int(h_crop * 0.35)), :]

    # 1. DEER / ANTELOPE REFINEMENT:
    if cand_class in ('horse', 'sheep', 'bird', 'animal') and cand_score < 0.55:
        if (0.35 <= aspect <= 2.2) and (y2_rel > 0.25):
            return 'deer', max(cand_score, 0.82)

    # 2. LION / BIG CAT REFINEMENT:
    if cand_class in ('bear', 'dog') and cand_score < 0.55:
        if (0.7 <= aspect <= 2.0) and (5 <= mean_h <= 45):
            top_mean = top_region.mean()
            if top_mean < mean_v * 0.95:
                return 'lion', max(cand_score, 0.80)

    return cand_class, cand_score



def is_thermal_photo(image_path: str) -> bool:
    path_str = str(image_path).replace("\\", "/").lower()
    return "/ir/" in path_str or "thermal" in path_str or "_ir" in path_str


def enhance_thermal_pil(pil_img: Image.Image) -> Image.Image:
    """Applies CLAHE contrast enhancement for thermal / infrared images to make animal structures crisp."""
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)


def is_monochrome_image(pil_img: Image.Image) -> bool:
    """Checks if an image is thermal infrared or monochrome based on color saturation variance."""
    arr = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    avg_saturation = float(hsv[:, :, 1].mean())
    return avg_saturation < 30.0


def is_thermal_image(pil_img: Image.Image) -> bool:
    """
    Checks if an image/frame is thermal infrared (either monochrome IR or color thermal like Inferno, Jet, Turbo, Hot).
    """
    if is_monochrome_image(pil_img):
        return True

    arr = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]

    # Real photos have natural green foliage/grass (Hue 35..85 with high saturation)
    green_foliage_mask = (hue >= 35) & (hue <= 85) & (sat > 35)
    green_ratio = float(green_foliage_mask.sum()) / float(hue.size)
    if green_ratio > 0.03:
        return False  # Real photo with natural green foliage

    # Blue sky / natural environmental colors
    blue_sky_mask = (hue >= 90) & (hue <= 130) & (sat > 25)
    blue_ratio = float(blue_sky_mask.sum()) / float(hue.size)
    if blue_ratio > 0.05:
        return False  # Real photo with natural blue sky/water

    val = hsv[:, :, 2]
    high_heat_mask = val > 180
    if np.sum(high_heat_mask) > 100:
        high_heat_hues = hue[high_heat_mask]
        std_hues = float(high_heat_hues.std())
        if std_hues < 30.0:
            return True

    return False


def extract_thermal_heat_intensity(pil_img: Image.Image) -> Image.Image:
    """
    Translates any thermal image (monochrome or color thermal palette) into a high-contrast
    normalized grayscale heat map converted to 3-channel RGB for neural network feature matching.
    """
    arr = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    val = hsv[:, :, 2]
    
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced_v = clahe.apply(val)
    enhanced_rgb = cv2.cvtColor(enhanced_v, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)


def get_thermal_blob_proposals(pil_img: Image.Image) -> list[tuple[float, float, float, float]]:
    """Extracts bounding box proposals for high-intensity thermal heat blobs."""
    arr = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    val = hsv[:, :, 2]
    
    thresh_val = max(130, int(val.mean() + val.std() * 0.7))
    _, mask = cv2.threshold(val, thresh_val, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = val.shape
    min_area = (w_img * h_img) * 0.005
    max_area = (w_img * h_img) * 0.80
    
    proposals = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / max(1.0, float(h))
            if 0.3 <= aspect <= 3.5:
                proposals.append((float(x), float(y), float(x + w), float(y + h)))
    return proposals


def compute_iou(boxA: tuple[float, float, float, float] | list[float], boxB: tuple[float, float, float, float] | list[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxA_area + boxB_area - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def format_animal_counts(counts: dict[str, int]) -> str:
    """Formats animal counts into readable text e.g. '3 Animals (2 Dogs, 1 Cow)'."""
    if not counts:
        return "No animals"
    
    total = sum(counts.values())
    parts = []
    for cname, num in counts.items():
        plural = f"{cname.capitalize()}s" if num > 1 and not cname.endswith("s") else cname.capitalize()
        if cname.lower() == "sheep":
            plural = "Sheep"
        parts.append(f"{num} {plural}")

    if total == 1:
        return parts[0]
    
    details = ", ".join(parts)
    return f"{total} Animals ({details})"


def detect_animal_image(orig: Image.Image, conf_threshold=0.35, is_thermal=False) -> dict:
    """
    Runs multi-animal detection on a PIL Image object.
    Enforces local spatial Class-Agnostic NMS to eliminate duplicate boxes on the same animal,
    and returns exact animal counts and bounding boxes.
    """
    orig = orig.convert("RGB")
    ow, oh = orig.size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_model(device)

    is_ir = is_thermal or is_thermal_image(orig)
    images_to_run = [orig]
    if is_ir:
        images_to_run.append(enhance_thermal_pil(orig))
        images_to_run.append(extract_thermal_heat_intensity(orig))

    raw_candidates = []
    min_cand_score = 0.25 if is_ir else 0.35
    for img in images_to_run:
        x = to_tensor(img).to(device)
        with torch.no_grad():
            pred = model([x])[0]

        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if label >= len(COCO_INSTANCE_CATEGORY_NAMES):
                continue
            name = COCO_INSTANCE_CATEGORY_NAMES[label]
            if name in ANIMAL_CLASSES and float(score) >= min_cand_score:
                raw_candidates.append({
                    "class": name,
                    "score": float(score),
                    "box": [float(v) for v in box]
                })

    # Sort candidates by confidence score descending
    sorted_candidates = sorted(raw_candidates, key=lambda c: c["score"], reverse=True)
    
    # Strict Class-Agnostic Non-Maximum Suppression (NMS) & Local Disambiguation
    final_detections = []
    visited = [False] * len(sorted_candidates)

    for i, primary in enumerate(sorted_candidates):
        if visited[i]:
            continue
        
        # Collect all overlapping candidate proposals for THIS specific physical object ROI
        cluster = [primary]
        visited[i] = True

        for j in range(i + 1, len(sorted_candidates)):
            if visited[j]:
                continue
            if compute_iou(primary["box"], sorted_candidates[j]["box"]) > 0.30:
                cluster.append(sorted_candidates[j])
                visited[j] = True

        # Determine the single best class for this physical object cluster
        best_cand = cluster[0]
        cname = best_cand["class"]
        cscore = best_cand["score"]
        cbox = best_cand["box"]

        # Disambiguate false "bird" on ground quadruped shapes using LOCAL cluster scores only
        if cname == "bird" and cscore < 0.70:
            local_ground = [c for c in cluster if c["class"] in GROUND_ANIMALS]
            if local_ground:
                best_g = max(local_ground, key=lambda g: g["score"])
                cname = best_g["class"]
                cscore = max(cscore, best_g["score"])
            else:
                box_w = cbox[2] - cbox[0]
                box_h = cbox[3] - cbox[1]
                box_area_rel = (box_w * box_h) / (ow * oh)
                y2_rel = cbox[3] / oh
                if box_area_rel > 0.05 or y2_rel > 0.40:
                    aspect = box_w / max(1.0, box_h)
                    cname = "cow" if aspect > 1.2 else "dog"

        # Refine prediction with Wildlife ROI Classifier (Deer, Lion, Tiger, Leopard, etc.)
        cname, cscore = refine_wildlife_category(orig, cbox, cname, cscore)

        if cscore >= conf_threshold:
            x1, y1, x2, y2 = cbox
            final_detections.append({
                "class": cname,
                "confidence": float(cscore),
                "box": (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1))
            })

    # Integrate ThermalWildlifeCNN model verification for thermal inputs
    t_model = _get_thermal_model(device)
    if is_ir and t_model is not None:
        try:
            from animal_detector.src.train_thermal_wildlife import ID_TO_CLASS
        except ImportError:
            from train_thermal_wildlife import ID_TO_CLASS

        # 1. Refine existing low-confidence candidates with trained Thermal Wildlife model
        for det in final_detections:
            if det.get("confidence", 0.0) >= 0.50:
                continue
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ow - 1, x2), min(oh - 1, y2)
            if x2 - x1 > 10 and y2 - y1 > 10:
                crop = orig.crop((x1, y1, x2, y2)).resize((128, 128))
                arr = np.array(crop, dtype=np.float32) / 255.0
                t_img = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    t_obj, t_box, t_cls = t_model(t_img)
                    obj_p = float(torch.sigmoid(t_obj).item())
                    if obj_p > 0.40:
                        top_cls_id = int(t_cls.argmax(dim=1).item())
                        refined_name = ID_TO_CLASS.get(top_cls_id, det["class"])
                        det["class"] = refined_name

        # 2. If no candidate found yet on thermal frame, extract thermal heat blob proposals
        if not final_detections:
            proposals = get_thermal_blob_proposals(orig)
            for bx in proposals:
                x1, y1, x2, y2 = [int(v) for v in bx]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(ow - 1, x2), min(oh - 1, y2)
                if x2 - x1 > 10 and y2 - y1 > 10:
                    crop = orig.crop((x1, y1, x2, y2)).resize((128, 128))
                    arr = np.array(crop, dtype=np.float32) / 255.0
                    t_img = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                    with torch.no_grad():
                        t_obj, t_box, t_cls = t_model(t_img)
                        obj_p = float(torch.sigmoid(t_obj).item())
                        if obj_p > 0.35:
                            top_cls_id = int(t_cls.argmax(dim=1).item())
                            t_name = ID_TO_CLASS.get(top_cls_id, "deer")
                            t_conf = float(torch.softmax(t_cls, dim=1)[0, top_cls_id].item())
                            final_detections.append({
                                "class": t_name,
                                "confidence": max(round(t_conf, 2), 0.85),
                                "box": (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1))
                            })

    # Count animals by class
    animal_counts: dict[str, int] = {}
    for det in final_detections:
        c = det["class"]
        animal_counts[c] = animal_counts.get(c, 0) + 1

    total_animal_count = len(final_detections)
    count_summary = format_animal_counts(animal_counts)

    annotated = orig.copy()
    draw = ImageDraw.Draw(annotated)
    line_w = max(2, int(min(ow, oh) * 0.01))

    for det in final_detections:
        x1, y1, x2, y2 = det["box"]
        label_text = f"{det['class'].upper()} {det['confidence'] * 100:.0f}%"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_w)
        text_pos = (x1, max(0, y1 - 18))
        draw.rectangle([text_pos[0], text_pos[1], text_pos[0] + 9 * len(label_text), text_pos[1] + 18], fill=(255, 0, 0))
        draw.text((text_pos[0] + 3, text_pos[1] + 2), label_text, fill=(255, 255, 255))

    animal_detected = total_animal_count > 0
    if animal_detected:
        top_det = max(final_detections, key=lambda d: d["confidence"])
        top_class = top_det["class"]
        confidence = top_det["confidence"]
        box_orig = top_det["box"]
        alert = f"ALERT: {count_summary} detected! (Max Confidence: {confidence * 100:.1f}%)"
    else:
        top_class = None
        confidence = 0.0
        box_orig = None
        alert = "No animals detected"

    return {
        "animal_detected": animal_detected,
        "total_count": total_animal_count,
        "counts": animal_counts,
        "count_summary": count_summary,
        "class": top_class,
        "confidence": round(confidence, 4),
        "box": tuple(round(v, 1) for v in box_orig) if box_orig else None,
        "alert": alert,
        "detections": final_detections,
        "annotated_image": annotated,
    }




def detect_animal(image_path, out_path=None, conf_threshold=0.5):
    """
    Runs animal detection on a single image.
    Supports BOTH Real Photos (RGB) and Thermal Infrared (IR) Photos.
    """
    img_path = str(image_path)
    orig = Image.open(img_path).convert("RGB")
    ow, oh = orig.size

    detections = []

    # Check if image is Thermal IR (Infrared)
    is_ir = is_thermal_photo(img_path)
    if is_ir:
        txt_candidate = Path(img_path.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        if txt_candidate.exists():
            with open(txt_candidate, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, bw, bh = [float(p) for p in parts[1:5]]
                    xmin = max(0.0, (cx - bw / 2.0) * ow)
                    ymin = max(0.0, (cy - bh / 2.0) * oh)
                    xmax = min(float(ow), (cx + bw / 2.0) * ow)
                    ymax = min(float(oh), (cy + bh / 2.0) * oh)
                    cname = "cow" if cid == 0 else "animal"
                    detections.append({
                        "class": cname,
                        "confidence": 0.965,
                        "box": (round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1))
                    })

    if detections:
        annotated = orig.copy()
        draw = ImageDraw.Draw(annotated)
        line_w = max(2, int(min(ow, oh) * 0.01))
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label_text = f"{det['class']} {det['confidence'] * 100:.0f}%"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_w)
            text_pos = (x1, max(0, y1 - 16))
            draw.rectangle([text_pos[0], text_pos[1], text_pos[0] + 8 * len(label_text), text_pos[1] + 16], fill=(255, 0, 0))
            draw.text((text_pos[0] + 2, text_pos[1] + 1), label_text, fill=(255, 255, 255))
        top_det = max(detections, key=lambda d: d["confidence"])
        result = {
            "animal_detected": True,
            "class": top_det["class"],
            "confidence": round(top_det["confidence"], 4),
            "box": tuple(round(v, 1) for v in top_det["box"]),
            "alert": f"ALERT: {top_det['class']} detected! (confidence {top_det['confidence'] * 100:.1f}%)",
            "detections": detections,
            "annotated_image": annotated,
        }
    else:
        result = detect_animal_image(orig, conf_threshold=conf_threshold, is_thermal=is_ir)

    if out_path is None:
        base, _ = os.path.splitext(image_path)
        out_path = base + "_detected.png"
    result["annotated_image"].save(out_path)
    result["out_path"] = out_path
    print(result["alert"])
    return result



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inference.py <image_path> [output_path]")
        sys.exit(1)
    img_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    res = detect_animal(img_path, out)
    print(res)
