"""
video_processor.py
------------------
Processes video streams and files for animal detection:
- Converts video frames into Thermal Infrared (IR) heatmaps (Inferno, Jet, Turbo, Ironbow).
- Detects animals on video frames using PyTorch Faster R-CNN.
- Triggers non-blocking Voice Alert announcements ("Warning! [Animal] is coming!").
- Renders thermal bounding boxes and alert overlays.
- Exports annotated thermal video files.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Any

import cv2
import numpy as np
from PIL import Image

# Ensure inference module is importable
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DETECTOR_SRC_DIR = ROOT_DIR / "animal_detector" / "src"
if str(DETECTOR_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTOR_SRC_DIR))

from inference import detect_animal_image  # noqa: E402


COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "bone": cv2.COLORMAP_BONE,
    "hot": cv2.COLORMAP_HOT,
    "magma": cv2.COLORMAP_MAGMA,
}

_last_spoken_time = 0.0
_last_spoken_animal = ""
_speech_lock = threading.Lock()


def reset_speech_state() -> None:
    """Resets the text-to-speech throttling state for a new video or image detection run."""
    global _last_spoken_time, _last_spoken_animal
    with _speech_lock:
        _last_spoken_time = 0.0
        _last_spoken_animal = ""


def speak_animal_warning(summary_text: str, min_interval_sec: float = 3.5) -> None:
    """Non-blocking text-to-speech warning announcement using Windows SAPI voice synthesizer."""
    global _last_spoken_time, _last_spoken_animal

    now = time.time()
    with _speech_lock:
        if summary_text == _last_spoken_animal and (now - _last_spoken_time) < min_interval_sec:
            return
        _last_spoken_time = now
        _last_spoken_animal = summary_text

    def _worker():
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            text = f"Warning! {summary_text} is approaching! Please be alert!"
            speaker.Speak(text)
        except Exception as err:
            print(f"[Voice Alert Error] {err}")

    threading.Thread(target=_worker, daemon=True).start()


def is_already_thermal_bgr(frame_bgr: np.ndarray) -> bool:
    """Detects if an input video frame is ALREADY a thermal infrared image (monochrome or color palette)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(hsv[:, :, 1].mean())
    if sat_mean < 30.0:
        return True
    high_heat = hsv[:, :, 2] > 170
    if np.sum(high_heat) > 40:
        hues = hsv[:, :, 0][high_heat]
        if float(hues.std()) < 45.0:
            return True
    return False


def convert_frame_to_thermal(
    frame_bgr: np.ndarray,
    colormap_name: str = "inferno",
    contrast_boost: float = 1.25,
    boxes: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """
    Converts a standard BGR image frame to a thermal infrared heatmap.
    If the frame is ALREADY a thermal video, enhances contrast while preserving native thermal colors.
    """
    if is_already_thermal_bgr(frame_bgr):
        # Enhance native thermal frame contrast without re-applying colormap
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast stretch / equalization for heat signature contrast
    enhanced = cv2.convertScaleAbs(gray, alpha=contrast_boost, beta=10)
    
    # If detected boxes are present, boost thermal intensity inside the ROI
    if boxes:
        thermal_mask = enhanced.copy()
        for (x1, y1, x2, y2) in boxes:
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(frame_bgr.shape[1], int(x2)), min(frame_bgr.shape[0], int(y2))
            if ix2 > ix1 and iy2 > iy1:
                # Boost ROI heat intensity
                roi = thermal_mask[iy1:iy2, ix1:ix2]
                thermal_mask[iy1:iy2, ix1:ix2] = cv2.addWeighted(roi, 1.3, np.ones_like(roi) * 30, 0, 0)
        enhanced = thermal_mask

    cmap_code = COLORMAPS.get(colormap_name.lower(), cv2.COLORMAP_INFERNO)
    thermal_bgr = cv2.applyColorMap(enhanced, cmap_code)
    return thermal_bgr


def draw_thermal_overlay(
    thermal_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    frame_index: int = 0,
    fps: float = 30.0,
) -> tuple[np.ndarray, str | None, dict[str, int]]:
    """
    Draws bounding boxes, thermal target reticles, and warning banner overlays on frame.
    Returns (annotated_frame, top_animal_class, animal_counts_dict).
    """
    annotated = thermal_bgr.copy()
    h, w, _ = annotated.shape
    top_animal = None
    max_conf = 0.0
    counts: dict[str, int] = {}

    for det in detections:
        cname = str(det.get("class", "animal")).lower()
        conf = float(det.get("confidence", 0.0))
        box = det.get("box")
        if not box:
            continue

        counts[cname] = counts.get(cname, 0) + 1

        if conf > max_conf:
            max_conf = conf
            top_animal = cname

        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Draw red thermal alert bounding box
        box_color = (0, 0, 255)  # Bright Red
        thickness = max(2, int(min(w, h) * 0.006))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        # Draw target corner accents for thermal HUD look
        corner_len = max(10, int(min(x2 - x1, y2 - y1) * 0.2))
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), thickness + 1)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), thickness + 1)
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), thickness + 1)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), thickness + 1)

        # Label tag
        label_text = f"HOT TARGET: {cname.upper()} {conf * 100:.0f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(text_h + 8, y1 - 8)
        cv2.rectangle(annotated, (x1, label_y - text_h - 6), (x1 + text_w + 10, label_y + baseline), (0, 0, 200), -1)
        cv2.putText(annotated, label_text, (x1 + 5, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Top Warning Header Banner
    banner_height = 42
    banner_bg = np.zeros((banner_height, w, 3), dtype=np.uint8)
    
    total_count = sum(counts.values())
    if total_count > 0:
        # Red warning banner with animal count
        banner_bg[:] = (20, 20, 180)
        breakdown_str = ", ".join(f"{num} {c.upper()}" for c, num in counts.items())
        warning_msg = f"ALERT: {total_count} ANIMAL(S) APPROACHING! ({breakdown_str})"
        icon = "[!!]"
    else:
        # Dark status banner
        banner_bg[:] = (30, 30, 30)
        warning_msg = "THERMAL CAMERA STREAMING · SCANNING FOR ANIMALS..."
        icon = "[OK]"

    cv2.putText(banner_bg, f"{icon} {warning_msg}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Merge top banner with frame
    if h > banner_height:
        annotated[0:banner_height, 0:w] = cv2.addWeighted(annotated[0:banner_height, 0:w], 0.2, banner_bg, 0.8, 0)

    return annotated, top_animal, counts



from collections import deque, Counter

def process_video(
    video_path: str,
    output_path: str | None = None,
    colormap: str = "inferno",
    conf_threshold: float = 0.35,
    sample_stride: int = 1,
    enable_voice: bool = True,
    frame_callback: Callable[[np.ndarray, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Processes an input video file:
    - Extracts frames.
    - Converts frames to Thermal Imaging Heatmap.
    - Runs animal detection with temporal voting across frames.
    - Triggers voice warnings for persistent detected animals.
    - Exports thermal annotated output video file.
    """
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        output_path = str(path.parent / f"{path.stem}_thermal_detected.mp4")

    # Output Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detected_animals: set[str] = set()
    total_detections_count = 0
    max_confidence = 0.0
    top_detected_class = None
    frame_index = 0

    last_detections: list[dict[str, Any]] = []
    detection_window: deque[tuple[str, float]] = deque(maxlen=15)

    reset_speech_state()

    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            frame_index += 1

            # Run detection on sampled frames for high speed performance
            if frame_index == 1 or frame_index % sample_stride == 0:
                # Convert BGR frame to PIL RGB image for PyTorch Faster R-CNN
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                res = detect_animal_image(pil_img, conf_threshold=conf_threshold)
                last_detections = res.get("detections", [])
                if last_detections:
                    top_f = max(last_detections, key=lambda d: d.get("confidence", 0.0))
                    detection_window.append((str(top_f["class"]), float(top_f["confidence"])))
                else:
                    detection_window.append(("none", 0.0))
            
            boxes = [d["box"] for d in last_detections if "box" in d]
            
            # Temporal Consensus Animal Determination across video frames
            temporal_animal = None
            if detection_window:
                valid_entries = [item for item in detection_window if item[0] != "none"]
                if valid_entries:
                    # Weight votes by confidence & quadruped priority
                    class_scores: dict[str, float] = {}
                    for cname, score in valid_entries:
                        weight = score * (1.5 if cname != "bird" else 1.0)
                        class_scores[cname] = class_scores.get(cname, 0.0) + weight
                    temporal_animal = max(class_scores, key=class_scores.get)

            # 1. Convert frame to Thermal Infrared Heatmap
            thermal_bgr = convert_frame_to_thermal(frame_bgr, colormap_name=colormap, boxes=boxes)

            # 2. Render overlays & warning alerts
            annotated_thermal, top_animal, frame_counts = draw_thermal_overlay(
                thermal_bgr, last_detections, frame_index=frame_index, fps=fps
            )

            current_animal = temporal_animal or top_animal
            total_current_count = sum(frame_counts.values())

            if total_current_count > 0 and current_animal:
                top_animal_str = str(current_animal)
                detected_animals.add(top_animal_str)
                total_detections_count += total_current_count
                for d in last_detections:
                    if d.get("confidence", 0.0) > max_confidence:
                        max_confidence = d.get("confidence", 0.0)
                        top_detected_class = top_animal_str

                # Format spoken summary for count
                count_str = ", ".join(f"{num} {c}" for c, num in frame_counts.items())
                spoken_summary = f"{total_current_count} animal{'s' if total_current_count > 1 else ''}: {count_str}"

                # Trigger non-blocking Voice Alert
                if enable_voice:
                    speak_animal_warning(spoken_summary)

            writer.write(annotated_thermal)

            if frame_callback:
                status = frame_callback(annotated_thermal, {
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                    "top_animal": current_animal,
                    "detections": last_detections,
                    "counts": frame_counts,
                    "total_count": total_current_count,
                })
                if status is False:
                    break

    finally:
        cap.release()
        writer.release()


    return {
        "video_path": str(path),
        "output_path": output_path,
        "total_frames": total_frames,
        "processed_frames": frame_index,
        "detected_animals": list(detected_animals),
        "animal_detected": len(detected_animals) > 0,
        "top_class": top_detected_class,
        "max_confidence": round(max_confidence, 4),
    }
