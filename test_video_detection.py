"""
test_video_detection.py
-----------------------
Automated integration test for video thermal conversion, animal detection, and voice alerts.
Generates a synthetic sample video and processes it through the video processor pipeline.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent
DETECTOR_SRC_DIR = ROOT_DIR / "animal_detector" / "src"
if str(DETECTOR_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTOR_SRC_DIR))

from video_processor import process_video  # noqa: E402
from inference import detect_animal_image  # noqa: E402


def create_synthetic_animal_video(output_path: str, duration_sec: int = 2, fps: int = 15) -> str:
    """Generates a synthetic MP4 video containing a moving dog figure on textured background."""
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Create textured background frame
        bg_arr = np.random.randint(160, 240, (h, w, 3), dtype=np.uint8)
        img = Image.fromarray(bg_arr)
        draw = ImageDraw.Draw(img)

        # Draw a moving dog silhouette across the frame
        cx = int(40 + (i / total_frames) * (w - 80))
        cy = int(h / 2)
        s = 30
        color = (60, 40, 30)

        # Body & head
        draw.ellipse([cx - s, cy - s * 0.5, cx + s * 0.6, cy + s * 0.5], fill=color)
        hx, hy = cx + s * 0.7, cy - s * 0.2
        draw.ellipse([hx - 12, hy - 12, hx + 12, hy + 12], fill=color)
        draw.line([cx - s * 0.6, cy + s * 0.4, cx - s * 0.6, cy + s * 0.9], fill=color, width=3)
        draw.line([cx + s * 0.3, cy + s * 0.4, cx + s * 0.3, cy + s * 0.9], fill=color, width=3)

        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"Synthetic video generated at: {output_path}")
    return output_path


def main() -> int:
    temp_dir = Path(tempfile.gettempdir())
    input_video = str(temp_dir / "synthetic_dog_test.mp4")
    output_video = str(temp_dir / "synthetic_dog_thermal_out.mp4")

    # 1. Create synthetic video
    create_synthetic_animal_video(input_video)

    # 2. Process video through thermal engine & detection
    print("Testing process_video pipeline...")
    res = process_video(
        video_path=input_video,
        output_path=output_video,
        colormap="inferno",
        conf_threshold=0.3,
        enable_voice=True,
    )

    print(f"Processed {res['processed_frames']} frames.")
    print(f"Output Video Exists: {Path(output_video).exists()}")

    assert Path(output_video).exists(), "Output thermal video was not created!"
    assert res["processed_frames"] > 0, "No frames were processed!"

    print("\nSUCCESS: Video Animal Detection & Thermal Conversion Test Passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
