"""
test_detection.py
------------------
Smoke test and validation script for animal detection.
Tests both PyTorch AI model and NumPy model on real and synthetic images.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "animal_detector" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import detect_animal


def test_samples():
    samples = [
        ROOT_DIR / "animal_detector" / "samples" / "Final_Dataset" / "train" / "cow" / "rgb" / "images" / "cow_0001.jpg",
        ROOT_DIR / "animal_detector" / "samples" / "Final_Dataset" / "train" / "cow" / "rgb" / "images" / "cow_0005.jpg",
        ROOT_DIR / "animal_detector" / "samples" / "Final_Dataset" / "train" / "non_cow" / "rgb" / "images" / "non_cow_0001.jpg",
    ]

    print("=" * 60)
    print("RUNNING ANIMAL DETECTION TESTS (PyTorch AI Engine)")
    print("=" * 60)

    for img_path in samples:
        if not img_path.exists():
            print(f"Skipping missing sample: {img_path.name}")
            continue
        print(f"\nTesting: {img_path.name}")
        res = detect_animal(img_path, model_type="pytorch")
        print(f"  Engine: {res.get('engine')}")
        print(f"  Detected: {res.get('animal_detected')}")
        print(f"  Top Class: {res.get('class')}")
        print(f"  Confidence: {res.get('confidence') * 100:.1f}%")
        print(f"  Bounding Box: {res.get('box')}")
        print(f"  Alert: {res.get('alert')}")
        if res.get("detections"):
            print(f"  All Detections: {res.get('detections')}")
        print("-" * 50)


if __name__ == "__main__":
    test_samples()
