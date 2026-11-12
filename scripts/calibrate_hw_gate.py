"""
scripts/calibrate_hw_gate.py
MediTwin AI — Chapter 16f: Calibrate Handwriting Gate (HW_GATE)

Runs EasyOCR on:
  (a) printed sample images (~/Pictures/meditwin_samples/meditwin_test_pack/samples/ocr_*.png)
  (b) 30 handwriting crops from the dataset's VALIDATION split

Computes box-level EasyOCR confidences and evaluates HW_GATE thresholds 0.3 to 0.9.
Recommends the lowest gate letting <= 1% of printed boxes through while keeping >= 90% of handwriting crops.

Usage:
    python scripts/calibrate_hw_gate.py
"""

import os, sys, glob
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.ocr_agent.ocr_agent import is_known_or_structured_box, _load_ocr_reader, _classes, EXTRA_MEDS


def find_validation_crops(n=30):
    cache_base = os.path.expanduser(
        "~/.cache/kagglehub/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset/versions/1"
    )
    matches = glob.glob(os.path.join(cache_base, "*", "Validation", "validation_words", "*.png"))
    if not matches:
        matches = glob.glob(os.path.join(cache_base, "*", "*", "Validation", "validation_words", "*.png"))
    return sorted(matches)[:n]


def find_printed_samples():
    pics_path = os.path.expanduser("~/Pictures/meditwin_samples/meditwin_test_pack/samples/ocr_*.png")
    imgs = sorted(glob.glob(pics_path))
    if not imgs:
        # Fallback to local logs/ocr_synthetic_images if any
        imgs = sorted(glob.glob(os.path.join(PROJECT_ROOT, "logs", "ocr_synthetic_images", "*.png")))
    return imgs


def main():
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    all_meds = set(_classes or []) | EXTRA_MEDS

    printed_imgs = find_printed_samples()
    print(f"Found {len(printed_imgs)} printed sample image(s).")
    hw_crops = find_validation_crops(30)
    print(f"Found {len(hw_crops)} validation handwriting crop(s).")

    # 1. Collect confidences on printed boxes
    print("\nScanning printed images...")
    printed_boxes_conf = []
    for p_path in printed_imgs:
        boxes = reader.readtext(p_path)
        for b in boxes:
            text = b[1]
            conf = float(b[2])
            # Only consider boxes that are not already known medicine, dose, frequency, duration, lab line, digits
            if not is_known_or_structured_box(text, all_meds):
                printed_boxes_conf.append((os.path.basename(p_path), text, conf))

    # 2. Collect confidences on handwriting crops
    print("Scanning handwriting crops...")
    hw_boxes_conf = []
    for h_path in hw_crops:
        boxes = reader.readtext(h_path)
        if boxes:
            for b in boxes:
                conf = float(b[2])
                hw_boxes_conf.append((os.path.basename(h_path), b[1], conf))
        else:
            # EasyOCR detected 0 text boxes on handwriting scribble -> effectively conf 0.0
            hw_boxes_conf.append((os.path.basename(h_path), "<no box>", 0.0))

    print(f"\nCollected {len(printed_boxes_conf)} candidate printed boxes (unknown to rules).")
    print(f"Collected {len(hw_boxes_conf)} handwriting boxes/crops.")

    print("\n" + "=" * 75)
    print(f"{'HW_GATE':<10} | {'Printed Through':<16} | {'Printed (%)':<14} | {'HW Kept':<10} | {'HW (%)':<10}")
    print("-" * 75)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rec_gate = None

    for gate in thresholds:
        # A box is classified if conf < gate
        n_p_through = sum(1 for _, _, c in printed_boxes_conf if c < gate)
        p_pct = (n_p_through / len(printed_boxes_conf) * 100.0) if printed_boxes_conf else 0.0

        n_hw_kept = sum(1 for _, _, c in hw_boxes_conf if c < gate)
        hw_pct = (n_hw_kept / len(hw_boxes_conf) * 100.0) if hw_boxes_conf else 0.0

        print(f"{gate:<10.1f} | {n_p_through:<16} | {p_pct:<14.2f} | {n_hw_kept:<10} | {hw_pct:<10.2f}")

        if p_pct <= 1.0 and hw_pct >= 90.0 and rec_gate is None:
            rec_gate = gate

    print("=" * 75)
    if rec_gate:
        print(f"Recommended HW_GATE: {rec_gate:.1f} (lets <= 1% printed through, keeps >= 90% HW crops)")
    else:
        print("Default HW_GATE: 0.5 (recommended standard)")


if __name__ == "__main__":
    main()

