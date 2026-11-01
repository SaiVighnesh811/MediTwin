"""
demo/demo_ocr.py — OCR Agent demo
Runs read_printed + extract_fields on 2 synthetic images,
and classify_handwritten_medicine on 2 test images (top-3 probs).
"""
import os, sys, time, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  MediTwin AI — OCR Agent Demo")
print("=" * 60)

from agents.ocr_agent.ocr_agent import read_printed, extract_fields, classify_handwritten_medicine

# ── 1. read_printed + extract_fields on 2 synthetic images ───────────────────
SYN_DIR  = os.path.join(PROJECT_ROOT, "logs", "ocr_synthetic_images")
syn_imgs = sorted(glob.glob(os.path.join(SYN_DIR, "*.png")))[:2]

print(f"\n  --- Printed-text OCR on {len(syn_imgs)} synthetic images ---")
for img in syn_imgs:
    t0 = time.time()
    text   = read_printed(img)
    fields = extract_fields(text)
    elapsed = time.time() - t0
    print(f"\n  Image  : {os.path.basename(img)}")
    print(f"  OCR    : {text}")
    print(f"  Fields : {fields}")
    print(f"  Time   : {elapsed:.2f}s")

# ── 2. classify_handwritten_medicine on 2 test images ────────────────────────
HW_BASE = os.path.expanduser(
    "~/.cache/kagglehub/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset"
    "/versions/1")
hw_dirs = glob.glob(os.path.join(HW_BASE, "*", "Testing", "testing_words", "*.png"))
hw_imgs = sorted(hw_dirs)[:2]

print(f"\n  --- Handwriting classifier on {len(hw_imgs)} test images ---")
if not hw_imgs:
    print("  No handwriting test images found in cache.")
else:
    # Load test labels mapping if available
    labels_csv = glob.glob(os.path.join(HW_BASE, "*", "Testing", "testing_labels.csv"))
    true_labels = {}
    if labels_csv and os.path.exists(labels_csv[0]):
        import pandas as pd
        df_lbl = pd.read_csv(labels_csv[0])
        true_labels = dict(zip(df_lbl["IMAGE"], df_lbl["MEDICINE_NAME"]))

    for img in hw_imgs:
        t0 = time.time()
        results = classify_handwritten_medicine(img)
        elapsed = time.time() - t0
        base = os.path.basename(img)
        true_lbl = true_labels.get(base, "Unknown")
        print(f"\n  Image      : {base}")
        print(f"  True Label : {true_lbl}")
        for r in results:
            print(f"    Rank {r['rank']}: {r['medicine_name']}  ({r['probability']*100:.1f}%)")
        print(f"  Time       : {elapsed:.2f}s")

print("\n  Demo complete. [demo_ocr PASS]")
