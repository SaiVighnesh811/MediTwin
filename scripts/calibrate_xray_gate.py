#!/usr/bin/env python3
"""
scripts/calibrate_xray_gate.py
MediTwin AI — Chapter 16c: X-Ray Plausibility Gate Calibration

Prints the four plausibility metrics for:
  (a) Real chest X-rays:
      - ~/Pictures/meditwin_samples/xray_*.jpeg
      - 40 random images from the Kermany chest-xray-pneumonia dataset
        under ~/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia
  (b) Non-X-ray files:
      - ~/Pictures/meditwin_samples/meditwin_test_pack/samples/ocr_*, bad_*

Recommends limits that accept every X-ray with a margin and reject every
non-X-ray, and says plainly when the two groups overlap.

Run: python scripts/calibrate_xray_gate.py
"""
import glob
import os
import random
import sys

# Add project root so we can import the xray_check module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_service.xray_check import compute_xray_metrics, SAT_MAX, WHITE_MAX, STD_MIN, ASPECT_RANGE
from PIL import Image


def collect_xray_paths():
    """Collect real chest X-ray file paths."""
    paths = []

    # Local sample X-rays
    sample_dir = os.path.expanduser("~/Pictures/meditwin_samples")
    paths.extend(sorted(glob.glob(os.path.join(sample_dir, "xray_*.jpeg"))))
    paths.extend(sorted(glob.glob(os.path.join(sample_dir, "xray_*.jpg"))))
    paths.extend(sorted(glob.glob(os.path.join(sample_dir, "xray_*.png"))))

    # Kermany dataset — walk through and collect up to 40 random images
    kermany_root = os.path.expanduser(
        "~/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia"
    )
    kermany_imgs = []
    if os.path.isdir(kermany_root):
        for root, dirs, files in os.walk(kermany_root):
            for f in files:
                if f.lower().endswith((".jpeg", ".jpg", ".png")):
                    kermany_imgs.append(os.path.join(root, f))
    if kermany_imgs:
        random.seed(42)
        selected = random.sample(kermany_imgs, min(40, len(kermany_imgs)))
        paths.extend(sorted(selected))
    else:
        print(f"[WARN] Kermany dataset not found at {kermany_root}")

    return paths


def collect_non_xray_paths():
    """Collect non-X-ray file paths."""
    paths = []
    samples_dir = os.path.expanduser(
        "~/Pictures/meditwin_samples/meditwin_test_pack/samples"
    )
    if os.path.isdir(samples_dir):
        for f in sorted(os.listdir(samples_dir)):
            if f.startswith("ocr_") or f.startswith("bad_"):
                paths.append(os.path.join(samples_dir, f))
    else:
        print(f"[WARN] Non-X-ray samples not found at {samples_dir}")
    return paths


def measure(path):
    """Returns metrics dict or None if the file can't be opened."""
    try:
        with Image.open(path) as img:
            return compute_xray_metrics(img)
    except Exception as e:
        return {"error": str(e)}


def print_table(label, paths):
    """Print metrics for a group of files."""
    print(f"\n{'=' * 80}")
    print(f"  {label}  ({len(paths)} files)")
    print(f"{'=' * 80}")
    print(f"{'File':<50} {'Sat':>6} {'WFrac':>6} {'Std':>7} {'Aspect':>7} {'WxH':>12}")
    print("-" * 96)

    metrics_list = []
    for p in paths:
        name = os.path.basename(p)
        m = measure(p)
        if m and "error" not in m:
            metrics_list.append(m)
            print(
                f"{name:<50} {m['sat']:6.3f} {m['white_frac']:6.3f} "
                f"{m['std']:7.2f} {m['aspect']:7.3f} {m['width']}x{m['height']}"
            )
        else:
            err = m.get("error", "unknown") if m else "unknown"
            print(f"{name:<50}  ** CANNOT OPEN: {err}")

    return metrics_list


def recommend(xray_metrics, non_xray_metrics):
    """Print recommended limits and overlap analysis."""
    if not xray_metrics or not non_xray_metrics:
        print("\n[WARN] Not enough data in one or both groups for recommendation.")
        return

    print(f"\n{'=' * 80}")
    print("  ANALYSIS & RECOMMENDATIONS")
    print(f"{'=' * 80}")

    # Saturation
    xr_sat_max = max(m["sat"] for m in xray_metrics)
    nr_sat_min = min(m["sat"] for m in non_xray_metrics)
    nr_sat_max = max(m["sat"] for m in non_xray_metrics)
    print(f"\nSaturation:  X-ray max = {xr_sat_max:.4f},  Non-Xray range = [{nr_sat_min:.4f}, {nr_sat_max:.4f}]")
    if xr_sat_max < nr_sat_max:
        margin_sat = (nr_sat_max - xr_sat_max) / 2
        rec_sat = xr_sat_max + margin_sat
        print(f"  → Recommend SAT_MAX = {rec_sat:.2f}  (midpoint; current = {SAT_MAX})")
    else:
        print(f"  → OVERLAP: some non-X-ray sat ({nr_sat_min:.4f}) is below X-ray max ({xr_sat_max:.4f})")

    # White fraction
    xr_wf_max = max(m["white_frac"] for m in xray_metrics)
    nr_wf_min = min(m["white_frac"] for m in non_xray_metrics)
    nr_wf_max = max(m["white_frac"] for m in non_xray_metrics)
    print(f"\nWhite frac:  X-ray max = {xr_wf_max:.4f},  Non-Xray range = [{nr_wf_min:.4f}, {nr_wf_max:.4f}]")
    if xr_wf_max < nr_wf_min:
        margin_wf = (nr_wf_min - xr_wf_max) / 2
        rec_wf = xr_wf_max + margin_wf
        print(f"  → Recommend WHITE_MAX = {rec_wf:.2f}  (midpoint of gap; current = {WHITE_MAX})")
    else:
        overlap_count = sum(1 for m in non_xray_metrics if m["white_frac"] <= xr_wf_max)
        print(f"  → OVERLAP: {overlap_count} non-X-ray(s) have white_frac ≤ X-ray max ({xr_wf_max:.4f})")

    # Std
    xr_std_min = min(m["std"] for m in xray_metrics)
    nr_std_min = min(m["std"] for m in non_xray_metrics)
    nr_std_max = max(m["std"] for m in non_xray_metrics)
    print(f"\nStd:         X-ray min = {xr_std_min:.2f},  Non-Xray range = [{nr_std_min:.2f}, {nr_std_max:.2f}]")
    if xr_std_min > nr_std_min:
        margin_std = (xr_std_min - nr_std_min) / 2
        rec_std = nr_std_min + margin_std
        print(f"  → Recommend STD_MIN = {rec_std:.1f}  (midpoint of gap; current = {STD_MIN})")
    else:
        print(f"  → OVERLAP: X-ray min std ({xr_std_min:.2f}) ≤ some non-X-ray std")

    # Aspect
    xr_asp_min = min(m["aspect"] for m in xray_metrics)
    xr_asp_max = max(m["aspect"] for m in xray_metrics)
    nr_asp_min = min(m["aspect"] for m in non_xray_metrics)
    nr_asp_max = max(m["aspect"] for m in non_xray_metrics)
    print(f"\nAspect:      X-ray range = [{xr_asp_min:.3f}, {xr_asp_max:.3f}],  Non-Xray range = [{nr_asp_min:.3f}, {nr_asp_max:.3f}]")
    overlaps = [m for m in non_xray_metrics if xr_asp_min <= m["aspect"] <= xr_asp_max]
    if overlaps:
        print(f"  → OVERLAP: {len(overlaps)} non-X-ray(s) fall within X-ray aspect range")
    else:
        lo = xr_asp_min * 0.9
        hi = xr_asp_max * 1.1
        print(f"  → Recommend ASPECT_RANGE = ({lo:.2f}, {hi:.2f})  (10% margin; current = {ASPECT_RANGE})")

    # Overall: which non-X-rays would pass ALL current checks?
    print(f"\n--- Overall with current defaults ---")
    print(f"  SAT_MAX={SAT_MAX}, WHITE_MAX={WHITE_MAX}, STD_MIN={STD_MIN}, ASPECT_RANGE={ASPECT_RANGE}")
    for m in non_xray_metrics:
        passes = (
            m["sat"] <= SAT_MAX
            and m["white_frac"] <= WHITE_MAX
            and m["std"] >= STD_MIN
            and ASPECT_RANGE[0] <= m["aspect"] <= ASPECT_RANGE[1]
        )
        if passes:
            print(f"  ⚠ PASSES ALL CHECKS: sat={m['sat']:.3f} wf={m['white_frac']:.3f} std={m['std']:.2f} asp={m['aspect']:.3f} ({m['width']}x{m['height']})")

    for m in xray_metrics:
        fails_any = (
            m["sat"] > SAT_MAX
            or m["white_frac"] > WHITE_MAX
            or m["std"] < STD_MIN
            or not (ASPECT_RANGE[0] <= m["aspect"] <= ASPECT_RANGE[1])
        )
        if fails_any:
            reasons = []
            if m["sat"] > SAT_MAX:
                reasons.append(f"sat={m['sat']:.3f}")
            if m["white_frac"] > WHITE_MAX:
                reasons.append(f"wf={m['white_frac']:.3f}")
            if m["std"] < STD_MIN:
                reasons.append(f"std={m['std']:.2f}")
            if not (ASPECT_RANGE[0] <= m["aspect"] <= ASPECT_RANGE[1]):
                reasons.append(f"asp={m['aspect']:.3f}")
            print(f"  ⚠ X-RAY REJECTED: {', '.join(reasons)} ({m['width']}x{m['height']})")

    print()


if __name__ == "__main__":
    xray_paths = collect_xray_paths()
    non_xray_paths = collect_non_xray_paths()

    xray_m = print_table("CHEST X-RAYS", xray_paths)
    non_xray_m = print_table("NON-X-RAY FILES", non_xray_paths)

    recommend(xray_m, non_xray_m)

