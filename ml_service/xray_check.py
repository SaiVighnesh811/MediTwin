"""
ml_service/xray_check.py
MediTwin AI — Chest X-Ray Plausibility Check (Chapter 16c)

Rejects uploaded images that are clearly not a chest X-ray
(such as colour photos, prescriptions, lab reports, landscape drawings,
or blank pages) before running heavy pediatric or adult X-ray inference.
"""
from typing import Tuple, Union
import numpy as np
from PIL import Image
from fastapi import HTTPException, status

# ── Four Named Constants ──────────────────────────────────────────────────────
SAT_MAX: float = 0.25
WHITE_MAX: float = 0.50
STD_MIN: float = 20.0
ASPECT_RANGE: Tuple[float, float] = (0.5, 2.2)

XRAY_REJECT_MESSAGE: str = (
    "This image does not look like a chest X-ray (it looks like a colour photo, "
    "a document or a blank page). Please upload a frontal chest X-ray."
)


def compute_xray_metrics(img: Image.Image) -> dict:
    """
    Computes the four plausibility metrics:
      1. mean HSV saturation (0.0 for grayscale/monochrome)
      2. fraction of near-white pixels (> 240 in grayscale)
      3. grayscale standard deviation (contrast and anatomical structure)
      4. aspect ratio (width / height)
    """
    w, h = img.size
    aspect = float(w / h) if h > 0 else 0.0

    # Grayscale image array for std and near-white fraction
    gray = np.array(img.convert("L"), dtype=np.float32)
    std = float(gray.std())
    white_frac = float((gray > 240).mean())

    # Saturation (HSV saturation channel)
    if img.mode in ("1", "L", "I"):
        mean_sat = 0.0
    else:
        rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        maxc = rgb.max(axis=-1)
        minc = rgb.min(axis=-1)
        denom = maxc + 1e-7
        sat = np.where(maxc == 0, 0, (maxc - minc) / denom)
        mean_sat = float(sat.mean())

    return {
        "sat": mean_sat,
        "white_frac": white_frac,
        "std": std,
        "aspect": aspect,
        "width": w,
        "height": h,
    }


def check_xray_plausibility(img_or_path: Union[str, Image.Image]) -> Tuple[bool, str, dict]:
    """
    Evaluates whether an image meets all chest X-ray plausibility limits.
    Returns (is_plausible, failure_reason, metrics_dict).
    """
    if isinstance(img_or_path, str):
        with Image.open(img_or_path) as img:
            metrics = compute_xray_metrics(img)
    elif isinstance(img_or_path, Image.Image):
        metrics = compute_xray_metrics(img_or_path)
    else:
        with Image.open(img_or_path) as img:
            metrics = compute_xray_metrics(img)

    reasons = []
    if metrics["sat"] > SAT_MAX:
        reasons.append(f"saturation {metrics['sat']:.3f} > {SAT_MAX}")
    if metrics["white_frac"] > WHITE_MAX:
        reasons.append(f"white_fraction {metrics['white_frac']:.3f} > {WHITE_MAX}")
    if metrics["std"] < STD_MIN:
        reasons.append(f"std {metrics['std']:.2f} < {STD_MIN}")
    if not (ASPECT_RANGE[0] <= metrics["aspect"] <= ASPECT_RANGE[1]):
        reasons.append(f"aspect {metrics['aspect']:.3f} outside {ASPECT_RANGE}")

    is_plausible = len(reasons) == 0
    return is_plausible, "; ".join(reasons), metrics


def validate_xray_plausibility(img_or_path: Union[str, Image.Image]) -> dict:
    """
    Validates X-ray plausibility. Raises HTTP 422 if the image fails any check.
    Returns computed metrics dict if valid.
    """
    is_plausible, reason, metrics = check_xray_plausibility(img_or_path)
    if not is_plausible:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": XRAY_REJECT_MESSAGE},
        )
    return metrics

