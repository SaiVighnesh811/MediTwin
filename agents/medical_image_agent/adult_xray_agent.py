"""
agents/medical_image_agent/adult_xray_agent.py
Adult chest X-ray screener using TorchXRayVision DenseNet121-res224-all.

DISCLAIMER: Screening aid only. Performance not independently measured in this project.
Training data overlaps every large public adult CXR dataset, so independent
accuracy cannot be measured here. Scores are op-normalized model outputs, not
calibrated probabilities.

License of base model weights: Apache-2.0
  Source: torchxrayvision model card / Hugging Face hub entry for densenet121-res224-all
  Papers: Cohen et al., arXiv:2111.00595 and arXiv:2002.02497
"""
import os
import sys
import time

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.inference_lock import INFERENCE_LOCK  # same lock used by pediatric agent

MODEL_NAME = "densenet121-res224-all"
POPULATION = (
    "Adult chest X-rays — training data: NIH, PadChest, CheXpert, "
    "MIMIC-CXR, Google, OpenI, RSNA"
)
MODEL_LABEL = "adult_torchxrayvision_densenet121_all"

# Pneumonia op-threshold from model_urls['all']['op_threshs'][8]
# Source: torchxrayvision/models.py, model_urls dict, line ~25
PNEUMONIA_OP_THRESH = 0.056810737

_model = None
_targets = None


def _load_model() -> None:
    """Lazy-load the DenseNet121 weights (cached after first call)."""
    global _model, _targets
    if _model is not None:
        return
    try:
        import torchxrayvision as xrv  # noqa: F401 — verify import
    except ImportError as exc:
        raise ImportError(
            "torchxrayvision is not installed. "
            "Activate the project venv before importing this module."
        ) from exc

    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights=MODEL_NAME)
    model.eval()
    _model = model
    _targets = model.targets  # list of 18 label strings


def _preprocess(image_path: str) -> torch.Tensor:
    """
    Preprocessing per torchxrayvision source (datasets.py, utils.py):

    1. Load as grayscale 2D uint8 array via PIL.
    2. xrv.datasets.normalize(img, 255, reshape=True)
         → shape (1, H, W), dtype float32, range ~[-1024, 1024]
         Source: utils.py normalize(), reshape=True extracts img[:,:,0] if 3D,
                 then adds channel dim.
    3. XRayCenterCrop() → center-square crop, shape (1, N, N).
         Source: datasets.py XRayCenterCrop.__call__, expects (1, H, W).
    4. XRayResizer(224) → shape (1, 224, 224).
         Source: datasets.py XRayResizer, skimage resize.
    5. torch.from_numpy().unsqueeze(0) → shape (1, 1, 224, 224).
    """
    import torchxrayvision as xrv

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        pil_img = Image.open(image_path).convert("L")  # grayscale 2D
        img_uint8 = np.array(pil_img)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot read image file: {exc}") from exc

    img = xrv.datasets.normalize(img_uint8, 255, reshape=True)   # (1, H, W) float32
    img = xrv.datasets.XRayCenterCrop()(img)                      # (1, N, N)
    img = xrv.datasets.XRayResizer(224)(img)                      # (1, 224, 224)
    tensor = torch.from_numpy(img).unsqueeze(0)                   # (1, 1, 224, 224)
    return tensor


def predict_adult_xray(image_path: str) -> dict:
    """
    Run adult chest X-ray inference with TorchXRayVision DenseNet121-res224-all.

    Returns a plain dict with:
      model_name, model_used, population,
      pneumonia_score, pneumonia_op_thresh, pneumonia_op_thresh_source,
      related_findings (Consolidation, Lung Opacity, Infiltration),
      all_scores (all 18 label → score),
      output_range_semantics,
      inference_time_seconds,
      limitations.

    Output semantics (verified from torchxrayvision/models.py forward(), lines 329-361):
      The model applies sigmoid then op_norm() so the operating threshold maps to 0.5.
      Outputs are op-normalized probabilities, NOT raw logits.
    """
    _load_model()
    t0 = time.time()
    tensor = _preprocess(image_path)

    with INFERENCE_LOCK:
        with torch.no_grad():
            out = _model(tensor)   # shape (1, 18); op_norm applied internally

    scores = out[0].cpu().numpy()
    targets = _targets

    def _score(name: str):
        try:
            idx = targets.index(name)
            return round(float(scores[idx]), 6)
        except ValueError:
            return None

    pneumonia_score = _score("Pneumonia")
    inference_time = round(time.time() - t0, 3)

    all_scores = {label: round(float(scores[i]), 6) for i, label in enumerate(targets)}

    return {
        "model_name": MODEL_NAME,
        "model_used": MODEL_LABEL,
        "population": POPULATION,
        "pneumonia_score": pneumonia_score,
        "pneumonia_op_thresh": PNEUMONIA_OP_THRESH,
        "pneumonia_op_thresh_source": (
            "model_urls['all']['op_threshs'][8] in "
            "torchxrayvision/models.py (verified from installed package source)"
        ),
        "related_findings": {
            "Consolidation": _score("Consolidation"),
            "Lung Opacity": _score("Lung Opacity"),
            "Infiltration": _score("Infiltration"),
        },
        "all_scores": all_scores,
        "output_range_semantics": (
            "Op-normalized probabilities. The model applies sigmoid then rescales "
            "outputs so that the operating threshold maps to 0.5. "
            "Source: torchxrayvision/models.py forward() lines 329-361."
        ),
        "inference_time_seconds": inference_time,
        "limitations": [
            "Pretrained public model; no additional fine-tuning was performed in this project.",
            (
                "Training data (NIH, PadChest, CheXpert, MIMIC-CXR, Google, OpenI, RSNA) "
                "overlaps every large public adult chest X-ray dataset; "
                "independent accuracy cannot be measured in this project."
            ),
            "Scores are op-normalized model outputs, not calibrated clinical probabilities.",
            (
                "Screening aid only — not a diagnosis. "
                "A licensed radiologist must interpret all results."
            ),
            (
                "Image quality, projection (PA vs AP), and patient positioning "
                "affect model outputs."
            ),
            "Performance of this model was not independently measured in this project.",
            "A simple image check cannot recognise every non-chest image (for example an X-ray of another body part).",
        ],
    }

