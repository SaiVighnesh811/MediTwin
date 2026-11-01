"""
agents/medical_image_agent/medical_image_agent.py
MediTwin AI — Chapter 3: Medical Image Agent

DISCLAIMER: This module is a screening aid only. It is NOT a diagnostic tool
and must NOT be used for clinical decision-making without validation by a
licensed medical professional.

Pretrained Model:
    ResNet18 binary classifier from:
    https://huggingface.co/luanacarolina/pneumonia-chest-xray-classifier
    Checkpoints: checkpoints/binary/resnet18.pt (CC BY 4.0)

Public API:
    predict_image(image_path, save_heatmap=True) -> dict
        Returns predicted class, probabilities, threshold, heatmap path, and disclaimer.
"""

import os, sys
import warnings; warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from utils.inference_lock import INFERENCE_LOCK

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "medical_image")
LOGS_DIR    = os.path.join(PROJECT_ROOT, "logs")
CKPT_PATH   = os.path.join(MODELS_DIR, "resnet18_xray_weights.pt")
ORIG_CKPT_PATH = os.path.join(MODELS_DIR, "resnet18_xray.pt")

DEVICE      = torch.device("cpu")
IMG_SIZE    = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Module-level cached model (lazy singleton) ─────────────────────────────────
_model     = None
_threshold = 0.5


def _load_model():
    """Load model once at runtime on CPU using weights_only=True."""
    global _model, _threshold
    if _model is not None:
        return
    target_path = CKPT_PATH if os.path.exists(CKPT_PATH) else ORIG_CKPT_PATH
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Model weights not found at {target_path}. "
            "Download resnet18.pt from Hugging Face luanacarolina/pneumonia-chest-xray-classifier first."
        )
    try:
        data = torch.load(target_path, map_location=DEVICE, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint safely with weights_only=True: {e}")

    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, 1)
    if isinstance(data, dict) and "model_state_dict" in data:
        m.load_state_dict(data["model_state_dict"])
    else:
        m.load_state_dict(data)
    m.eval()
    m.to(DEVICE)
    _model = m
    _threshold = 0.5


def _preprocess(image_path: str):
    """Validate and preprocess image according to model card specs."""
    if not image_path or not isinstance(image_path, str):
        raise ValueError(f"Invalid image path: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    try:
        img = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Cannot read image file {image_path}: {e}")

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(img), img


class _GradCAM:
    """Single-use Grad-CAM on model.layer4[-1] for single-logit binary classification."""
    def __init__(self, model):
        self.acts  = None
        self.grads = None
        self.h1 = model.layer4[-1].register_forward_hook(self._save_acts)
        self.h2 = model.layer4[-1].register_full_backward_hook(self._save_grads)

    def _save_acts(self, _, __, output):
        self.acts = output.detach()

    def _save_grads(self, _, __, grad_output):
        self.grads = grad_output[0].detach()

    def compute(self, img_tensor: torch.Tensor, model) -> np.ndarray:
        try:
            model.eval()
            x = img_tensor.clone().requires_grad_(True)
            out = model(x)
            model.zero_grad()
            out[0, 0].backward()
            w = self.grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * self.acts).sum(dim=1, keepdim=True))
            cam = cam.squeeze().cpu().numpy()
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            return cam
        finally:
            self.h1.remove()
            self.h2.remove()


def predict_image(image_path: str, save_heatmap: bool = True) -> dict:
    """
    Predict Normal vs Pneumonia for a chest X-ray image.

    Args:
        image_path:    Absolute path to the image file (.jpeg / .png / .jpg).
        save_heatmap:  If True, save a Grad-CAM overlay PNG to logs/.

    Returns:
        dict with keys:
            predicted_class       : "NORMAL" or "PNEUMONIA"
            probability           : float, P(PNEUMONIA)
            normal_probability    : float, P(NORMAL)
            pneumonia_probability : float, P(PNEUMONIA)
            threshold_used        : float
            heatmap_path          : str or None
            disclaimer            : str

    DISCLAIMER: Screening aid only. Not validated for clinical use.
    """
    _load_model()
    img_tensor, img_pil = _preprocess(image_path)

    with INFERENCE_LOCK:
        with torch.inference_mode():
            logit = _model(img_tensor.unsqueeze(0).to(DEVICE)).item()
            prob_pneu = float(torch.sigmoid(torch.tensor(logit)).item())
            prob_norm = float(1.0 - prob_pneu)

        pred_idx   = int(prob_pneu >= _threshold)
        pred_class = CLASS_NAMES[pred_idx]

        heatmap_path = None
        if save_heatmap:
            try:
                gcam    = _GradCAM(_model)
                cam_map = gcam.compute(img_tensor.unsqueeze(0).to(DEVICE), _model)

                cam_r = np.array(
                    Image.fromarray(np.uint8(255 * cam_map))
                    .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                ) / 255.0
                heatmap  = mpl_cm.jet(cam_r)[:, :, :3]
                img_np   = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(float) / 255.0
                overlay  = np.clip(0.55 * img_np + 0.45 * heatmap, 0, 1)

                fig, axes = plt.subplots(1, 2, figsize=(7, 3))
                axes[0].imshow(img_np);  axes[0].set_title("Input");   axes[0].axis("off")
                axes[1].imshow(overlay); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
                fig.suptitle(
                    f"pred={pred_class}  P(Pneumonia)={prob_pneu:.3f}  "
                    f"threshold={_threshold:.2f}", fontsize=9
                )
                plt.tight_layout()
                os.makedirs(LOGS_DIR, exist_ok=True)
                fname = os.path.basename(image_path).rsplit(".", 1)[0]
                heatmap_path = os.path.join(LOGS_DIR, f"gradcam_api_{fname}.png")
                plt.savefig(heatmap_path, dpi=100)
                plt.close(fig)
            except Exception as e:
                warnings.warn(f"Grad-CAM generation failed: {e}")
                heatmap_path = None

    return {
        "predicted_class":       pred_class,
        "probability":           round(prob_pneu, 4),
        "normal_probability":    round(prob_norm, 4),
        "pneumonia_probability": round(prob_pneu, 4),
        "threshold_used":        _threshold,
        "heatmap_path":          heatmap_path,
        "disclaimer": (
            "SCREENING AID ONLY. This is a research prototype pretrained on single-hospital "
            "pediatric chest X-rays (Guangzhou Women and Children's Medical Center; Luana Carolina Reis & "
            "Jakub Błaszczyk, CC BY 4.0). It has NOT been validated for clinical use. "
            "A licensed physician must interpret all results."
        ),
    }


# ── Quick self-test when run directly ─────────────────────────────────────────
if __name__ == "__main__":
    import glob
    DATA_ROOT = ("/home/sai/.cache/kagglehub/datasets/paultimothymooney"
                 "/chest-xray-pneumonia/versions/2/chest_xray")
    test_imgs = glob.glob(os.path.join(DATA_ROOT, "test", "NORMAL", "*.jpeg"))
    if not test_imgs:
        print("ERROR: No test images found. Check DATA_ROOT.")
        sys.exit(1)

    img_path = test_imgs[0]
    print(f"Testing on: {os.path.basename(img_path)}")
    result = predict_image(img_path, save_heatmap=True)
    print(f"  Predicted class       : {result['predicted_class']}")
    print(f"  P(Pneumonia)          : {result['pneumonia_probability']}")
    print(f"  P(Normal)             : {result['normal_probability']}")
    print(f"  Threshold used        : {result['threshold_used']}")
    print(f"  Heatmap saved         : {result['heatmap_path']}")
    print(f"  Disclaimer            : {result['disclaimer']}")
