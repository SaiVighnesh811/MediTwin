"""
Configuration for MediTwin ML Service.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load environment from ml_service/.env if exists
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "127.0.0.1")
ML_DEBUG = os.getenv("ML_DEBUG", "0").lower() in ("1", "true", "yes")

# Honest caveats & disclaimers
DISCLAIMER_TEXT = "Decision-support prototype. Not a medical diagnosis."

LIMITATIONS = {
    "risk": [
        "Medium risk (prediabetes) class is weak/unreliable due to severe BRFSS class imbalance (1.8% of training data; PR-AUC=0.028).",
        "Assumes non-provided survey features using training medians unless extracted from patient documents.",
        "Educational and decision-support only; does not substitute clinical laboratory HbA1c or OGTT screening."
    ],
    "xai": [
        "SHAP explanations are computed via permutation sampling and LIME via local perturbations.",
        "Explanations reflect model behavior on BRFSS survey patterns, not causal biological truth.",
        "Features marked as assumed reflect training set defaults, not verified clinical measurements."
    ],
    "image": [
        "ResNet18 binary classifier trained on pediatric chest X-ray Kermany dataset (Guangzhou Women and Children's Medical Center).",
        "Official test set metrics: 98.5% recall, 62.0% specificity (high false positive rate on normal scans).",
        "Single-center pediatric cohort; not calibrated across adult demographics; screening aid only, not for clinical use."
    ],
    "ocr": [
        "EasyOCR printed reader evaluated on synthetic documents only; character error rate may be higher on real hospital scans.",
        "Handwriting classifier is a closed-set model restricted to 78 Bangladeshi medicine names."
    ],
    "nlp": [
        "Pretrained d4data/biomedical-ner-all (DistilBERT) model with no independent accuracy score on MediTwin data.",
        "Subword tokenization may extract partial entity stems; negation detection is heuristic-based."
    ],
    "recommendation": [
        "Recommendations are strictly rule-based educational guidance derived from CDC/AHA/WHO public guidelines.",
        "Zero diagnosis or medication prescription; physician consultation mandatory for all symptoms and abnormal findings."
    ],
    "pipeline": [
        "Full pipeline coordinates multiple independent models with partial execution fallback.",
        "Patient Memory, Recommendation, and Coordination components were tested on synthetic/retrospective records.",
        "Not validated in a prospective clinical environment."
    ]
}

