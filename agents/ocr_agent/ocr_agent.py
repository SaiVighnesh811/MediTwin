"""
agents/ocr_agent/ocr_agent.py
MediTwin AI — Chapter 4: OCR Agent   Public API

    read_printed(image_path)                 → OCR text from printed/typed image
    classify_handwritten_medicine(image_path) → top-3 medicine names + probs
    extract_fields(text)                     → structured fields from OCR text

DISCLAIMER: Closed-set classifier (78 BD medicine names only).
            EasyOCR on synthetic/printed images only — not validated on real scans.
"""

import os, sys, re, json
import warnings; warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "ocr")
CKPT_PATH  = os.path.join(MODELS_DIR, "handwriting_cnn.pt")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Module-level cached objects ────────────────────────────────────────────────
_hw_model   = None
_classes    = None
_ocr_reader = None

IMG_SIZE    = 128   # default; overridden from checkpoint if available

# ── Known medicine names (78 from dataset + common extras) ────────────────────
EXTRA_MEDS = {
    "Metformin","Amoxicillin","Paracetamol","Amlodipine","Insulin Glargine",
    "Atorvastatin","Azithromycin","Omeprazole","Fexofenadine","Clonazepam",
    "Cetirizine","Metronidazole","Fluconazole","Baclofen","Esomeprazole",
}

# ── Regex patterns ─────────────────────────────────────────────────────────────
DOSAGE_RE   = re.compile(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|iu)\b', re.IGNORECASE)
FREQ_TERMS  = ["once daily","twice daily","three times daily","thrice daily",
               "every 6 hours","every 8 hours","at bedtime","at night","before meals",
               "before breakfast","single dose","once"]
DURATION_RE = re.compile(r'(?:for\s+)?(\d+)\s+(day|days|month|months|week|weeks)', re.IGNORECASE)
TEST_RE     = re.compile(
    r'^(HbA1c|Fasting Blood Glucose|Total Cholesterol|Haemoglobin|Serum Creatinine|'
    r'ALT SGPT|TSH|Platelet Count|Urea|WBC|LDL Cholesterol|HDL Cholesterol|'
    r'Uric Acid|Sodium|Potassium)\b', re.IGNORECASE)
VALUE_RE    = re.compile(r'(\d+(?:\.\d+)?(?:\s*[-\/]\s*\d+(?:\.\d+)?)?)\s*([a-zA-Z/%_]+(?:/[a-zA-Z]+)?)?')
REFRANGE_RE = re.compile(
    r'(?:Ref(?:erence)?\s*|less than |greater than |>|<)(\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)',
    re.IGNORECASE)


def _load_hw_model():
    global _hw_model, _classes
    if _hw_model is not None:
        return
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Handwriting model not found at {CKPT_PATH}. "
            "Run agents/ocr_agent/train.py first."
        )
    ckpt     = torch.load(CKPT_PATH, map_location=DEVICE)
    _classes = ckpt["classes"]                    # list of 78 medicine names
    n        = len(_classes)
    m        = models.resnet18(weights=None)
    m.fc     = nn.Linear(m.fc.in_features, n)
    m.load_state_dict(ckpt.get("best_state_dict") or ckpt["model_state_dict"])
    m.eval(); m.to(DEVICE)
    _hw_model = m


def _load_ocr_reader():
    global _ocr_reader
    if _ocr_reader is not None:
        return
    import easyocr
    _ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)


# ──────────────────────────────────────────────────────────────────────────────
def read_printed(image_path: str) -> str:
    """
    Run EasyOCR on a printed/typed image and return the concatenated text.
    EasyOCR is a pretrained model — no fine-tuning performed.
    NOTE: Validated on synthetic PIL images only; not tested on real scans.
    """
    _load_ocr_reader()
    results = _ocr_reader.readtext(image_path, detail=0, paragraph=True)
    return " ".join(results).strip()


def classify_handwritten_medicine(image_path: str) -> list[dict]:
    """
    Classify a handwritten medicine word image.
    Returns top-3 predictions with probabilities.
    CLOSED SET: only the 78 BD medicine names from the training dataset.
    """
    _load_hw_model()
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img  = Image.open(image_path).convert("RGB")
    x    = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(_hw_model(x), dim=1)[0].cpu().numpy()
    top3_idx = np.argsort(probs)[::-1][:3]
    return [
        {"rank": i+1, "medicine_name": _classes[idx], "probability": round(float(probs[idx]), 4)}
        for i, idx in enumerate(top3_idx)
    ]


def extract_fields(text: str) -> dict:
    """
    Extract structured fields from OCR text.
    Tries prescription fields first, then lab-report fields.
    Returns a dict with type ('prescription'|'lab'|'unknown') and extracted fields.
    """
    tl = text.lower()

    # Determine if prescription or lab
    is_lab = bool(TEST_RE.search(text))
    is_rx  = any(kw in tl for kw in ["tab","cap","syp","inj","once","daily","mg","ml"])

    if is_lab and not is_rx:
        tm  = TEST_RE.search(text)
        test_end = len(tm.group(0).split()) if tm else 0
        rest = " ".join(text.split()[test_end:])
        vm  = VALUE_RE.search(rest)
        rm  = REFRANGE_RE.search(text)
        ref = rm.group(1).strip() if rm else None
        if ref is None:
            for kw in ["less than", "greater than"]:
                if kw in tl:
                    m2 = re.search(kw + r'\s+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
                    if m2: ref = kw + " " + m2.group(1)
        return {
            "type": "lab",
            "test":      tm.group(1) if tm else None,
            "value":     vm.group(1).strip() if vm else None,
            "unit":      vm.group(2).strip() if (vm and vm.group(2)) else None,
            "reference": ref,
        }

    # Prescription
    all_meds = (_classes or set()) | EXTRA_MEDS
    found_med = None
    for med in sorted(all_meds, key=len, reverse=True):
        if med.lower() in tl:
            found_med = med; break
    d   = DOSAGE_RE.search(text)
    dur = DURATION_RE.search(text)
    frq = None
    for f in FREQ_TERMS:
        if f in tl: frq = f; break
    return {
        "type":      "prescription",
        "medicine":  found_med,
        "dosage":    f"{d.group(1)} {d.group(2).lower()}" if d else None,
        "frequency": frq,
        "duration":  f"{dur.group(1)} {dur.group(2).lower()}" if dur else None,
    }


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import glob

    print("=== Self-test: read_printed ===")
    syn_dir = os.path.join(PROJECT_ROOT, "logs", "ocr_synthetic_images")
    syn_imgs = sorted(glob.glob(os.path.join(syn_dir, "*.png")))
    if syn_imgs:
        text = read_printed(syn_imgs[0])
        print(f"  Image: {os.path.basename(syn_imgs[0])}")
        print(f"  OCR  : {text}")
        print(f"  Fields: {json.dumps(extract_fields(text), indent=2)}")
    else:
        print("  No synthetic images found. Run train.py first.")

    print("\n=== Self-test: classify_handwritten_medicine ===")
    DATA_BASE = os.path.expanduser(
        "~/.cache/kagglehub/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset"
        "/versions/1/Doctor's Handwritten Prescription BD dataset"
    )
    hw_imgs = sorted(glob.glob(os.path.join(DATA_BASE, "Testing/testing_words/*.png")))
    if hw_imgs:
        result = classify_handwritten_medicine(hw_imgs[0])
        print(f"  Image: {os.path.basename(hw_imgs[0])}")
        for r in result:
            print(f"  Rank {r['rank']}: {r['medicine_name']} ({r['probability']*100:.1f}%)")
    else:
        print("  No handwriting test images found.")

    print("\n=== Self-test: extract_fields (prescription) ===")
    rx = "Tab Metformin 500 mg twice daily for 30 days"
    print(f"  Input : {rx}")
    print(f"  Output: {json.dumps(extract_fields(rx), indent=2)}")

    print("\n=== Self-test: extract_fields (lab) ===")
    lab = "HbA1c 6.8 % Ref 4.0-5.6"
    print(f"  Input : {lab}")
    print(f"  Output: {json.dumps(extract_fields(lab), indent=2)}")
