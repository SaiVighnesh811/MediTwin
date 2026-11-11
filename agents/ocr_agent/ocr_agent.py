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
from typing import Optional, List, Dict, Any, Union
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
DOSAGE_RE   = re.compile(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)\b', re.IGNORECASE)
FREQ_TERMS  = [
    "three times daily", "thrice daily", "twice daily", "once daily",
    "every 6 hours", "every 8 hours", "every 12 hours", "at bedtime",
    "at night", "before meals", "before breakfast", "after meals",
    "single dose", "once", "daily"
]
DURATION_RE = re.compile(r'(?:for\s+)?(\d+)\s+(days?|months?|weeks?|years?)\b', re.IGNORECASE)
TEST_NAMES  = [
    "HbA1c", "Fasting Blood Glucose", "Total Cholesterol", "Haemoglobin", "Hemoglobin",
    "Serum Creatinine", "ALT SGPT", "ALT", "SGPT", "AST", "TSH", "Platelet Count",
    "Urea", "WBC", "LDL Cholesterol", "HDL Cholesterol", "Cholesterol", "Blood Glucose",
    "Glucose", "Uric Acid", "Sodium", "Potassium", "Triglycerides", "BUN"
]
TEST_RE     = re.compile(
    r'^(HbA1c|Fasting Blood Glucose|Total Cholesterol|Haemoglobin|Hemoglobin|Serum Creatinine|'
    r'ALT SGPT|ALT|SGPT|AST|TSH|Platelet Count|Urea|WBC|LDL Cholesterol|HDL Cholesterol|'
    r'Cholesterol|Glucose|Blood Glucose|Uric Acid|Sodium|Potassium|Triglycerides|BUN)\b',
    re.IGNORECASE)
VALUE_RE    = re.compile(r'(\d+(?:\.\d+)?(?:\s*[-\/]\s*\d+(?:\.\d+)?)?)\s*([a-zA-Z/%_]+(?:/[a-zA-Z]+)?)?')
REFRANGE_RE = re.compile(
    r'(?:Ref(?:erence)?\s*|less than |greater than |>|<)(\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)',
    re.IGNORECASE)


def _split_line_into_segments(line: str, all_meds: set) -> List[str]:
    """
    Fallback for already-joined text (PDFs, tests, or single-line strings):
    Splits line into segments starting at:
      - Every recognised medicine name (and at 'Tab'/'Cap'/'Inj' prefixes)
      - Every lab test name followed by a number and unit
    """
    if not line or not line.strip():
        return []

    clean = line.strip()
    prefixes = r'(?:\b(?:Tab(?:let)?|Cap(?:sule)?|Inj(?:ection)?|Syrup|Syp|Drops?|Rx)\.?\s+)'
    meds_pat = '|'.join(re.escape(m) for m in sorted(all_meds, key=len, reverse=True))
    tests_pat = '|'.join(re.escape(t) for t in sorted(TEST_NAMES, key=len, reverse=True))

    med_re = re.compile(r'(?:' + prefixes + r')?\b(?:' + meds_pat + r')\b', re.IGNORECASE)
    lab_re = re.compile(r'\b(?:' + tests_pat + r')\b(?:\s*[:=]?\s*)\d+(?:\.\d+)?\s*[a-zA-Z/%_]+', re.IGNORECASE)

    matches = []
    for m in med_re.finditer(clean):
        matches.append((m.start(), m.end(), 'med'))
    for m in lab_re.finditer(clean):
        matches.append((m.start(), m.end(), 'lab'))

    if not matches:
        return [clean]

    matches.sort(key=lambda x: x[0])

    non_overlapping = []
    last_end = -1
    for start, end, kind in matches:
        if start >= last_end:
            non_overlapping.append(start)
            last_end = end

    segments = []
    prev = 0
    for pos in non_overlapping:
        if pos > prev:
            seg = clean[prev:pos].strip()
            if seg:
                segments.append(seg)
        prev = pos
    if prev < len(clean):
        seg = clean[prev:].strip()
        if seg:
            segments.append(seg)
    return segments


def eval_ref_range(val_str: Optional[str], ref_str: Optional[str]) -> Optional[str]:
    """
    Evaluates numeric lab value against printed reference string.
    Returns 'above', 'within', or 'below' only when printed range is parseable.
    Never uses any unprinted range.
    """
    if not ref_str or not val_str:
        return None
    try:
        val = float(val_str)
    except ValueError:
        return None

    # 'less than X' or '< X' or '<= X'
    m_lt = re.search(r'(?:less than|<|<=)\s*(\d+(?:\.\d+)?)', ref_str, re.IGNORECASE)
    if m_lt:
        high = float(m_lt.group(1))
        return "above" if val > high else "within"

    # 'greater than X' or '> X' or '>= X'
    m_gt = re.search(r'(?:greater than|>|>=)\s*(\d+(?:\.\d+)?)', ref_str, re.IGNORECASE)
    if m_gt:
        low = float(m_gt.group(1))
        return "below" if val < low else "within"

    # 'low - high' or 'low to high'
    m_range = re.search(r'(\d+(?:\.\d+)?)\s*(?:[-–]|to)\s*(\d+(?:\.\d+)?)', ref_str, re.IGNORECASE)
    if m_range:
        low = float(m_range.group(1))
        high = float(m_range.group(2))
        if val < low:
            return "below"
        elif val > high:
            return "above"
        else:
            return "within"

    return None


def parse_line_rx(line: str, all_meds: set) -> Optional[dict]:
    """Extracts prescription fields from a single line."""
    found_med = None
    med_end = 0
    for med in sorted(all_meds, key=len, reverse=True):
        m = re.search(r'\b' + re.escape(med) + r'\b', line, re.IGNORECASE)
        if m:
            found_med = med
            med_end = m.end()
            break
    if not found_med:
        return None

    after_med = line[med_end:]

    # Dose strictly after medicine name on the same line
    d = DOSAGE_RE.search(after_med)
    dosage = None
    if d:
        unit = d.group(2)
        unit_str = "IU" if unit.upper() == "IU" else unit.lower()
        dosage = f"{d.group(1)} {unit_str}"

    # Frequency: the words actually present
    freq = None
    for f in FREQ_TERMS:
        if re.search(r'\b' + re.escape(f) + r'\b', after_med, re.IGNORECASE):
            freq = f
            break

    # Duration: full words e.g. "30 days", not "30 day"
    dur = DURATION_RE.search(after_med)
    duration = None
    if dur:
        duration = f"{dur.group(1)} {dur.group(2).lower()}"

    return {
        "type": "prescription",
        "medicine": found_med,
        "dosage": dosage,
        "frequency": freq,
        "duration": duration,
    }


def parse_line_lab(line: str) -> Optional[dict]:
    """Extracts lab test fields from a single line."""
    tm = TEST_RE.search(line)
    if not tm:
        return None
    test_name = tm.group(1).strip()
    rest = line[tm.end():].strip()

    # Look for Ref / Reference
    ref_match = re.search(r'(?:Ref(?:erence)?[:\s]+)(.+)$', rest, re.IGNORECASE)
    ref_text = None
    if ref_match:
        ref_text = ref_match.group(1).strip()
        before_ref = rest[:ref_match.start()].strip()
    else:
        m2 = re.search(r'(?:less than|greater than|[<>]|\b\d+\s*[-–]\s*\d+).+$', rest, re.IGNORECASE)
        if m2:
            ref_text = m2.group(0).strip()
            before_ref = rest[:m2.start()].strip()
        else:
            before_ref = rest

    vm = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z/%_]+(?:/[a-zA-Z]+)?)?', before_ref)
    val = vm.group(1) if vm else None
    unit = vm.group(2).strip() if (vm and vm.group(2)) else None
    flag = eval_ref_range(val, ref_text) if (val and ref_text) else None

    return {
        "type": "lab",
        "test": test_name,
        "value": val,
        "unit": unit,
        "reference": ref_text,
        "flag": flag,
    }


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


def _get_box_coords(box):
    """Sort key for EasyOCR box: (min_y, min_x)."""
    if isinstance(box, (list, tuple)) and len(box) >= 2:
        bbox = box[0]
        if isinstance(bbox, (list, tuple)) and len(bbox) > 0:
            if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 2:
                # 4-point polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                min_y = min(pt[1] for pt in bbox)
                min_x = min(pt[0] for pt in bbox)
                return (min_y, min_x)
            elif len(bbox) >= 2:
                # [x, y, w, h] or [x, y]
                return (bbox[1], bbox[0])
    return (0, 0)


def _get_box_text(box):
    """Extract text string from box tuple or string."""
    if isinstance(box, (list, tuple)):
        if len(box) >= 2:
            return str(box[1])
        elif len(box) == 1:
            return str(box[0])
    return str(box)


# ──────────────────────────────────────────────────────────────────────────────
def read_printed(image_path: str) -> str:
    """
    Run EasyOCR on a printed/typed image and return lines sorted by position.
    EasyOCR is a pretrained model — no fine-tuning performed.
    Sorts boxes top to bottom by bbox y (then left to right).
    Keeps each box's text as ONE line separated by newlines.
    NOTE: Validated on synthetic PIL images only; not tested on real scans.
    """
    _load_ocr_reader()
    results = _ocr_reader.readtext(image_path)
    if not results:
        return ""
    sorted_boxes = sorted(results, key=_get_box_coords)
    lines = [_get_box_text(b).strip() for b in sorted_boxes if _get_box_text(b).strip()]
    return "\n".join(lines).strip()


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


def extract_fields(text: Union[str, List[Any]]) -> dict:
    """
    Extract structured fields from OCR text line by line.
    Parses prescriptions and lab lines into separate structured entries.
    Returns a dict containing medicines[], labs[], raw_text, and extracted[].
    """
    if text is None:
        return {
            "type": "unknown",
            "medicines": [],
            "labs": [],
            "raw_text": "",
            "extracted": [],
        }

    all_meds = set(_classes or []) | EXTRA_MEDS

    if isinstance(text, (list, tuple)):
        sorted_boxes = sorted(text, key=_get_box_coords)
        raw_lines = [_get_box_text(b).strip() for b in sorted_boxes if _get_box_text(b).strip()]
        raw_text = "\n".join(raw_lines)
    else:
        raw_text = str(text)
        raw_lines = raw_text.splitlines()

    lines = []
    for rl in raw_lines:
        rl_clean = rl.strip()
        if not rl_clean:
            continue
        segs = _split_line_into_segments(rl_clean, all_meds)
        lines.extend(segs)

    medicines = []
    labs = []
    extracted = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        rx_item = parse_line_rx(line_clean, all_meds)
        if rx_item:
            medicines.append(rx_item)
            extracted.append(rx_item)
            continue

        lab_item = parse_line_lab(line_clean)
        if lab_item:
            labs.append(lab_item)
            extracted.append(lab_item)
            continue

    primary_type = "unknown"
    if medicines and labs:
        primary_type = "document"
    elif medicines:
        primary_type = "prescription"
    elif labs:
        primary_type = "lab"

    first_med = medicines[0] if medicines else {}
    first_lab = labs[0] if labs else {}

    return {
        "type": primary_type,
        "medicines": medicines,
        "labs": labs,
        "raw_text": raw_text,
        "extracted": extracted,
        # Backward compatibility for legacy tests checking single fields
        "medicine": first_med.get("medicine"),
        "dosage": first_med.get("dosage"),
        "frequency": first_med.get("frequency"),
        "duration": first_med.get("duration"),
        "test": first_lab.get("test"),
        "value": first_lab.get("value"),
        "unit": first_lab.get("unit"),
        "reference": first_lab.get("reference"),
        "flag": first_lab.get("flag"),
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
