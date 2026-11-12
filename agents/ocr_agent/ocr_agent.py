"""
agents/ocr_agent/ocr_agent.py
MediTwin AI — Chapter 4: OCR Agent   Public API

    read_printed(image_path)                 → OCR text from printed/typed image
    classify_handwritten_medicine(image_path) → top-3 medicine names + probs
    extract_fields(text)                     → structured fields from OCR text

DISCLAIMER: Closed-set classifier (78 BD medicine names only).
            EasyOCR on synthetic/printed images only — not validated on real scans.
"""

import os, sys, re, json, math, statistics
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
    # Original Chapter 4-15
    "Metformin", "Amoxicillin", "Paracetamol", "Amlodipine", "Insulin Glargine",
    "Atorvastatin", "Azithromycin", "Omeprazole", "Fexofenadine", "Clonazepam",
    "Cetirizine", "Metronidazole", "Fluconazole", "Baclofen", "Esomeprazole",
    # Chapter 16a additions
    "Losartan", "Telmisartan", "Glimepiride", "Gliclazide", "Sitagliptin",
    "Aspirin", "Clopidogrel", "Rosuvastatin", "Simvastatin", "Pantoprazole",
    "Ramipril", "Enalapril", "Atenolol", "Metoprolol", "Furosemide",
    "Hydrochlorothiazide", "Levothyroxine", "Ciprofloxacin", "Ibuprofen",
}

# ── Safety & Unit Definitions ─────────────────────────────────────────────────
ALLOWED_DOSE_UNITS = {"mg", "mcg", "ug", "g", "ml", "iu", "%"}
VALID_LAB_UNITS = ["mg/dL", "mmol/L", "g/dL", "%", "U/L", "IU/L", "ng/mL", "ug/dL", "mEq/L"]

FREQ_MAP = [
    (re.compile(r'\b(?:three times daily|thrice daily|tds|tid)\b', re.IGNORECASE), "three times daily"),
    (re.compile(r'\b(?:twice daily|bd|bid)\b', re.IGNORECASE), "twice daily"),
    (re.compile(r'\b(?:once daily|od)\b', re.IGNORECASE), "once daily"),
    (re.compile(r'\b(?:four times daily|qid)\b', re.IGNORECASE), "four times daily"),
    (re.compile(r'\b(?:at bedtime|at night|hs)\b', re.IGNORECASE), "at bedtime"),
    (re.compile(r'\b(?:when needed|as needed|sos|prn)\b', re.IGNORECASE), "when needed"),
    (re.compile(r'\bevery 6 hours\b', re.IGNORECASE), "every 6 hours"),
    (re.compile(r'\bevery 8 hours\b', re.IGNORECASE), "every 8 hours"),
    (re.compile(r'\bevery 12 hours\b', re.IGNORECASE), "every 12 hours"),
    (re.compile(r'\bbefore meals\b', re.IGNORECASE), "before meals"),
    (re.compile(r'\bbefore breakfast\b', re.IGNORECASE), "before breakfast"),
    (re.compile(r'\bafter meals\b', re.IGNORECASE), "after meals"),
    (re.compile(r'\bsingle dose\b', re.IGNORECASE), "single dose"),
    (re.compile(r'\bdaily\b', re.IGNORECASE), "daily"),
]
FREQ_TERMS = [term for _, term in FREQ_MAP]

DURATION_RE = re.compile(r'(?:for\s+|x\s*|×\s*)?(\d+)\s+(days?|months?|weeks?|years?)\b', re.IGNORECASE)
DOSAGE_RE   = re.compile(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)\b', re.IGNORECASE)

LAB_ALIASES = {
    "hba1c": "HbA1c",
    "hbalc": "HbA1c",
    "a1c": "HbA1c",
    "glycated hemoglobin": "HbA1c",
    "glycated haemoglobin": "HbA1c",
    "fasting blood glucose": "Fasting Blood Glucose",
    "blood glucose": "Blood Glucose",
    "glucose": "Glucose",
    "total cholesterol": "Total Cholesterol",
    "hdl cholesterol": "HDL Cholesterol",
    "ldl cholesterol": "LDL Cholesterol",
    "cholesterol": "Cholesterol",
    "triglycerides": "Triglycerides",
    "serum creatinine": "Serum Creatinine",
    "creatinine": "Serum Creatinine",
    "haemoglobin": "Haemoglobin",
    "hemoglobin": "Hemoglobin",
    "alt sgpt": "ALT SGPT",
    "alt": "ALT",
    "sgpt": "SGPT",
    "ast": "AST",
    "tsh": "TSH",
    "platelet count": "Platelet Count",
    "platelets": "Platelet Count",
    "urea": "Urea",
    "wbc": "WBC",
    "uric acid": "Uric Acid",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "bun": "BUN",
    "bilirubin": "Bilirubin",
    "total bilirubin": "Total Bilirubin",
    "direct bilirubin": "Direct Bilirubin",
    "albumin": "Albumin",
    "total protein": "Total Protein",
    "alkaline phosphatase": "Alkaline Phosphatase",
    "alp": "ALP",
    "calcium": "Calcium",
    "phosphorus": "Phosphorus",
    "esr": "ESR",
}

TEST_NAMES = sorted(list(set(LAB_ALIASES.values())), key=len, reverse=True)
TEST_RE = re.compile(
    r'^(?:' + '|'.join(re.escape(k) for k in sorted(LAB_ALIASES.keys(), key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)
VALUE_RE = re.compile(r'(\d+(?:\.\d+)?(?:\s*[-\/]\s*\d+(?:\.\d+)?)?)\s*([a-zA-Z/%_]+(?:/[a-zA-Z]+)?)?')
REFRANGE_RE = re.compile(
    r'(?:Ref(?:erence)?\s*|less than |more than |greater than |>|<)(\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)',
    re.IGNORECASE
)


def levenshtein(s1: str, s2: str) -> int:
    """Calculates Levenshtein edit distance between two strings case-insensitively."""
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[len(s2)]


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
    tests_pat = '|'.join(re.escape(t) for t in sorted(LAB_ALIASES.keys(), key=len, reverse=True))

    med_re = re.compile(r'(?:' + prefixes + r')?\b(?:' + meds_pat + r')\b', re.IGNORECASE)
    lab_re = re.compile(r'\b(?:' + tests_pat + r')\b(?:\s*[:=]?\s*)\d+(?:\.\d+)?\s*[a-zA-Z/%_\^]+', re.IGNORECASE)

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
    Unreadable reference returns None. Never uses unprinted range.
    """
    if not ref_str or not val_str:
        return None
    try:
        val = float(val_str)
    except ValueError:
        return None

    # less than / < / <=
    m_lt = re.search(r'(?:less than|<|<=)\s*(\d+(?:\.\d+)?)', ref_str, re.IGNORECASE)
    if m_lt:
        try:
            high = float(m_lt.group(1))
            return "above" if val > high else "within"
        except ValueError:
            return None

    # more than / greater than / > / >=
    m_gt = re.search(r'(?:more than|greater than|>|>=)\s*(\d+(?:\.\d+)?)', ref_str, re.IGNORECASE)
    if m_gt:
        try:
            low = float(m_gt.group(1))
            return "below" if val < low else "within"
        except ValueError:
            return None

    # range lo - hi or lo to hi
    m_range = re.search(r'(\d+(?:\.\d+)?)\s*(?:[-–]|to)\s*(\d+(?:\.\d+)?)', ref_str, re.IGNORECASE)
    if m_range:
        try:
            low = float(m_range.group(1))
            high = float(m_range.group(2))
            if val < low:
                return "below"
            elif val > high:
                return "above"
            else:
                return "within"
        except ValueError:
            return None

    return None


def match_medicine(line: str, all_meds: set):
    """
    Finds medicine name in line.
    1. Exact match (case-insensitive).
    2. Fuzzy match (edit dist 1 for >=6 chars, 2 for >=10 chars; only when single best match).
    Returns (found_med, raw_token, end_index, is_fuzzy).
    """
    # 1. Exact match
    for med in sorted(all_meds, key=len, reverse=True):
        m = re.search(r'\b' + re.escape(med) + r'\b', line, re.IGNORECASE)
        if m:
            return med, m.group(0), m.end(), False

    # 2. Fuzzy match
    tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r'\b[a-zA-Z]{4,}\b', line)]
    skip_tokens = {'tab', 'tablet', 'cap', 'capsule', 'inj', 'syrup', 'daily', 'once', 'twice', 'thrice', 'days', 'weeks', 'months', 'dose'}
    candidates = []
    for token_text, t_start, t_end in tokens:
        if token_text.lower() in skip_tokens:
            continue
        for med in all_meds:
            max_d = 2 if len(med) >= 10 else (1 if len(med) >= 6 else 0)
            if max_d == 0 or abs(len(token_text) - len(med)) > max_d:
                continue
            d = levenshtein(token_text, med)
            if d <= max_d:
                candidates.append((d, med, token_text, t_end))

    if not candidates:
        return None, None, None, False

    min_d = min(c[0] for c in candidates)
    best = [c for c in candidates if c[0] == min_d]
    unique_meds = set(c[1] for c in best)
    if len(unique_meds) == 1:
        d, med, raw_token, t_end = best[0]
        return med, raw_token, t_end, True

    return None, None, None, False


def parse_line_rx(line: str, all_meds: set) -> Optional[dict]:
    """Extracts prescription fields from a single line."""
    found_med, raw_med, med_end, is_fuzzy = match_medicine(line, all_meds)
    if not found_med:
        return None

    after_med = line[med_end:]

    # Dose parsing and safety
    dosage = None
    dose_unclear = False

    # Misread digits like Smg, Omg
    m_misread = re.search(r'\b([S|s|O|o][0-9]*)\s*(mg|mcg|ug|g|ml|iu)\b', after_med)
    if m_misread:
        dosage = None
        dose_unclear = True
    else:
        m_num_unit = re.search(r'\b(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)\b', after_med)
        if m_num_unit:
            val, unit = m_num_unit.group(1), m_num_unit.group(2).lower()
            if unit in {'days', 'day', 'weeks', 'week', 'months', 'month', 'years', 'year', 'times', 'x'}:
                dosage = None
            elif unit in ALLOWED_DOSE_UNITS:
                unit_str = "IU" if unit == "iu" else unit
                dosage = f"{val} {unit_str}"
            else:
                dosage = None
                dose_unclear = True

    # Frequency parsing and safety
    freq = None
    frequency_unclear = False

    # Check for misread word preceding 'daily' (e.g. 'cwice daily')
    m_daily = re.search(r'\b([a-zA-Z]+)\s+daily\b', after_med, re.IGNORECASE)
    if m_daily:
        prec = m_daily.group(1).lower()
        if prec in {"once", "twice", "thrice"}:
            freq = "three times daily" if prec == "thrice" else f"{prec} daily"
        elif prec == "times":
            m_times = re.search(r'\b(three|four)\s+times\s+daily\b', after_med, re.IGNORECASE)
            if m_times:
                freq = f"{m_times.group(1).lower()} times daily"
            else:
                frequency_unclear = True
        elif prec in ALLOWED_DOSE_UNITS or prec in {"day", "days"}:
            freq = "daily"
        else:
            # Misread token like "cwice"
            frequency_unclear = True
            freq = None

    if not freq and not frequency_unclear:
        for pattern, term in FREQ_MAP:
            if pattern.search(after_med):
                freq = term
                break

    # Duration
    dur = DURATION_RE.search(after_med)
    duration = None
    if dur:
        dur_num = dur.group(1)
        dur_unit = dur.group(2).lower()
        if dur_unit == "day" and dur_num != "1":
            dur_unit = "days"
        elif dur_unit == "days" and dur_num == "1":
            dur_unit = "day"
        duration = f"{dur_num} {dur_unit}"

    return {
        "type": "prescription",
        "medicine": found_med,
        "raw_medicine": raw_med,
        "dosage": dosage,
        "dose_unclear": dose_unclear,
        "frequency": freq,
        "frequency_unclear": frequency_unclear,
        "duration": duration,
    }


def parse_line_lab(line: str) -> Optional[dict]:
    """Extracts lab test fields from a single line or table row."""
    sorted_keys = sorted(LAB_ALIASES.keys(), key=len, reverse=True)
    pat = r'\b(' + '|'.join(re.escape(k) for k in sorted_keys) + r')\b'
    tm = re.search(pat, line, re.IGNORECASE)
    if not tm:
        return None
    test_name = LAB_ALIASES[tm.group(1).lower()]
    rest = line[tm.end():].strip()

    # Must find a numeric value
    vm = re.search(r'\b(\d+(?:\.\d+)?)\b', rest)
    if not vm:
        return None
    val_str = vm.group(1)
    after_val = rest[vm.end():].strip()

    # Extract unit (candidate)
    unit = None
    unit_unclear = False
    ref_start_str = after_val

    um = re.match(r'([a-zA-Z/%_\^]+(?:/[a-zA-Z]+)?)', after_val)
    if um:
        cand_unit = um.group(1)
        matched_valid = None
        for vu in VALID_LAB_UNITS:
            if cand_unit.lower() == vu.lower():
                matched_valid = vu
                break
        if matched_valid:
            unit = matched_valid
        else:
            dists = [(levenshtein(cand_unit, vu), vu) for vu in VALID_LAB_UNITS]
            min_d, best_vu = min(dists, key=lambda x: x[0])
            if min_d <= 2:
                unit = best_vu
            else:
                unit = None
                unit_unclear = True
        ref_start_str = after_val[um.end():].strip()

    # Printed Flag (H, L, HH, LL)
    flag_printed = None
    m_flag = re.search(r'(?:\[?(HH|LL|H|L)\]?|\b(High|Low)\b)\s*$', ref_start_str, re.IGNORECASE)
    if m_flag:
        raw_f = m_flag.group(1)
        if raw_f:
            flag_printed = raw_f.upper()
        elif m_flag.group(2):
            flag_printed = "H" if m_flag.group(2).lower() == "high" else "L"
        ref_start_str = ref_start_str[:m_flag.start()].strip()

    # Reference range
    ref_text = None
    reference_inferred = False
    reference_unclear = False
    flag = None

    if ref_start_str:
        cleaned_ref = re.sub(r'^(?:Ref(?:erence)?[:\s]*|[:\s]+)', '', ref_start_str, flags=re.IGNORECASE).strip()
        cleaned_ref = re.sub(r'\s+(?:\[?[HL]\]?|High|Low)\s*$', '', cleaned_ref, flags=re.IGNORECASE).strip()
        if cleaned_ref:
            ref_text = cleaned_ref

    if ref_text:
        # Check for garbage like "200 ;" or standalone semicolon/question mark
        if ";" in ref_text or "?" in ref_text:
            reference_unclear = True
            flag = None
        else:
            # Check for two ascending numbers with no comparator (e.g. "70 100" or "70.0 100.0")
            m_asc = re.match(r'^(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)$', ref_text)
            if m_asc:
                low_val = float(m_asc.group(1))
                high_val = float(m_asc.group(2))
                if low_val < high_val:
                    reference_inferred = True
                    try:
                        val_f = float(val_str)
                        if val_f < low_val:
                            flag = "below"
                        elif val_f > high_val:
                            flag = "above"
                        else:
                            flag = "within"
                    except ValueError:
                        flag = None
                else:
                    reference_unclear = True
                    flag = None
            else:
                # Check for bare number with NO comparator (e.g. "200")
                m_bare = re.match(r'^\d+(?:\.\d+)?$', ref_text)
                if m_bare:
                    reference_unclear = True
                    flag = None
                else:
                    # Valid comparator (<, >, <=, >=, less than, more than) or range (70 - 100)
                    flag = eval_ref_range(val_str, ref_text)
                    if flag is None:
                        reference_unclear = True

    return {
        "type": "lab",
        "test": test_name,
        "value": val_str,
        "unit": unit,
        "unit_unclear": unit_unclear,
        "reference": ref_text,
        "reference_inferred": reference_inferred,
        "reference_unclear": reference_unclear,
        "flag": flag,
        "flag_printed": flag_printed,
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
                min_y = min(pt[1] for pt in bbox)
                min_x = min(pt[0] for pt in bbox)
                return (min_y, min_x)
            elif len(bbox) >= 2:
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


def group_boxes_into_rows(boxes: list) -> List[str]:
    """
    Groups EasyOCR boxes into rows:
    1. Keeps boxes' corner points.
    2. Estimates skew as median angle of boxes' top edges.
    3. Rotates box centres by minus that angle.
    4. Groups boxes into rows when rotated centre-y differ by less than 0.5 * median box height.
    5. Sorts each row left to right (rotated x).
    6. Joins boxes in each row with one space.
    """
    if not boxes:
        return []

    parsed_boxes = []
    angles = []
    heights = []

    for b in boxes:
        text = _get_box_text(b).strip()
        if not text:
            continue
        bbox = b[0] if (isinstance(b, (list, tuple)) and len(b) >= 2) else None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and isinstance(bbox[0], (list, tuple)):
            p0, p1, p2, p3 = bbox[0], bbox[1], bbox[2], bbox[3]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            angle = math.atan2(dy, dx)
            h = 0.5 * (math.hypot(p3[0] - p0[0], p3[1] - p0[1]) + math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            cx = sum(p[0] for p in bbox) / 4.0
            cy = sum(p[1] for p in bbox) / 4.0
            angles.append(angle)
            if h > 0:
                heights.append(h)
        else:
            coords = _get_box_coords(b)
            cy, cx = coords[0], coords[1]
            angle = 0.0
            h = 20.0
        parsed_boxes.append({"text": text, "cx": cx, "cy": cy, "h": h})

    if not parsed_boxes:
        return []

    skew_angle = statistics.median(angles) if angles else 0.0
    median_h = statistics.median(heights) if heights else 20.0
    if median_h <= 0:
        median_h = 20.0
    row_thresh = 0.5 * median_h

    # Rotate box centers by -skew_angle
    theta = -skew_angle
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    for b in parsed_boxes:
        b["rot_x"] = b["cx"] * cos_t - b["cy"] * sin_t
        b["rot_y"] = b["cx"] * sin_t + b["cy"] * cos_t

    # Sort boxes by rot_y
    parsed_boxes.sort(key=lambda x: x["rot_y"])

    # Group into rows
    rows = []
    for b in parsed_boxes:
        matched_row = None
        min_diff = None
        for r in rows:
            diff = abs(b["rot_y"] - r["mean_y"])
            if diff < row_thresh:
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    matched_row = r
        if matched_row is not None:
            matched_row["boxes"].append(b)
            matched_row["mean_y"] = sum(x["rot_y"] for x in matched_row["boxes"]) / len(matched_row["boxes"])
        else:
            rows.append({"mean_y": b["rot_y"], "boxes": [b]})

    # Sort rows top to bottom
    rows.sort(key=lambda r: r["mean_y"])

    # In each row, sort left to right by rot_x and join with one space
    result_lines = []
    for r in rows:
        r["boxes"].sort(key=lambda x: x["rot_x"])
        line_str = " ".join(b["text"] for b in r["boxes"]).strip()
        if line_str:
            result_lines.append(line_str)

    return result_lines


# ──────────────────────────────────────────────────────────────────────────────
def read_printed(image_path: str) -> str:
    """
    Run EasyOCR on a printed/typed image and return lines grouped by row.
    EasyOCR is a pretrained model — no fine-tuning performed.
    Corrects skew and groups boxes into rows.
    NOTE: Validated on synthetic PIL images only; not tested on real scans.
    """
    _load_ocr_reader()
    results = _ocr_reader.readtext(image_path)
    if not results:
        return ""
    lines = group_boxes_into_rows(results)
    return "\n".join(lines).strip()


def classify_handwritten_medicine(image_path: Union[str, Image.Image]) -> list[dict]:
    """
    Classify a handwritten medicine word image (path or PIL Image).
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
    if isinstance(image_path, Image.Image):
        img = image_path.convert("RGB")
    elif isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="white")
    x    = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(_hw_model(x), dim=1)[0].cpu().numpy()
    top3_idx = np.argsort(probs)[::-1][:3]
    return [
        {"rank": i+1, "medicine_name": _classes[idx], "probability": round(float(probs[idx]), 4)}
        for i, idx in enumerate(top3_idx)
    ]


HANDWRITING_CONFIDENCE_THRESHOLD = 0.3
HW_GATE = 0.5
MIN_INK_STD = 8.0

TABLE_HEADERS = {
    "test", "result", "results", "unit", "units", "reference", "ref", "flag", "flags",
    "range", "interval", "parameter", "investigation", "biological", "status", "sample"
}


def is_purely_digits_units_symbols(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return True
    if re.fullmatch(r'[\d\s\.,;:/\-–+*()%<>=;\[\]#]+', clean):
        return True
    tokens = clean.split()
    for t in tokens:
        t_clean = re.sub(r'[\d\s\.,;:/\-–+*()%<>=;\[\]#]+', '', t).lower()
        if not t_clean:
            continue
        if t_clean in ALLOWED_DOSE_UNITS or t_clean in {
            "mg/dl", "mmol/l", "g/dl", "u/l", "iu/l", "ng/ml", "ug/dl", "meq/l", "dl", "ref", "reference"
        }:
            continue
        return False
    return True


def is_known_lab_name(text: str) -> bool:
    clean_lower = text.strip().lower()
    return clean_lower in LAB_ALIASES


def has_enough_ink(img: Image.Image, min_std: float = MIN_INK_STD) -> bool:
    try:
        gray = img.convert("L")
        arr = np.array(gray)
        return float(np.std(arr)) >= min_std
    except Exception:
        return False


def is_known_or_structured_box(text: str, all_meds: set) -> bool:
    """
    Checks if a box text represents a known medicine, dose, frequency,
    duration, table header, or purely numeric/symbol string, so that it is
    NOT routed to handwriting classification.
    """
    clean = text.strip()
    if not clean:
        return True

    clean_lower = clean.lower()

    # 1. Purely digits, units, or symbols
    if is_purely_digits_units_symbols(clean):
        return True

    # 2. Table headers (Test, Result, Unit, Reference, Flag, etc.)
    if clean_lower in TABLE_HEADERS:
        return True

    # 3. Known lab test name or parsed lab line
    if is_known_lab_name(clean):
        return True
    if parse_line_lab(clean) is not None:
        return True
    for alias_key in LAB_ALIASES:
        if alias_key in clean_lower:
            return True

    # 4. Dose token or dosage pattern
    if DOSAGE_RE.search(clean):
        return True
    for token in clean_lower.split():
        for u in ALLOWED_DOSE_UNITS:
            if token.endswith(u) and any(c.isdigit() for c in token):
                return True
        if token in ALLOWED_DOSE_UNITS:
            return True

    # 5. Frequency term
    for pattern, _ in FREQ_MAP:
        if pattern.search(clean):
            return True

    # 6. Duration pattern
    if DURATION_RE.search(clean):
        return True

    # 7. Known printed medicine (exact or fuzzy)
    matched = match_medicine(clean, all_meds)
    if matched and matched[0]:
        return True

    for prefix in ["tab", "cap", "inj", "syp"]:
        if clean_lower.startswith(prefix + " "):
            rem = clean[len(prefix) + 1:].strip()
            rem_m = match_medicine(rem, all_meds)
            if rem_m and rem_m[0]:
                return True

    # 8. Common prefixes/labels
    if clean_lower in {"tab", "cap", "inj", "syp", "ref", "reference", "rx"}:
        return True

    return False


def crop_box(img: Image.Image, bbox: Any, pad: int = 5) -> Optional[Image.Image]:
    """Crops a bounding box with padding from a PIL image."""
    try:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w, h = img.size
        left = max(0, int(min_x - pad))
        top = max(0, int(min_y - pad))
        right = min(w, int(max_x + pad))
        bottom = min(h, int(max_y + pad))
        if right > left and bottom > top:
            return img.crop((left, top, right, bottom))
    except Exception:
        pass
    return None


def process_document(image_input: Any, run_handwriting: bool = True) -> dict:
    """
    Unified entry point for document analysis (used by /v1/ocr and /v1/analyze orchestrator).
    Runs printed OCR extraction and conditionally invokes the handwriting classifier
    on unrecognized boxes below HW_GATE confidence or small single-word crops with enough ink.
    """
    all_meds = set(_classes or []) | EXTRA_MEDS
    img = None
    boxes = []

    if isinstance(image_input, str) and os.path.exists(image_input):
        try:
            img = Image.open(image_input).convert("RGB")
        except Exception:
            img = None
        _load_ocr_reader()
        boxes = _ocr_reader.readtext(image_input)
    elif isinstance(image_input, (list, tuple)):
        boxes = list(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        _load_ocr_reader()
        boxes = _ocr_reader.readtext(np.array(img))

    parsed = extract_fields(boxes if boxes else "")
    hw_candidates = []
    unclear_regions = 0

    if run_handwriting:
        is_small_crop = False
        if img is not None:
            w, h = img.size
            longer_side = max(w, h)
            # Whole-image path: only when longer side <= 600 px AND enough ink
            if longer_side <= 600 and has_enough_ink(img, MIN_INK_STD):
                if len(boxes) <= 1:
                    is_small_crop = True

        if is_small_crop:
            skip = False
            box_conf = 0.0
            if boxes and len(boxes) == 1:
                box_text = boxes[0][1] if len(boxes[0]) > 1 else ""
                box_conf = float(boxes[0][2]) if len(boxes[0]) > 2 else 0.0
                if box_conf >= HW_GATE or is_known_or_structured_box(box_text, all_meds):
                    skip = True
            if not skip:
                target_img = img if img is not None else Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="white")
                top3 = classify_handwritten_medicine(target_img)
                if top3:
                    top_p = float(top3[0]["probability"])
                    if top_p >= HANDWRITING_CONFIDENCE_THRESHOLD:
                        name = top3[0]["medicine_name"]
                    else:
                        name = "Unclear"
                        unclear_regions += 1
                    hw_candidates.append({
                        "name": name,
                        "probability": top_p,
                        "top3": top3,
                        "source": "handwriting_model",
                        "ocr_confidence": round(box_conf, 4),
                    })
        else:
            for b in boxes:
                try:
                    bbox = b[0]
                    text = b[1]
                    conf = float(b[2]) if len(b) > 2 else 0.0
                except (IndexError, TypeError):
                    continue

                # Gate: classify only if EasyOCR confidence < HW_GATE
                if conf >= HW_GATE:
                    continue
                # Gate: exclude known structured tokens, headers, digits, etc.
                if is_known_or_structured_box(text, all_meds):
                    continue

                crop = crop_box(img, bbox, pad=5) if img is not None else Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="white")
                target = crop if crop is not None else (img if img is not None else Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="white"))
                top3 = classify_handwritten_medicine(target)
                if top3:
                    top_p = float(top3[0]["probability"])
                    if top_p >= HANDWRITING_CONFIDENCE_THRESHOLD:
                        name = top3[0]["medicine_name"]
                    else:
                        name = "Unclear"
                        unclear_regions += 1
                    hw_candidates.append({
                        "name": name,
                        "probability": top_p,
                        "top3": top3,
                        "source": "handwriting_model",
                        "ocr_confidence": round(conf, 4),
                    })

    # Sort candidates by probability descending
    hw_candidates.sort(key=lambda x: x["probability"], reverse=True)

    # Deduplicate by name (keep highest probability)
    deduped = []
    seen_names = set()
    for c in hw_candidates:
        c_name = c["name"]
        if c_name == "Unclear":
            if "Unclear" not in seen_names:
                seen_names.add("Unclear")
                deduped.append(c)
        else:
            if c_name not in seen_names:
                seen_names.add(c_name)
                deduped.append(c)

    # Keep at most 5 handwriting candidates
    hw_candidates = deduped[:5]

    parsed["handwriting_candidates"] = hw_candidates
    parsed["unclear_regions"] = unclear_regions
    return parsed


def extract_fields(text: Union[str, List[Any]]) -> dict:
    """
    Extract structured fields from OCR text or EasyOCR box list line by line.
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
            "handwriting_candidates": [],
            "unclear_regions": 0,
        }

    all_meds = set(_classes or []) | EXTRA_MEDS

    if isinstance(text, (list, tuple)):
        lines = group_boxes_into_rows(text)
        raw_text = "\n".join(lines)
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
        "handwriting_candidates": [],
        "unclear_regions": 0,
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
