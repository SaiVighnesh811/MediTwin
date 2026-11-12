"""
agents/nlp_agent/nlp_agent.py
MediTwin AI — Chapter 5: NLP Agent (Pretrained Biomedical NER)

DISCLAIMER: This module is for research, screening, and educational aid only.
It is NOT validated for clinical decision-making.

Pretrained Model:
    DistilBERT token classification model from:
    https://huggingface.co/d4data/biomedical-ner-all
    License: Apache-2.0

Public API:
    extract_entities(text: str) -> dict
        Returns:
            diseases:    [{"text": str, "start": int, "end": int}, ...]
            symptoms:    [{"text": str, "start": int, "end": int}, ...]
            medications: [{"text": str, "start": int, "end": int}, ...]
            procedures:  [{"text": str, "start": int, "end": int}, ...]
            entities:    [{"text": str, "entity_type": str, "group": str, "confidence": float, "start": int, "end": int}, ...]
"""

import os, sys, re
import warnings; warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
torch.set_num_threads(1)
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from utils.inference_lock import INFERENCE_LOCK

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "nlp")
MODEL_PATH = os.path.join(MODELS_DIR, "d4data_biomedical_ner_all")

# ── Target Group Mappings ──────────────────────────────────────────────────────
GROUP_MAP = {
    "Disease_disorder": "DISEASE",
    "Sign_symptom": "SYMPTOM",
    "Medication": "MEDICATION",
    "Diagnostic_procedure": "PROCEDURE",
    "Therapeutic_procedure": "PROCEDURE",
    "History": "HISTORY",
}

KEY_MAP = {
    "DISEASE": "diseases",
    "SYMPTOM": "symptoms",
    "MEDICATION": "medications",
    "PROCEDURE": "procedures",
    "HISTORY": "history",
    "FAMILY_HISTORY": "family_history",
}

# ── Clause & Cue Patterns ──────────────────────────────────────────────────────
# Clause ends at ., ;, :, or conjunctions (but, however, although, though, except)
CLAUSE_SPLIT_RE = re.compile(
    r"([;:\n!?]+|\.(?!\d)|\b(?:but|however|although|though|except)\b)",
    re.IGNORECASE
)

# Negation cue pattern (negates entities until end of clause)
NEGATION_CUE_RE = re.compile(
    r"\b(no|not|denies|denied|without|negative\s+for|free\s+of)\b",
    re.IGNORECASE
)

# Family history cues
FAMILY_CUE_RE = re.compile(
    r"\b(father|mother|brother|sister|parent|parents|grandfather|grandmother|grandparent|grandparents|family\s+history\s+of|family\s+history|\bfh\b)\b",
    re.IGNORECASE
)

STOPLIST = {"pt", "pateint", "patient"}
DOSAGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|ml|iu|units?)\b", re.IGNORECASE)
DOSE_PHRASE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|ml)(?:\s+(?:daily|once\s+daily|twice\s+daily|bd|bid|tds|tid|qid|od|hs|prn|[a-z0-9]+))*\s*)$", re.IGNORECASE)
DIGITS_SLASHES_RE = re.compile(r"^[\d/\s\.\-]+$")
DATE_RE = re.compile(
    r"^(?:\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}|(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+\d{1,2})?(?:,?\s+\d{2,4})?)$",
    re.IGNORECASE
)
DROP_SINGLE_WORDS = {"father", "mother", "sister", "brother", "mild", "moderate", "severe", "bp", "blood pressure"}

# ── Rule Lexicons (abbreviation/term, canonical_text, entity_type, group) ──────
RULE_LEXICONS = [
    # Diseases / Conditions
    ("htn", "hypertension", "Disease_disorder", "DISEASE"),
    ("t2dm", "type 2 diabetes mellitus", "Disease_disorder", "DISEASE"),
    ("t1dm", "diabetes mellitus", "Disease_disorder", "DISEASE"),
    ("dm", "diabetes mellitus", "Disease_disorder", "DISEASE"),
    ("cad", "coronary artery disease", "Disease_disorder", "DISEASE"),
    ("mi", "myocardial infarction", "Disease_disorder", "DISEASE"),
    ("ckd", "chronic kidney disease", "Disease_disorder", "DISEASE"),
    ("copd", "chronic obstructive pulmonary disease", "Disease_disorder", "DISEASE"),
    ("hld", "hyperlipidemia", "Disease_disorder", "DISEASE"),
    ("hypertention", "hypertension", "Disease_disorder", "DISEASE"),
    ("diabetis", "diabetes mellitus", "Disease_disorder", "DISEASE"),
    # Symptoms
    ("sob", "shortness of breath", "Sign_symptom", "SYMPTOM"),
    ("shortness of breath", "shortness of breath", "Sign_symptom", "SYMPTOM"),
    ("chest pain", "chest pain", "Sign_symptom", "SYMPTOM"),
    ("sore throat", "sore throat", "Sign_symptom", "SYMPTOM"),
    ("fever", "fever", "Sign_symptom", "SYMPTOM"),
    ("nausea", "nausea", "Sign_symptom", "SYMPTOM"),
    ("vomiting", "vomiting", "Sign_symptom", "SYMPTOM"),
    ("cough", "cough", "Sign_symptom", "SYMPTOM"),
    ("headache", "headache", "Sign_symptom", "SYMPTOM"),
    ("dizziness", "dizziness", "Sign_symptom", "SYMPTOM"),
    ("fatigue", "fatigue", "Sign_symptom", "SYMPTOM"),
    ("diarrhea", "diarrhea", "Sign_symptom", "SYMPTOM"),
    ("rash", "rash", "Sign_symptom", "SYMPTOM"),
]
# Sort by pattern length descending
RULE_LEXICONS = sorted(RULE_LEXICONS, key=lambda x: len(x[0]), reverse=True)

_all_known_meds_cache = None


def get_known_medicines() -> set:
    """Returns set of known medicine names (78 classes + EXTRA_MEDS)."""
    global _all_known_meds_cache
    if _all_known_meds_cache is not None:
        return _all_known_meds_cache
    classes = set()
    try:
        from agents.ocr_agent.ocr_agent import EXTRA_MEDS, CKPT_PATH
        if os.path.exists(CKPT_PATH):
            ckpt = torch.load(CKPT_PATH, map_location="cpu")
            classes = set(ckpt.get("classes", []))
        _all_known_meds_cache = classes | EXTRA_MEDS
    except Exception:
        _all_known_meds_cache = set()
    return _all_known_meds_cache


def split_clauses(text: str) -> list:
    """
    Splits text into clauses delimited by ., ;, :, \\n, !, ? or conjunctions
    (but, however, although, though, except). Lists with commas or 'and' remain
    inside the same clause.
    Returns list of (c_start, c_end, clause_text).
    """
    clauses = []
    last_end = 0
    for m in CLAUSE_SPLIT_RE.finditer(text):
        is_conjunction = bool(re.match(r"\b(?:but|however|although|though|except)\b", m.group(), re.I))
        if is_conjunction:
            c_text = text[last_end:m.start()]
            if c_text.strip():
                clauses.append((last_end, m.start(), c_text))
            last_end = m.start()
        else:
            c_text = text[last_end:m.end()]
            if c_text.strip():
                clauses.append((last_end, m.end(), c_text))
            last_end = m.end()
    if last_end < len(text):
        c_text = text[last_end:]
        if c_text.strip():
            clauses.append((last_end, len(text), c_text))
    return clauses


def get_clause_for_span(clauses: list, start: int, end: int):
    for c_start, c_end, c_text in clauses:
        if c_start <= start and end <= c_end:
            return (c_start, c_end, c_text)
    for c_start, c_end, c_text in clauses:
        if c_start <= start < c_end:
            return (c_start, c_end, c_text)
    return (0, len(clauses[-1][2]) if clauses else 0, "")


def _in_same_sentence(text: str, s1: int, e1: int, s2: int, e2: int) -> bool:
    """Checks whether two spans reside within the same sentence (no sentence terminators between them)."""
    if s1 > s2:
        s1, e1, s2, e2 = s2, e2, s1, e1
    inter = text[e1:s2]
    return not bool(re.search(r"[!?\n]|(?<!\d)\.(?!\d)", inter))


def evaluate_span(clauses: list, start: int, end: int, text: str = ""):
    """
    Evaluates whether an entity span has subject 'family' or 'patient',
    and whether it is negated (cue preceding entity in clause).
    """
    c_start, c_end, c_text = get_clause_for_span(clauses, start, end)
    is_family = bool(FAMILY_CUE_RE.search(c_text))
    if not is_family and text:
        # Check if preceded in the same sentence by a family history cue
        last_break = max([text.rfind(sep, 0, start) for sep in (".", "\n", "!", "?")] + [-1])
        prefix = text[last_break + 1:start]
        if FAMILY_CUE_RE.search(prefix) and not re.search(r"\b(?:but|however|although|though|except)\b", prefix, re.I):
            is_family = True
    subject = "family" if is_family else "patient"

    clause_neg_cues = [c_start + m.start() for m in NEGATION_CUE_RE.finditer(c_text)]
    negated = any(cue_pos <= start or (start <= cue_pos < end) for cue_pos in clause_neg_cues)

    return subject, negated


def find_rule_entities(text: str) -> list:
    """
    Finds rule entities in text (abbreviations and missed symptoms) with character spans.
    """
    rule_entities = []
    matched_spans = []

    for pattern, canonical, etype, grp in RULE_LEXICONS:
        regex = re.compile(r"\b" + re.escape(pattern) + r"\b", re.IGNORECASE)
        for m in regex.finditer(text):
            s, e = m.start(), m.end()
            if any(ms <= s and e <= me for ms, me in matched_spans):
                continue
            matched_spans.append((s, e))
            rule_entities.append({
                "text": canonical,
                "raw_text": m.group(0),
                "entity_type": etype,
                "group": grp,
                "confidence": 1.0,
                "start": s,
                "end": e,
                "source": "rule",
            })
    return rule_entities


def check_negation(text: str, start: int, max_words_before: int = 5) -> bool:
    """Backward compatibility wrapper for negation detection."""
    clauses = split_clauses(text)
    _, negated = evaluate_span(clauses, start, start + 1)
    return negated

# ── Module-level cached model (lazy singleton) ─────────────────────────────────
_model     = None
_tokenizer = None
_pipeline  = None


def _load():
    """Load pretrained DistilBERT token classifier locally on CPU."""
    global _model, _tokenizer, _pipeline
    if _pipeline is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Pretrained NLP model not found at {MODEL_PATH}. "
            "Please download d4data/biomedical-ner-all to models/nlp/d4data_biomedical_ner_all first."
        )

    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        _model = AutoModelForTokenClassification.from_pretrained(
            MODEL_PATH, local_files_only=True, use_safetensors=True
        )
        _model.eval()
        _pipeline = pipeline(
            "token-classification",
            model=_model,
            tokenizer=_tokenizer,
            aggregation_strategy="max",
            stride=128,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load NLP model from {MODEL_PATH}: {e}")


def extract_entities(text: str) -> dict:
    """
    Extract biomedical entities with character spans from input text.
    Handles long inputs using overlapping chunking (max_length=256, stride=128).
    Protected by process-wide INFERENCE_LOCK.

    Args:
        text: Clinical note, case report, or medical sentence.

    Returns:
        dict with:
            diseases:    list of {"text": str, "start": int, "end": int}
            symptoms:    list of {"text": str, "start": int, "end": int}
            medications: list of {"text": str, "start": int, "end": int}
            procedures:  list of {"text": str, "start": int, "end": int}
            entities:    list of {"text": str, "entity_type": str, "group": str, "confidence": float, "start": int, "end": int}
    """
    if text is None:
        raise ValueError("Input text cannot be None")
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    if not text.strip():
        return {
            "diseases": [],
            "symptoms": [],
            "medications": [],
            "procedures": [],
            "history": [],
            "family_history": [],
            "entities": [],
        }

    _load()

    with INFERENCE_LOCK:
        raw_output = _pipeline(text)

    # Merge adjacent same-label pieces
    merged = []
    for ent in raw_output:
        if merged and merged[-1]["entity_group"] == ent["entity_group"] and ent["start"] <= merged[-1]["end"] + 1:
            gap = text[merged[-1]["end"]:ent["start"]]
            if gap in ("", " "):
                prev = merged[-1]
                prev["end"] = ent["end"]
                prev["score"] = max(float(prev["score"]), float(ent["score"]))
                continue
        merged.append(dict(ent))

    clauses = split_clauses(text)
    rule_ents = find_rule_entities(text)

    claimed_rule_indices = set()
    processed_entities = []

    # 1. Process merged model entities
    for item in merged:
        s = int(item["start"])
        e = int(item["end"])
        span_text = text[s:e].strip()
        etype = item["entity_group"]
        grp = GROUP_MAP.get(etype, "OTHER")
        src = "model"

        # Check if overlaps with any rule entity
        for r_idx, rent in enumerate(rule_ents):
            if not (e <= rent["start"] or s >= rent["end"]):
                span_text = rent["text"]
                etype = rent["entity_type"]
                grp = rent["group"]
                src = "rule"
                claimed_rule_indices.add(r_idx)
                break

        processed_entities.append({
            "text": span_text,
            "entity_type": etype,
            "group": grp,
            "confidence": round(float(item["score"]), 4),
            "start": s,
            "end": e,
            "source": src,
        })

    # 2. Add unclaimed rule entities
    for r_idx, rent in enumerate(rule_ents):
        if r_idx not in claimed_rule_indices:
            processed_entities.append(rent)

    # 3. Attach dose phrases from OTHER entities to nearest MEDICATION in the same sentence
    med_candidates = [e for e in processed_entities if e.get("group") == "MEDICATION"]
    for ent in processed_entities:
        if ent.get("group") == "OTHER":
            t_str = ent.get("text", "").strip()
            if DOSE_PHRASE_RE.search(t_str) or re.search(r"^\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|ml)\b", t_str, re.I):
                # Find nearest medication in same sentence
                best_med = None
                best_dist = float("inf")
                for med in med_candidates:
                    if _in_same_sentence(text, ent["start"], ent["end"], med["start"], med["end"]):
                        dist = min(abs(ent["start"] - med["end"]), abs(med["start"] - ent["end"]))
                        if dist < best_dist:
                            best_dist = dist
                            best_med = med
                if best_med is not None and "dose" not in best_med:
                    best_med["dose"] = t_str

    # 4. Filter and evaluate
    all_meds = get_known_medicines()
    try:
        from agents.ocr_agent.ocr_agent import match_medicine
    except Exception:
        match_medicine = None

    valid_entities = []
    for ent in processed_entities:
        ent_text = ent["text"].strip()
        lower_text = ent_text.lower()

        # Drop group OTHER entirely (never returned in entities)
        if ent.get("group") == "OTHER":
            continue

        # Drop punctuation-only
        if not re.search(r"[a-zA-Z0-9]", ent_text):
            continue

        # Drop 2 characters or fewer (e.g. "Pt", "BP", "/")
        if len(ent_text) <= 2:
            continue

        # Drop single words: Father, Mother, Sister, Brother, mild, moderate, severe, etc.
        if lower_text in DROP_SINGLE_WORDS or lower_text in STOPLIST:
            continue

        # Drop anything that is only a date or only digits/slashes (e.g. 12/03/2025, 150/95)
        if DIGITS_SLASHES_RE.match(ent_text) or DATE_RE.match(lower_text):
            continue

        # Medication filter:
        # A MEDICATION entity must match the known medicine list (with same fuzzy match),
        # or be followed within 3 tokens by a dose, or have an attached dose; otherwise drop it.
        if ent["group"] == "MEDICATION":
            is_known = False
            if match_medicine is not None and all_meds:
                matched_name, _, _, _ = match_medicine(ent_text, all_meds)
                if matched_name:
                    is_known = True

            has_dose = bool(ent.get("dose"))
            ent_end = ent.get("end", 0)
            if not has_dose and ent_end <= len(text):
                trailing_tokens = text[ent_end:].strip().split()[:3]
                if trailing_tokens and DOSAGE_RE.search(" ".join(trailing_tokens)):
                    has_dose = True

            if not is_known and not has_dose:
                continue

        subject, negated = evaluate_span(clauses, ent["start"], ent["end"], text=text)
        ent["subject"] = subject
        ent["negated"] = negated
        valid_entities.append(ent)

    # 4. Deduplicate by (lowercase text, group, negated, subject) with count; cap at 30
    deduped = []
    seen = {}

    for ent in valid_entities:
        key = (
            ent["text"].strip().lower(),
            ent["group"],
            bool(ent.get("negated", False)),
            ent.get("subject", "patient"),
        )
        if key in seen:
            idx = seen[key]
            deduped[idx]["count"] += 1
            deduped[idx]["confidence"] = max(deduped[idx]["confidence"], ent["confidence"])
        else:
            c_ent = dict(ent)
            c_ent["count"] = 1
            seen[key] = len(deduped)
            deduped.append(c_ent)

    deduped = deduped[:30]

    result = {
        "diseases": [],
        "symptoms": [],
        "medications": [],
        "procedures": [],
        "history": [],
        "family_history": [],
        "entities": deduped,
    }

    for ent in deduped:
        subj = ent.get("subject", "patient")
        grp = ent["group"]
        c_item = {
            "text": ent["text"],
            "start": ent.get("start", 0),
            "end": ent.get("end", 0),
            "negated": ent.get("negated", False),
            "count": ent.get("count", 1),
            "subject": subj,
            "source": ent.get("source", "model"),
        }
        if "dose" in ent:
            c_item["dose"] = ent["dose"]
        if subj == "family":
            result["family_history"].append(c_item)
        elif grp == "DISEASE":
            result["diseases"].append(c_item)
        elif grp == "SYMPTOM":
            result["symptoms"].append(c_item)
        elif grp == "MEDICATION":
            result["medications"].append(c_item)
        elif grp == "PROCEDURE":
            result["procedures"].append(c_item)
        elif grp == "HISTORY":
            result["history"].append(c_item)

    return result


# ── Direct Self-Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_sentence = (
        "The patient was diagnosed with hypertension and type 2 diabetes mellitus. "
        "She was prescribed Metformin 500 mg twice daily. "
        "A coronary angiography was performed."
    )
    print("Input:", test_sentence)
    res = extract_entities(test_sentence)
    print("\nCategorized:")
    for k in ["diseases", "symptoms", "medications", "procedures"]:
        print(f"  {k}: {res[k]}")
    print("\nAll structured entities:")
    for ent in res["entities"]:
        print(f"  {ent['text']:<25} [{ent['group']:<10} / {ent['entity_type']:<22}] conf={ent['confidence']:.3f} ({ent['start']}:{ent['end']})")
