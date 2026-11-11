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
}

# Negation cue patterns (heuristic check within preceding words)
NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bdenies\b",
    r"\bwithout\b",
    r"\bnegative\s+for\b",
    r"\bnot\b",
]
NEGATION_REGEX = re.compile(r"(" + "|".join(NEGATION_PATTERNS) + r")", re.IGNORECASE)


def check_negation(text: str, start: int, max_words_before: int = 5) -> bool:
    """
    Heuristic negation detection.
    Looks up to `max_words_before` words immediately preceding `start` for negation cues,
    or checks if the entity span itself begins with a negation cue (e.g. 'no history of ...').
    Returns True if a negation cue is found without intervening sentence boundaries.
    """
    # 1. Check if the entity text starting at start begins with a negation pattern
    first_two_words = text[start:].strip().split()[:2]
    if first_two_words and NEGATION_REGEX.search(" ".join(first_two_words)):
        return True

    # 2. Check preceding window
    prefix = text[:start]
    clauses = re.split(r"[.;,]", prefix)
    target_clause = clauses[-1] if clauses else prefix
    words = target_clause.strip().split()
    search_window = " ".join(words[-max_words_before:]) if words else ""
    return bool(NEGATION_REGEX.search(search_window))

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

    # Format result structure
    result = {
        "diseases": [],
        "symptoms": [],
        "medications": [],
        "procedures": [],
        "history": [],
        "entities": [],
    }

    for item in merged:
        s = int(item["start"])
        e = int(item["end"])
        span_text = text[s:e].strip()
        etype = item["entity_group"]
        grp = GROUP_MAP.get(etype, "OTHER")
        negated = check_negation(text, s)
        structured_item = {
            "text": span_text,
            "entity_type": etype,
            "group": grp,
            "confidence": round(float(item["score"]), 4),
            "start": s,
            "end": e,
            "negated": negated,
        }
        result["entities"].append(structured_item)
        if grp in KEY_MAP:
            result[KEY_MAP[grp]].append({"text": span_text, "start": s, "end": e, "negated": negated})

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
