"""
agents/recommendation_agent/recommendation_agent.py
Recommendation Agent for MediTwin AI.

Produces structured, strictly educational health and lifestyle guidance.
Never provides diagnosis, prescription, or clinical treatment advice.
Safety rules checked before every response.
"""

import os
import re
from typing import Any, Dict, List, Optional
from agents.recommendation_agent.rules import (
    RULES,
    GLOBAL_DISCLAIMER,
    SEE_A_DOCTOR_NOTE_HIGH,
    SEE_A_DOCTOR_NOTE_MEDIUM,
    SEE_A_DOCTOR_NOTE_ROUTINE,
)

# Mandatory safety check: forbidden clinical assertion phrases
FORBIDDEN_PATTERNS = [
    re.compile(r'\bdiagnos(?:e|is|ed|ing)\b', re.IGNORECASE),
    re.compile(r'\bprescrib(?:e|ed|ing|tion)\b', re.IGNORECASE),
    re.compile(r'\bdos(?:e|age|ing)\b', re.IGNORECASE),
    re.compile(r'\byou have\b', re.IGNORECASE),
    re.compile(r'\bstop taking\b', re.IGNORECASE),
    re.compile(r'\bstart taking\b', re.IGNORECASE),
]


def check_safety(text: str) -> List[str]:
    """Returns any matched forbidden phrases in generated guidance."""
    violations = []
    for pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(text)
        if match:
            violations.append(match.group(0))
    return violations


class RecommendationAgent:
    """Generates educational guidance from risk, XAI, imaging, and patient memory inputs."""

    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None):
        self.rules = rules or RULES

    def generate_recommendations(
        self,
        risk_level: str,
        probabilities: Optional[Dict[str, float]] = None,
        top_shap_features: Optional[List[Any]] = None,
        memory_summary: Optional[Dict[str, Any]] = None,
        imaging_result: Optional[Dict[str, Any]] = None,
        patient_profile: Optional[Dict[str, Any]] = None,
        assumed_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates rules against provided context and returns structured guidance.
        """
        mem_sum = memory_summary or {}
        img_res = imaging_result or {}
        profile = patient_profile or {}
        probs = probabilities or {}
        top_shap = top_shap_features or []
        assumed = assumed_features or []

        # Context build for rule evaluation
        ctx = {
            "risk_level": risk_level,
            "probabilities": probs,
            "top_shap_features": top_shap,
            "features": profile,
            "latest_labs": mem_sum.get("latest_labs", {}),
            "diagnoses": mem_sum.get("diagnoses", []),
            "symptoms": mem_sum.get("symptoms", []),
            "medications": mem_sum.get("medications", []),
            "imaging_finding": {
                "finding": img_res.get("predicted_class") or img_res.get("finding"),
                "pneumonia_probability": img_res.get("pneumonia_probability") or img_res.get("probability", 0.0),
            },
        }

        guidance_items = []
        for r in self.rules:
            try:
                if r["condition"](ctx):
                    guidance_items.append({
                        "rule_id": r["rule_id"],
                        "category": r.get("category", "general"),
                        "text": r["guidance_text"],
                        "reason": r["reason"],
                        "source_note": r["source_note"],
                    })
            except Exception:
                continue

        # Determine doctor note based on severity
        is_imaging_positive = ctx["imaging_finding"]["finding"] == "PNEUMONIA" or float(ctx["imaging_finding"]["pneumonia_probability"] or 0) >= 0.5
        if risk_level == "High" or is_imaging_positive:
            doc_note = SEE_A_DOCTOR_NOTE_HIGH
        elif risk_level == "Medium":
            doc_note = SEE_A_DOCTOR_NOTE_MEDIUM
        else:
            doc_note = SEE_A_DOCTOR_NOTE_ROUTINE

        # Informative note about assumed/unverified risk factors
        assumed_factors_note = None
        if assumed:
            assumed_factors_note = (
                f"Note on completeness: {len(assumed)} of 21 survey risk factors were unverified from records "
                f"and assumed from standard baseline defaults ({', '.join(sorted(assumed))}). "
                "Consult a healthcare professional to review these specific parameters."
            )

        # Validate safety across all output texts
        texts_to_check = [item["text"] for item in guidance_items] + [doc_note]
        if assumed_factors_note:
            texts_to_check.append(assumed_factors_note)
        combined_text = " ".join(texts_to_check)
        violations = check_safety(combined_text)
        if violations:
            raise ValueError(f"Safety violation detected in generated guidance: {violations}")

        return {
            "risk_level": risk_level,
            "risk_probabilities": probs,
            "guidance": guidance_items,
            "see_a_doctor_note": doc_note,
            "unverified_factors_note": assumed_factors_note,
            "assumed_features": assumed,
            "disclaimer": GLOBAL_DISCLAIMER,
        }

