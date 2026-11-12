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
    CONFIRMED_MEDICINE_LOOKUP,
    MEDICINE_CONDITION_MAP,
    FIXED_MED_DISCLAIMER,
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
    """
    Returns any matched forbidden phrases in generated guidance.
    Allows informational extraction labels such as 'dose unclear' and the
    mandatory fixed medicine disclaimer.
    """
    # Sanitize benign extracted medicine phrases and fixed disclaimer
    sanitized = re.sub(r'\bdose\s+unclear\b', '', text, flags=re.IGNORECASE)
    sanitized = re.sub(r'Extracted from your document[^.]*\.', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'\bDo not start, stop or change any medicine because of this page\.', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'\bdose\b(?=[\s:,0-9mg])', '', sanitized, flags=re.IGNORECASE)

    violations = []
    for pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(sanitized)
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
        sources: Optional[Dict[str, str]] = None,
        saved_records: Optional[List[Dict[str, Any]]] = None,
        document_medicines: Optional[List[Any]] = None,
        document_labs: Optional[List[Any]] = None,
        ocr_result: Optional[Dict[str, Any]] = None,
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
        src_map = sources or {}
        saved_recs = saved_records or []

        # Document labs from current upload
        doc_labs = document_labs
        if doc_labs is None and ocr_result and isinstance(ocr_result, dict):
            doc_labs = ocr_result.get("labs", [])
        doc_labs = doc_labs or []

        # Context build for rule evaluation
        ctx = {
            "risk_level": risk_level,
            "probabilities": probs,
            "top_shap_features": top_shap,
            "features": profile,
            "sources": src_map,
            "saved_records": saved_recs,
            "document_labs": doc_labs,
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
                    rule_id = r["rule_id"]
                    text = r["guidance_text"]
                    based_on = r["based_on_fn"](ctx) if "based_on_fn" in r else "general advice (no personal data used)"

                    # ---------------------------------------------------------
                    # CHAPTER 17 & 17b: Rule card based_on and dynamic wording
                    # ---------------------------------------------------------

                    # 1. Glycemic card
                    if rule_id == "REC_GLYCEMIC_LAB":
                        # Check document labs first (CURRENT upload)
                        doc_hba1c = next((l for l in doc_labs if any(k in str(l.get("test", "")).lower() for k in ["hba1c", "a1c"])), None)
                        doc_fbg = next((l for l in doc_labs if any(k in str(l.get("test", "")).lower() for k in ["glucose", "sugar", "fbg"])), None)
                        
                        has_elevated_lab = False
                        if doc_hba1c and float(doc_hba1c.get("value", 0) or 0) >= 5.7:
                            has_elevated_lab = True
                            val = doc_hba1c.get("value")
                            unit = doc_hba1c.get("unit") or "%"
                            val_str = f"{val}%" if not str(val).endswith("%") and unit == "%" else f"{val} {unit}".strip()
                            text = f"Glycemic monitoring: Prioritize discussing recent elevated glycemic indicators (HbA1c: {val_str}, uploaded document) with your physician for formal assessment and glycemic health management."
                            based_on = f"uploaded document: HbA1c {val_str}"
                        elif doc_fbg and float(doc_fbg.get("value", 0) or 0) >= 100.0:
                            has_elevated_lab = True
                            val = doc_fbg.get("value")
                            unit = doc_fbg.get("unit") or "mg/dL"
                            val_str = f"{val} {unit}".strip()
                            text = f"Glycemic monitoring: Prioritize discussing recent elevated glycemic indicators (Fasting glucose: {val_str}, uploaded document) with your physician for formal assessment and glycemic health management."
                            based_on = f"uploaded document: Fasting glucose {val_str}"
                        
                        # Then check saved lab records
                        if not has_elevated_lab:
                            labs = ctx.get("latest_labs", {})
                            hba1c_val = labs.get("HbA1c", {}).get("value")
                            fbg_val = labs.get("Fasting Blood Glucose", {}).get("value")
                            if hba1c_val is not None and float(hba1c_val or 0) >= 5.7:
                                has_elevated_lab = True
                                date = labs.get("HbA1c", {}).get("timestamp") or "recent"
                                text = f"Glycemic monitoring: Prioritize discussing recent elevated glycemic indicators (HbA1c: {hba1c_val}%, saved lab record, {date}) with your physician for formal assessment and glycemic health management."
                                based_on = f"saved lab record: HbA1c {hba1c_val}% ({date})"
                            elif fbg_val is not None and float(fbg_val or 0) >= 100.0:
                                has_elevated_lab = True
                                date = labs.get("Fasting Blood Glucose", {}).get("timestamp") or "recent"
                                text = f"Glycemic monitoring: Prioritize discussing recent elevated glycemic indicators (Fasting glucose: {fbg_val} mg/dL, saved lab record, {date}) with your physician for formal assessment and glycemic health management."
                                based_on = f"saved lab record: Fasting glucose {fbg_val} mg/dL ({date})"

                        if not has_elevated_lab:
                            # Diabetes answer exists without elevated lab reading
                            has_dm_ans = (
                                float(profile.get("Diabetes_binary", 0) or 0) == 1.0
                                or float(profile.get("Diabetes", 0) or 0) in (1.0, 2.0)
                            )
                            if has_dm_ans:
                                text = "Glycemic and nutrition guidance: Emphasize balanced nutrition, low glycemic-index carbohydrates, regular meal timing, and routine health check-ups with your physician."
                                based_on = (
                                    "saved record: diabetes = yes"
                                    if src_map.get("Diabetes_binary") == "saved_record" or src_map.get("Diabetes") == "saved_record"
                                    else "your answer: diabetes = yes"
                                )
                            else:
                                text = "Glycemic and nutrition guidance: Emphasize balanced nutrition, low glycemic-index carbohydrates, regular meal timing, and routine health check-ups with your physician."
                                based_on = "general advice (no personal data used)"

                    # 2. Lipid card
                    elif rule_id == "REC_CHOLESTEROL":
                        # Check document labs (CURRENT upload)
                        doc_tc = next((l for l in doc_labs if any(k in str(l.get("test", "")).lower() for k in ["cholesterol", "lipid"])), None)
                        has_tc_doc = (doc_tc is not None and doc_tc.get("value") is not None and float(doc_tc.get("value", 0) or 0) >= 200.0)
                        
                        # Check saved lab record
                        labs = ctx.get("latest_labs", {})
                        tc_val = labs.get("Total Cholesterol", {}).get("value")
                        has_tc_saved_lab = (tc_val is not None and float(tc_val or 0) >= 200.0)

                        has_highchol_provided = (src_map.get("HighChol") == "provided" and float(profile.get("HighChol", 0) or 0) == 1.0)
                        has_highchol_saved = (src_map.get("HighChol") == "saved_record" and float(profile.get("HighChol", 0) or 0) == 1.0)

                        if has_tc_doc:
                            val = doc_tc.get("value")
                            unit = doc_tc.get("unit") or "mg/dL"
                            val_str = f"{val} {unit}".strip()
                            text = f"Cardiovascular wellness: Total cholesterol indicator ({val_str}, uploaded document). Consider increasing soluble fiber intake (oats, legumes), limiting saturated fats, and reviewing lipid profiles periodically with a healthcare professional."
                            based_on = f"uploaded document: Total Cholesterol {val_str}"
                        elif has_tc_saved_lab:
                            date = labs.get("Total Cholesterol", {}).get("timestamp") or "recent"
                            text = f"Cardiovascular wellness: Total cholesterol indicator ({tc_val} mg/dL, saved lab record, {date}). Consider increasing soluble fiber intake (oats, legumes), limiting saturated fats, and reviewing lipid profiles periodically with a healthcare professional."
                            based_on = f"saved lab record: Total Cholesterol {tc_val} mg/dL ({date})"
                        elif has_highchol_provided:
                            text = "Cardiovascular wellness: Consider increasing soluble fiber intake (oats, legumes), limiting saturated fats, and reviewing lipid profiles periodically with a healthcare professional."
                            based_on = "your answer: high cholesterol = yes"
                        elif has_highchol_saved:
                            text = "Cardiovascular wellness: Consider increasing soluble fiber intake (oats, legumes), limiting saturated fats, and reviewing lipid profiles periodically with a healthcare professional."
                            based_on = "saved record: high cholesterol = yes"
                        else:
                            # General wording without claiming personal data
                            text = "Lipid and heart wellness: Support healthy cholesterol levels by incorporating dietary fiber, heart-healthy unsaturated fats, and regular physical exercise."
                            based_on = "general advice (no personal data used)"

                    # 3. Blood Pressure card
                    elif rule_id == "REC_BP_HIGH":
                        # Check document labs (CURRENT upload)
                        doc_bp = next((l for l in doc_labs if any(k in str(l.get("test", "")).lower() for k in ["blood pressure", "bp"])), None)
                        has_bp_doc = (doc_bp is not None and doc_bp.get("value") is not None)

                        # Check saved lab record
                        labs = ctx.get("latest_labs", {})
                        saved_bp = labs.get("Blood Pressure", {}).get("value")
                        has_bp_saved_lab = (saved_bp is not None)

                        has_highbp_provided = (src_map.get("HighBP") == "provided" and float(profile.get("HighBP", 0) or 0) == 1.0)
                        has_highbp_saved = (src_map.get("HighBP") == "saved_record" and float(profile.get("HighBP", 0) or 0) == 1.0)

                        if has_bp_doc:
                            val = doc_bp.get("value")
                            text = "Blood pressure management: Adopt sodium reduction, DASH dietary patterns (rich in vegetables, fruits, and low-fat dairy), and regular blood pressure monitoring under medical supervision."
                            based_on = f"uploaded document: Blood Pressure {val}"
                        elif has_bp_saved_lab:
                            date = labs.get("Blood Pressure", {}).get("timestamp")
                            date_str = f" ({date})" if date else ""
                            text = "Blood pressure management: Adopt sodium reduction, DASH dietary patterns (rich in vegetables, fruits, and low-fat dairy), and regular blood pressure monitoring under medical supervision."
                            based_on = f"saved lab record: Blood Pressure {saved_bp}{date_str}"
                        elif has_highbp_provided:
                            text = "Blood pressure management: Adopt sodium reduction, DASH dietary patterns (rich in vegetables, fruits, and low-fat dairy), and regular blood pressure monitoring under medical supervision."
                            based_on = "your answer: high blood pressure = yes"
                        elif has_highbp_saved:
                            text = "Blood pressure management: Adopt sodium reduction, DASH dietary patterns (rich in vegetables, fruits, and low-fat dairy), and regular blood pressure monitoring under medical supervision."
                            based_on = "saved record: high blood pressure = yes"
                        else:
                            # General wording without claiming personal data
                            text = "Blood pressure wellness: Maintain a balanced diet low in sodium, engage in regular aerobic physical activity, and monitor blood pressure during routine medical checkups."
                            based_on = "general advice (no personal data used)"

                    # 4. Nutrition card
                    elif rule_id == "REC_BMI_NUTRITION":
                        bmi_val = profile.get("BMI")
                        if bmi_val is not None and float(bmi_val or 0) >= 25.0:
                            if src_map.get("BMI") == "provided":
                                based_on = f"your answer: BMI = {bmi_val}"
                            elif src_map.get("BMI") == "saved_record":
                                based_on = f"saved record: BMI = {bmi_val}"
                            elif src_map.get("BMI") == "extracted":
                                based_on = f"uploaded document: BMI {bmi_val}"
                            else:
                                based_on = "general advice (no personal data used)"
                        else:
                            based_on = "general advice (no personal data used)"

                    # 5. Preventive card
                    elif rule_id == "REC_ROUTINE_CHECKUP":
                        based_on = "general advice (no personal data used)"

                    guidance_items.append({
                        "rule_id": rule_id,
                        "category": r.get("category", "general"),
                        "text": text,
                        "based_on": based_on,
                        "reason": r["reason"],
                        "source_note": r["source_note"],
                    })
            except Exception:
                continue

        # -------------------------------------------------------------
        # Chapter 17: Medicine Guidance (Item 3)
        # -------------------------------------------------------------
        # Gather confirmed medicines from document or saved records.
        # NEVER use handwriting candidates.
        confirmed_raw = []
        doc_med_src = document_medicines
        if doc_med_src is None and ocr_result and isinstance(ocr_result, dict):
            doc_med_src = ocr_result.get("medicines", [])
        
        if doc_med_src:
            for item in doc_med_src:
                if isinstance(item, dict):
                    if item.get("is_handwritten") or item.get("candidate"):
                        continue
                    m_name = item.get("medicine") or item.get("name") or item.get("text")
                    if m_name:
                        confirmed_raw.append({
                            "name": str(m_name).strip(),
                            "dose": item.get("dose"),
                            "frequency": item.get("frequency"),
                            "duration": item.get("duration"),
                            "source": "uploaded document",
                        })
                elif isinstance(item, str) and item.strip():
                    confirmed_raw.append({
                        "name": item.strip(),
                        "dose": None,
                        "frequency": None,
                        "duration": None,
                        "source": "uploaded document",
                    })

        # Also inspect confirmed medicines from patient memory (saved records)
        for item in mem_sum.get("medications", []):
            if isinstance(item, dict):
                m_name = item.get("medicine") or item.get("name") or item.get("text")
                if m_name:
                    confirmed_raw.append({
                        "name": str(m_name).strip(),
                        "dose": item.get("dose"),
                        "frequency": item.get("frequency"),
                        "duration": item.get("duration"),
                        "source": "saved record",
                    })
            elif isinstance(item, str) and item.strip():
                confirmed_raw.append({
                    "name": item.strip(),
                    "dose": None,
                    "frequency": None,
                    "duration": None,
                    "source": "saved record",
                })

        def _normalize_med_name(name_str: str) -> str:
            lower = name_str.lower().strip()
            for key in CONFIRMED_MEDICINE_LOOKUP:
                if key in lower:
                    return key
            return re.sub(r'[^a-z0-9]', '', lower)

        # Count occurrences for duplicate checking (Consistency Note a)
        counts: Dict[str, int] = {}
        for med in confirmed_raw:
            norm = _normalize_med_name(med["name"])
            counts[norm] = counts.get(norm, 0) + 1

        # Build medicine guidance cards
        for idx, med in enumerate(confirmed_raw):
            norm_name = _normalize_med_name(med["name"])
            plain_use = CONFIRMED_MEDICINE_LOOKUP.get(
                norm_name,
                "Listed in your document; ask your doctor or pharmacist what it is for."
            )

            # Build dose info: dose and frequency and duration exactly as extracted, or "dose unclear" when missing
            dose_parts = []
            if med.get("dose"):
                dose_parts.append(str(med["dose"]).strip())
            if med.get("frequency"):
                dose_parts.append(str(med["frequency"]).strip())
            if med.get("duration"):
                dose_parts.append(str(med["duration"]).strip())

            if not med.get("dose"):
                if dose_parts:
                    dose_info = f"dose unclear, {', '.join(dose_parts)}"
                else:
                    dose_info = "dose unclear"
            else:
                dose_info = ", ".join(dose_parts)

            # Consistency notes
            notes = []
            # Note a: Duplicate check
            if counts.get(norm_name, 0) > 1:
                notes.append("listed more than once; please check")

            # Note b: Contradiction check (FIX 2)
            # MUST use ONLY values the user actually provided in this analysis (source "provided"),
            # never assumed defaults and never values filled from saved records or documents.
            if norm_name in MEDICINE_CONDITION_MAP:
                mapping = MEDICINE_CONDITION_MAP[norm_name]
                cond_name = mapping["condition_name"]
                for fkey in mapping["feature_keys"]:
                    is_provided_by_user = (src_map.get(fkey) == "provided")
                    fval = profile.get(fkey)
                    if is_provided_by_user and (fval == 0.0 or fval == 0):
                        notes.append(
                            f"You listed a medicine commonly used for {cond_name} but answered No to {cond_name}; please check your answers."
                        )
                        break

            # Build text exactly per Chapter 17 specification
            text_segments = [
                f"{med['name']}: {dose_info}.",
                f"Common use: {plain_use}.",
                FIXED_MED_DISCLAIMER,
            ]
            for n in notes:
                text_segments.append(f"Note: {n}")
            med_card_text = " ".join(text_segments)

            src_label = med.get("source", "uploaded document")
            dose_label = f" {med['dose']}" if med.get("dose") else ""
            based_on_med = f"{src_label}: {med['name']}{dose_label}"

            guidance_items.append({
                "rule_id": f"REC_MED_{norm_name.upper()}_{idx}",
                "category": "medicines",
                "title": "Medicines found in your document",
                "text": med_card_text,
                "based_on": based_on_med,
                "notes": notes,
                "reason": f"Medicine identified from {src_label}: {med['name']}",
                "source_note": "MediTwin small built-in medicine list (6 names, not a verified drug database)",
            })

        # Ensure every card has a based_on line
        for item in guidance_items:
            if not item.get("based_on"):
                item["based_on"] = "general advice (no personal data used)"

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
