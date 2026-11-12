"""
agents/coordination/feature_assembler.py
MediTwin AI — Feature Assembler (Chapter 8)

Builds the 21 BRFSS survey features in priority order:
1. Form / explicitly provided values (highest priority)
2. Values extracted from clinical documents via Patient Memory Agent
3. Defaults (training set median/mode from data/processed/diabetes_labelled.csv)

Each feature tracks its origin source: 'provided' | 'extracted' | 'assumed'.
Also tracks whether a documented diabetes diagnosis or diabetes medication is present
as an outcome (known_condition=True), rather than an input feature.
"""

import os
import sys
import json
import re
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULTS_FILE = os.path.join(PROJECT_ROOT, "models", "risk_prediction_v2", "training_feature_defaults.json")

BRFSS_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

HEALTHY_DEFAULTS = {
    'HighBP': 0.0,
    'HighChol': 0.0,
    'CholCheck': 1.0,
    'BMI': 22.0,
    'Smoker': 0.0,
    'Stroke': 0.0,
    'HeartDiseaseorAttack': 0.0,
    'PhysActivity': 1.0,
    'Fruits': 1.0,
    'Veggies': 1.0,
    'HvyAlcoholConsump': 0.0,
    'AnyHealthcare': 1.0,
    'NoDocbcCost': 0.0,
    'GenHlth': 1.0,       # 1 = Excellent
    'MentHlth': 0.0,      # 0 days bad mental health
    'PhysHlth': 0.0,      # 0 days bad physical health
    'DiffWalk': 0.0,
    'Sex': 0.0,
    'Age': 5.0,           # middle category (Age 40-44)
    'Education': 6.0,     # College 4+ years
    'Income': 8.0,        # >= $75,000
}


def load_training_defaults() -> Dict[str, float]:
    """Loads median/mode defaults computed from training set."""
    if os.path.exists(DEFAULTS_FILE):
        with open(DEFAULTS_FILE, "r") as f:
            data = json.load(f)
            return {k: float(v["default"]) for k, v in data.items()}
    # Fallback if json not found
    return {
        'HighBP': 0.0, 'HighChol': 0.0, 'CholCheck': 1.0, 'BMI': 27.0,
        'Smoker': 0.0, 'Stroke': 0.0, 'HeartDiseaseorAttack': 0.0,
        'PhysActivity': 1.0, 'Fruits': 1.0, 'Veggies': 1.0,
        'HvyAlcoholConsump': 0.0, 'AnyHealthcare': 1.0, 'NoDocbcCost': 0.0,
        'GenHlth': 2.0, 'MentHlth': 0.0, 'PhysHlth': 0.0, 'DiffWalk': 0.0,
        'Sex': 0.0, 'Age': 8.0, 'Education': 5.0, 'Income': 7.0
    }


def extract_features_from_memory(
    memory_summary: Dict[str, Any],
    memory_agent=None,
    patient_id: Optional[str] = None,
    return_details: bool = False,
) -> Any:
    """
    Extracts inferrable BRFSS features from Patient Memory record.
    Extraction rules:
      - HighBP: documented non-negated hypertension or SBP>=130 / DBP>=80
      - HighChol: documented non-negated hyperlipidemia/hypercholesterolemia or total chol >= 200 mg/dL
      - CholCheck: 1 if lipid panel / cholesterol lab is recorded
      - Stroke: documented non-negated stroke
      - HeartDiseaseorAttack: documented non-negated myocardial infarction / coronary heart disease
      - Smoker: documented active smoking / tobacco use
      - BMI: stated in memory or labs
    Also checks for diabetes diagnosis or diabetes medications -> sets known_condition=True.
    Returns (extracted_dict, known_condition_flag) or (extracted_dict, known_condition_flag, details).
    """
    extracted: Dict[str, float] = {}
    details: Dict[str, Dict[str, Any]] = {}
    known_condition = False

    # Retrieve all events if memory_agent and patient_id available
    events = []
    if memory_agent and patient_id:
        try:
            events = memory_agent.get_timeline(patient_id)
        except Exception:
            events = []

    # 1. Inspect conditions & diagnoses
    conditions = memory_summary.get("conditions", [])
    for cond in conditions:
        c_text = str(cond).lower()
        if any(fam in c_text for fam in ["family", "father", "mother", "sister", "brother", "parent", "fh"]):
            continue
        if "diabet" in c_text:
            known_condition = True
        if any(h in c_text for h in ["hypertens", "high blood pressure"]):
            extracted["HighBP"] = 1.0
            details["HighBP"] = {"saved_text": str(cond), "date": None}
        if any(c in c_text for c in ["hypercholester", "hyperlipid", "high cholesterol"]):
            extracted["HighChol"] = 1.0
            details["HighChol"] = {"saved_text": str(cond), "date": None}
        if "stroke" in c_text:
            extracted["Stroke"] = 1.0
            details["Stroke"] = {"saved_text": str(cond), "date": None}
        if any(hd in c_text for hd in ["heart disease", "myocardial", "heart attack", "coronary"]):
            extracted["HeartDiseaseorAttack"] = 1.0
            details["HeartDiseaseorAttack"] = {"saved_text": str(cond), "date": None}
        if any(sm in c_text for sm in ["smoker", "tobacco", "smoking"]):
            extracted["Smoker"] = 1.0
            details["Smoker"] = {"saved_text": str(cond), "date": None}

    # 2. Inspect medications (e.g. insulin or metformin indicates diabetes)
    medications = memory_summary.get("medications", [])
    for m in medications:
        m_name = str(m).lower()
        if any(fam in m_name for fam in ["family", "father", "mother", "sister", "brother", "parent", "fh"]):
            continue
        if any(dm in m_name for dm in ["metformin", "insulin", "glimepiride", "glipizide"]):
            known_condition = True
        if any(bp in m_name for bp in ["amlodipine", "losartan", "lisinopril", "telmisartan", "atenolol"]):
            extracted["HighBP"] = 1.0
            details["HighBP"] = {
                "saved_text": f"Medication: {str(m)}",
                "date": None,
                "inferred_from_medicine": True,
            }
        if any(st in m_name for st in ["atorvastatin", "rosuvastatin", "simvastatin"]):
            extracted["HighChol"] = 1.0
            details["HighChol"] = {
                "saved_text": f"Medication: {str(m)}",
                "date": None,
                "inferred_from_medicine": True,
            }

    # 3. Inspect lab values
    labs = memory_summary.get("latest_labs", {})
    # Check CholCheck
    if any(k in labs for k in ["Total Cholesterol", "LDL Cholesterol", "HDL Cholesterol", "Triglycerides", "lipid"]):
        extracted["CholCheck"] = 1.0
        details["CholCheck"] = {
            "saved_text": "Lipid / Cholesterol lab record",
            "date": labs.get("Total Cholesterol", {}).get("timestamp"),
        }
        # Check Total Cholesterol threshold
        tc_val = labs.get("Total Cholesterol", {}).get("value")
        if tc_val is not None:
            try:
                if float(tc_val) >= 200.0:
                    extracted["HighChol"] = 1.0
                    details["HighChol"] = {
                        "saved_text": f"Total Cholesterol: {tc_val} {labs.get('Total Cholesterol', {}).get('unit', '')}".strip(),
                        "date": labs.get("Total Cholesterol", {}).get("timestamp"),
                    }
            except ValueError:
                pass

    # Check Blood Pressure labs if present
    bp_val = labs.get("Blood Pressure", {}).get("value")
    if bp_val and "/" in str(bp_val):
        try:
            parts = str(bp_val).split("/")
            sbp = float(parts[0].strip())
            dbp = float(parts[1].strip())
            if sbp >= 130.0 or dbp >= 80.0:
                extracted["HighBP"] = 1.0
                details["HighBP"] = {
                    "saved_text": f"Blood Pressure: {bp_val}",
                    "date": labs.get("Blood Pressure", {}).get("timestamp"),
                }
        except ValueError:
            pass

    # Check BMI if recorded
    if "BMI" in labs:
        try:
            bmi_val = float(labs["BMI"]["value"])
            extracted["BMI"] = bmi_val
            details["BMI"] = {
                "saved_text": f"BMI: {bmi_val}",
                "date": labs.get("BMI", {}).get("timestamp"),
            }
        except ValueError:
            pass

    # 4. Fill timeline timestamps if events available
    if events:
        for ev in events:
            ts = ev.get("timestamp")
            data = ev.get("data", {})
            d_name = str(data.get("diagnosis") or data.get("name") or "").lower()
            m_name = str(data.get("medicine") or data.get("name") or "").lower()
            for feat, d_info in details.items():
                if not d_info.get("date") and ts:
                    st = d_info.get("saved_text", "").lower()
                    if d_name and d_name in st:
                        d_info["date"] = ts
                    elif m_name and m_name in st:
                        d_info["date"] = ts

    if return_details:
        return extracted, known_condition, details
    return extracted, known_condition


def extract_features_from_ocr(ocr_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Extracts BRFSS features directly from OCR findings from the CURRENT document upload."""
    if not ocr_data or not isinstance(ocr_data, dict):
        return {}
    extracted: Dict[str, float] = {}

    # 1. Labs
    labs = ocr_data.get("labs", [])
    if isinstance(labs, list):
        for lab in labs:
            if not isinstance(lab, dict):
                continue
            tname = str(lab.get("test", "")).lower()
            val = lab.get("value")
            if any(k in tname for k in ["cholesterol", "lipid", "ldl", "hdl", "triglyceride"]):
                extracted["CholCheck"] = 1.0
                if "total cholesterol" in tname and val is not None:
                    try:
                        if float(val) >= 200.0:
                            extracted["HighChol"] = 1.0
                    except (ValueError, TypeError):
                        pass
            if any(k in tname for k in ["blood pressure", "bp"]):
                if val and "/" in str(val):
                    try:
                        parts = str(val).split("/")
                        if float(parts[0].strip()) >= 130.0 or float(parts[1].strip()) >= 80.0:
                            extracted["HighBP"] = 1.0
                    except (ValueError, TypeError):
                        pass
            if "bmi" in tname and val is not None:
                try:
                    extracted["BMI"] = float(val)
                except (ValueError, TypeError):
                    pass

    # 2. Medicines (only confirmed printed text; never handwriting candidates)
    medicines = ocr_data.get("medicines", [])
    if isinstance(medicines, list):
        for med in medicines:
            if isinstance(med, dict) and (med.get("is_handwritten") or med.get("candidate")):
                continue
            m_name = (med.get("medicine", "") if isinstance(med, dict) else str(med)).lower()
            if any(bp in m_name for bp in ["amlodipine", "losartan", "lisinopril", "telmisartan", "atenolol"]):
                extracted["HighBP"] = 1.0
            if any(st in m_name for st in ["atorvastatin", "rosuvastatin", "simvastatin"]):
                extracted["HighChol"] = 1.0

    return extracted


def assemble_features(
    provided_profile: Optional[Dict[str, Any]] = None,
    memory_summary: Optional[Dict[str, Any]] = None,
    memory_agent=None,
    patient_id: Optional[str] = None,
    default_strategy: str = "training_median",
    use_memory: bool = True,
    current_doc_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assembles complete 21 BRFSS features following the priority order:
    provided > extracted (current document) > saved_record (memory) > assumed.
    """
    provided_profile = provided_profile or {}
    memory_summary = memory_summary or {}
    current_doc_features = current_doc_features or {}

    # 1. Extract from memory with details
    extracted_features, known_condition, memory_details = extract_features_from_memory(
        memory_summary=memory_summary,
        memory_agent=memory_agent,
        patient_id=patient_id,
        return_details=True,
    )

    # 2. Defaults
    if default_strategy == "healthy":
        defaults = dict(HEALTHY_DEFAULTS)
    else:
        defaults = load_training_defaults()

    final_features: Dict[str, float] = {}
    sources: Dict[str, str] = {}
    saved_records: List[Dict[str, Any]] = []
    assumed_list: List[str] = []
    provided_or_extracted_count = 0

    for feat in BRFSS_FEATURES:
        if feat in provided_profile and provided_profile[feat] is not None:
            final_features[feat] = float(provided_profile[feat])
            sources[feat] = "provided"
            provided_or_extracted_count += 1
        elif feat in current_doc_features and current_doc_features[feat] is not None:
            final_features[feat] = float(current_doc_features[feat])
            sources[feat] = "extracted"
            provided_or_extracted_count += 1
        elif use_memory and feat in extracted_features:
            final_features[feat] = float(extracted_features[feat])
            sources[feat] = "saved_record"
            provided_or_extracted_count += 1
            rec_detail = memory_details.get(feat, {})
            record_item = {
                "feature": feat,
                "value": float(extracted_features[feat]),
                "saved_text": rec_detail.get("saved_text", "Saved clinical record"),
                "date": rec_detail.get("date"),
            }
            if rec_detail.get("inferred_from_medicine"):
                record_item["inferred_from_medicine"] = True
            saved_records.append(record_item)
        else:
            final_features[feat] = float(defaults[feat])
            sources[feat] = "assumed"
            assumed_list.append(feat)

    assumed_count = len(assumed_list)
    completeness_msg = f"{provided_or_extracted_count} of 21 provided or extracted, {assumed_count} assumed"

    return {
        "features": final_features,
        "sources": sources,
        "saved_records": saved_records,
        "provided_or_extracted_count": provided_or_extracted_count,
        "assumed_count": assumed_count,
        "assumed_features": assumed_list,
        "completeness_message": completeness_msg,
        "known_condition": known_condition,
        "default_strategy": default_strategy,
        "use_memory": use_memory,
    }

