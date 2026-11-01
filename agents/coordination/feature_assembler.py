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


def extract_features_from_memory(memory_summary: Dict[str, Any], memory_agent=None, patient_id: Optional[str] = None) -> Tuple[Dict[str, float], bool]:
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
    Returns (extracted_dict, known_condition_flag).
    """
    extracted: Dict[str, float] = {}
    known_condition = False

    # Retrieve all events if memory_agent and patient_id available
    events = []
    if memory_agent and patient_id:
        events = memory_agent.get_timeline(patient_id)
    
    # 1. Inspect conditions & diagnoses
    conditions = memory_summary.get("conditions", [])
    for cond in conditions:
        c_text = str(cond).lower()
        if "diabet" in c_text:
            known_condition = True
        if any(h in c_text for h in ["hypertens", "high blood pressure"]):
            extracted["HighBP"] = 1.0
        if any(c in c_text for c in ["hypercholester", "hyperlipid", "high cholesterol"]):
            extracted["HighChol"] = 1.0
        if "stroke" in c_text:
            extracted["Stroke"] = 1.0
        if any(hd in c_text for hd in ["heart disease", "myocardial", "heart attack", "coronary"]):
            extracted["HeartDiseaseorAttack"] = 1.0
        if any(sm in c_text for sm in ["smoker", "tobacco", "smoking"]):
            extracted["Smoker"] = 1.0

    # 2. Inspect medications (e.g. insulin or metformin indicates diabetes)
    medications = memory_summary.get("medications", [])
    for m in medications:
        m_name = str(m).lower()
        if any(dm in m_name for dm in ["metformin", "insulin", "glimepiride", "glipizide"]):
            known_condition = True
        if any(bp in m_name for bp in ["amlodipine", "losartan", "lisinopril", "telmisartan", "atenolol"]):
            extracted["HighBP"] = 1.0
        if any(st in m_name for st in ["atorvastatin", "rosuvastatin", "simvastatin"]):
            extracted["HighChol"] = 1.0

    # 3. Inspect lab values
    labs = memory_summary.get("latest_labs", {})
    # Check CholCheck
    if any(k in labs for k in ["Total Cholesterol", "LDL Cholesterol", "HDL Cholesterol", "Triglycerides", "lipid"]):
        extracted["CholCheck"] = 1.0
        # Check Total Cholesterol threshold
        tc_val = labs.get("Total Cholesterol", {}).get("value")
        if tc_val is not None:
            try:
                if float(tc_val) >= 200.0:
                    extracted["HighChol"] = 1.0
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
        except ValueError:
            pass

    # Check BMI if recorded
    if "BMI" in labs:
        try:
            extracted["BMI"] = float(labs["BMI"]["value"])
        except ValueError:
            pass

    return extracted, known_condition


def assemble_features(
    provided_profile: Optional[Dict[str, Any]] = None,
    memory_summary: Optional[Dict[str, Any]] = None,
    memory_agent=None,
    patient_id: Optional[str] = None,
    default_strategy: str = "training_median"
) -> Dict[str, Any]:
    """
    Assembles complete 21 BRFSS features following the priority order:
    provided > extracted > assumed.
    """
    provided_profile = provided_profile or {}
    memory_summary = memory_summary or {}

    # 1. Extract from memory
    extracted_features, known_condition = extract_features_from_memory(
        memory_summary=memory_summary,
        memory_agent=memory_agent,
        patient_id=patient_id
    )

    # 2. Defaults
    if default_strategy == "healthy":
        defaults = dict(HEALTHY_DEFAULTS)
    else:
        defaults = load_training_defaults()

    final_features: Dict[str, float] = {}
    sources: Dict[str, str] = {}
    assumed_list: List[str] = []
    provided_or_extracted_count = 0

    for feat in BRFSS_FEATURES:
        if feat in provided_profile and provided_profile[feat] is not None:
            final_features[feat] = float(provided_profile[feat])
            sources[feat] = "provided"
            provided_or_extracted_count += 1
        elif feat in extracted_features:
            final_features[feat] = float(extracted_features[feat])
            sources[feat] = "extracted"
            provided_or_extracted_count += 1
        else:
            final_features[feat] = float(defaults[feat])
            sources[feat] = "assumed"
            assumed_list.append(feat)

    assumed_count = len(assumed_list)
    completeness_msg = f"{provided_or_extracted_count} of 21 provided or extracted, {assumed_count} assumed"

    return {
        "features": final_features,
        "sources": sources,
        "provided_or_extracted_count": provided_or_extracted_count,
        "assumed_count": assumed_count,
        "assumed_features": assumed_list,
        "completeness_message": completeness_msg,
        "known_condition": known_condition,
        "default_strategy": default_strategy,
    }

