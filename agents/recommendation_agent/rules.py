"""
agents/recommendation_agent/rules.py
Readable, transparent rule base for the Recommendation Agent.
Content is based solely on widely published general public-health guidelines
(CDC, WHO, AHA, ADA) for educational lifestyle and preventive health guidance.

CRITICAL SAFETY CONSTRAINTS:
- General educational guidance only.
- NEVER diagnose, prescribe, adjust dosage, or replace professional care.
- No forbidden phrases (diagnose, prescribe, dose, 'you have', stop taking, start taking).
"""

from typing import Any, Dict, List, Optional

# Mandatory global disclaimer attached to every single recommendation response
GLOBAL_DISCLAIMER = (
    "DISCLAIMER: This system provides general educational health information only and does NOT provide "
    "clinical evaluation, medical management, or medical directives. All risk ratings and findings are automated screening "
    "indicators. Consult a qualified healthcare professional before making any medical, dietary, or medication changes."
)

# Mandatory consultation note for elevated findings
SEE_A_DOCTOR_NOTE_HIGH = (
    "Based on elevated risk indicators or imaging findings, we strongly advise scheduling a comprehensive "
    "evaluation with a qualified healthcare professional or primary care physician for clinical confirmation and guidance."
)

SEE_A_DOCTOR_NOTE_MEDIUM = (
    "Notice: Moderate risk classification acts as a screening indicator (with recognized false-positive rates). "
    "Discuss these lifestyle factors and laboratory values with your healthcare provider at your next routine check-up."
)

SEE_A_DOCTOR_NOTE_ROUTINE = (
    "Continue regular preventive wellness check-ups and routine health screenings with your primary care provider."
)

# CHAPTER 17: Fixed hand-written lookup table (generic name -> plain-English use)
# Strictly restricted to ONLY these 6 names:
CONFIRMED_MEDICINE_LOOKUP: Dict[str, str] = {
    "metformin": "type 2 diabetes",
    "amlodipine": "blood pressure, and also chest pain",
    "losartan": "blood pressure",
    "atorvastatin": "cholesterol",
    "omeprazole": "stomach acid",
    "paracetamol": "pain and fever",
}

# Mapping of medicines to health conditions for consistency checking (Note b)
MEDICINE_CONDITION_MAP: Dict[str, Dict[str, Any]] = {
    "metformin": {
        "condition_name": "type 2 diabetes",
        "feature_keys": ["Diabetes_binary", "Diabetes"],
    },
    "amlodipine": {
        "condition_name": "blood pressure",
        "feature_keys": ["HighBP"],
    },
    "losartan": {
        "condition_name": "blood pressure",
        "feature_keys": ["HighBP"],
    },
    "atorvastatin": {
        "condition_name": "cholesterol",
        "feature_keys": ["HighChol"],
    },
}

FIXED_MED_DISCLAIMER = (
    "Extracted from your document - check it against the original and confirm with your doctor or pharmacist. "
    "Do not start, stop or change any medicine because of this page."
)


def _bp_condition(ctx: Dict[str, Any]) -> bool:
    """True if HighBP is provided/saved, BP lab exists in doc/memory, or SHAP/diagnoses."""
    feats = ctx.get("features", {})
    if feats.get("HighBP") == 1.0 or feats.get("HighBP") == 1:
        return True
    # Document labs
    for l in ctx.get("document_labs", []):
        if any(k in str(l.get("test", "")).lower() for k in ["blood pressure", "bp"]):
            return True
    labs = ctx.get("latest_labs", {})
    bp_val = labs.get("Blood Pressure", {}).get("value")
    if bp_val:
        return True
    if any("HighBP" in str(f) for f in ctx.get("top_shap_features", [])):
        return True
    if any(h in str(d).lower() for d in ctx.get("diagnoses", []) for h in ["hypertens", "high blood pressure"]):
        return True
    return False


def _chol_condition(ctx: Dict[str, Any]) -> bool:
    """True if HighChol is provided/saved, cholesterol lab exists in doc/memory, or SHAP/diagnoses."""
    feats = ctx.get("features", {})
    if feats.get("HighChol") == 1.0 or feats.get("HighChol") == 1:
        return True
    # Document labs
    for l in ctx.get("document_labs", []):
        if any(k in str(l.get("test", "")).lower() for k in ["cholesterol", "lipid"]):
            return True
    labs = ctx.get("latest_labs", {})
    tc_val = labs.get("Total Cholesterol", {}).get("value")
    if tc_val is not None:
        try:
            if float(tc_val) >= 200.0:
                return True
        except Exception:
            pass
    if any("HighChol" in str(f) for f in ctx.get("top_shap_features", [])):
        return True
    if any(c in str(d).lower() for d in ctx.get("diagnoses", []) for c in ["hypercholester", "hyperlipid", "high cholesterol"]):
        return True
    return False


def _glycemic_condition(ctx: Dict[str, Any]) -> bool:
    """
    True ONLY when a lab value (HbA1c >= 5.7 or fasting glucose >= 100) or a diabetes answer exists.
    Otherwise absent.
    """
    # 1. Document labs (CURRENT upload)
    for l in ctx.get("document_labs", []):
        tname = str(l.get("test", "")).lower()
        val = l.get("value")
        if any(k in tname for k in ["hba1c", "a1c"]) and val is not None:
            try:
                if float(str(val).rstrip("%")) >= 5.7:
                    return True
            except Exception:
                pass
        if any(k in tname for k in ["glucose", "sugar", "fbg"]) and val is not None:
            try:
                if float(val) >= 100.0:
                    return True
            except Exception:
                pass

    # 2. Saved labs (memory)
    labs = ctx.get("latest_labs", {})
    hba1c = labs.get("HbA1c", {}).get("value")
    if hba1c is not None:
        try:
            if float(hba1c) >= 5.7:
                return True
        except Exception:
            pass
    fbg = labs.get("Fasting Blood Glucose", {}).get("value")
    if fbg is not None:
        try:
            if float(fbg) >= 100.0:
                return True
        except Exception:
            pass

    # 3. Diabetes answer / diagnosis
    feats = ctx.get("features", {})
    if feats.get("Diabetes_binary") == 1.0 or feats.get("Diabetes") in (1.0, 2.0, 1, 2):
        return True
    if any("diabet" in str(d).lower() for d in ctx.get("diagnoses", [])):
        return True
    return False


# Rule definitions
RULES = [
    {
        "rule_id": "REC_RISK_HIGH",
        "category": "risk_overall",
        "condition": lambda ctx: ctx.get("risk_level") == "High",
        "guidance_text": "Elevated risk profile detected. Prioritize scheduling an in-person clinical assessment and comprehensive metabolic panel with your primary care physician.",
        "reason": "Multivariate BRFSS model classified overall metabolic risk profile as High.",
        "source_note": "CDC Division of Diabetes Translation - Screening Guidelines (2024)",
        "based_on_fn": lambda ctx: "your risk assessment: High",
    },
    {
        "rule_id": "REC_RISK_MEDIUM",
        "category": "risk_overall",
        "condition": lambda ctx: ctx.get("risk_level") == "Medium",
        "guidance_text": "Moderate risk screening indicator observed. Note: The moderate classification is a sensitive screening flag with moderate precision (~3% in population BRFSS calibration). Discuss preventive lifestyle strategies with your doctor during your next scheduled visit.",
        "reason": "Multivariate BRFSS model flagged moderate risk category.",
        "source_note": "BRFSS Public Health Screening Framework & MediTwin calibration note",
        "based_on_fn": lambda ctx: "your risk assessment: Medium",
    },
    {
        "rule_id": "REC_BP_HIGH",
        "category": "blood_pressure",
        "condition": _bp_condition,
        "guidance_text": "Blood pressure management: Adopt sodium reduction, DASH dietary patterns (rich in vegetables, fruits, and low-fat dairy), and regular blood pressure monitoring under medical supervision.",
        "reason": "High blood pressure indicated in survey profile, saved record, or clinical vital signs.",
        "source_note": "AHA/ACC Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure",
        "based_on_fn": lambda ctx: (
            "saved record: high blood pressure = yes"
            if ctx.get("sources", {}).get("HighBP") == "saved_record"
            else "your answer: high blood pressure = yes"
        ),
    },
    {
        "rule_id": "REC_CHOLESTEROL",
        "category": "cholesterol",
        "condition": _chol_condition,
        "guidance_text": "Cardiovascular wellness: Consider increasing soluble fiber intake (oats, legumes), limiting saturated fats, and reviewing lipid profiles periodically with a healthcare professional.",
        "reason": "Elevated cholesterol indicated in profile, laboratory record, or clinical history.",
        "source_note": "AHA/ACC Multi-Society Guideline on the Management of Blood Cholesterol",
        "based_on_fn": lambda ctx: (
            f"saved lab record: Total Cholesterol {ctx.get('latest_labs', {}).get('Total Cholesterol', {}).get('value')} mg/dL"
            + (f" ({ctx.get('latest_labs', {}).get('Total Cholesterol', {}).get('timestamp')})" if ctx.get('latest_labs', {}).get('Total Cholesterol', {}).get('timestamp') else "")
            if float(ctx.get("latest_labs", {}).get("Total Cholesterol", {}).get("value", 0) or 0) >= 200
            else (
                "saved record: high cholesterol = yes"
                if ctx.get("sources", {}).get("HighChol") == "saved_record"
                else "your answer: high cholesterol = yes"
            )
        ),
    },
    {
        "rule_id": "REC_PHYSICAL_ACTIVITY",
        "category": "lifestyle",
        "condition": lambda ctx: (
            ctx.get("features", {}).get("PhysActivity") == 0
            or ctx.get("features", {}).get("PhysActivity") == 0.0
        ),
        "guidance_text": "Physical activity: Aim for at least 150 minutes of moderate-intensity aerobic exercise (such as brisk walking) per week, plus muscle-strengthening activities on 2 or more days, as tolerated.",
        "reason": "Low physical activity identified in profile.",
        "source_note": "U.S. Department of Health and Human Services - Physical Activity Guidelines for Americans (2nd ed.)",
        "based_on_fn": lambda ctx: (
            "saved record: physical activity in past 30 days = no"
            if ctx.get("sources", {}).get("PhysActivity") == "saved_record"
            else "your answer: physical activity in past 30 days = no"
        ),
    },
    {
        "rule_id": "REC_BMI_NUTRITION",
        "category": "lifestyle",
        "condition": lambda ctx: float(ctx.get("features", {}).get("BMI", 0) or 0) >= 25.0,
        "guidance_text": "Weight and nutrition: Emphasize balanced portion sizes, whole grains, lean proteins, and plenty of vegetables and fruits. Consult a registered dietitian for tailored nutritional planning.",
        "reason": "BMI indicator (>= 25) highlighted in profile.",
        "source_note": "WHO Guidelines on Physical Activity and Sedentary Behaviour / CDC Healthy Weight Guidelines",
        "based_on_fn": lambda ctx: f"your answer: BMI = {ctx.get('features', {}).get('BMI')}",
    },
    {
        "rule_id": "REC_SMOKING",
        "category": "lifestyle",
        "condition": lambda ctx: (
            ctx.get("features", {}).get("Smoker") == 1
            or ctx.get("features", {}).get("Smoker") == 1.0
        ),
        "guidance_text": "Tobacco cessation: Inquire with your healthcare provider about evidence-based smoking cessation resources, counseling, and cessation support programs.",
        "reason": "Smoking history reported in profile.",
        "source_note": "U.S. Preventive Services Task Force (USPSTF) Tobacco Smoking Cessation in Adults Recommendation",
        "based_on_fn": lambda ctx: "your answer: smoker = yes",
    },
    {
        "rule_id": "REC_GLYCEMIC_LAB",
        "category": "laboratory",
        "condition": _glycemic_condition,
        "guidance_text": "Glycemic monitoring: Prioritize discussing recent elevated glycemic indicators with your physician for formal assessment and glycemic health management.",
        "reason": "Laboratory records show elevated glycemic indicators or diabetes reported in profile.",
        "source_note": "American Diabetes Association (ADA) Standards of Care in Diabetes (2024)",
        "based_on_fn": lambda ctx: "your answer: diabetes = yes",
    },
    {
        "rule_id": "REC_IMAGING_ABNORMAL",
        "category": "imaging",
        "condition": lambda ctx: (
            ctx.get("imaging_finding", {}).get("finding") == "PNEUMONIA"
            or float(ctx.get("imaging_finding", {}).get("pneumonia_probability", 0) or 0) >= 0.5
        ),
        "guidance_text": "Chest imaging screening: Automated chest screening model flagged features suggestive of pulmonary opacity/pneumonia. Urgent clinical radiological review and clinical correlation are strongly advised.",
        "reason": "Pretrained chest X-ray screening model flagged potential pneumonia findings.",
        "source_note": "American College of Radiology (ACR) Appropriateness Criteria - Acute Respiratory Illness",
        "based_on_fn": lambda ctx: "uploaded chest X-ray: pulmonary opacity / pneumonia screening indicator",
    },
    {
        "rule_id": "REC_ROUTINE_CHECKUP",
        "category": "preventive",
        "condition": lambda ctx: ctx.get("risk_level") == "Low" and not (
            ctx.get("imaging_finding", {}).get("finding") == "PNEUMONIA"
            or float(ctx.get("imaging_finding", {}).get("pneumonia_probability", 0) or 0) >= 0.5
        ),
        "guidance_text": "Preventive maintenance: Maintain balanced nutrition, adequate hydration, regular restorative sleep, and schedule annual routine wellness visits.",
        "reason": "Profile demonstrates baseline low risk across primary indicators.",
        "source_note": "CDC Preventive Health Care Guidelines for Adults",
        "based_on_fn": lambda ctx: "general advice (no personal data used)",
    },
]
