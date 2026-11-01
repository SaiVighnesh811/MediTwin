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

# Rule definitions
RULES = [
    {
        "rule_id": "REC_RISK_HIGH",
        "category": "risk_overall",
        "condition": lambda ctx: ctx.get("risk_level") == "High",
        "guidance_text": "Elevated risk profile detected. Prioritize scheduling an in-person clinical assessment and comprehensive metabolic panel with your primary care physician.",
        "reason": "Multivariate BRFSS model classified overall metabolic risk profile as High.",
        "source_note": "CDC Division of Diabetes Translation - Screening Guidelines (2024)",
    },
    {
        "rule_id": "REC_RISK_MEDIUM",
        "category": "risk_overall",
        "condition": lambda ctx: ctx.get("risk_level") == "Medium",
        "guidance_text": "Moderate risk screening indicator observed. Note: The moderate classification is a sensitive screening flag with moderate precision (~3% in population BRFSS calibration). Discuss preventive lifestyle strategies with your doctor during your next scheduled visit.",
        "reason": "Multivariate BRFSS model flagged moderate risk category.",
        "source_note": "BRFSS Public Health Screening Framework & MediTwin calibration note",
    },
    {
        "rule_id": "REC_BP_HIGH",
        "category": "blood_pressure",
        "condition": lambda ctx: (
            ctx.get("features", {}).get("HighBP") == 1
            or any("HighBP" in str(f) for f in ctx.get("top_shap_features", []))
        ),
        "guidance_text": "Blood pressure management: Adopt sodium reduction, DASH dietary patterns (rich in vegetables, fruits, and low-fat dairy), and regular blood pressure monitoring under medical supervision.",
        "reason": "High blood pressure indicated in survey profile or identified as a key contributing factor by XAI SHAP analysis.",
        "source_note": "AHA/ACC Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure",
    },
    {
        "rule_id": "REC_CHOLESTEROL",
        "category": "cholesterol",
        "condition": lambda ctx: (
            ctx.get("features", {}).get("HighChol") == 1
            or any("HighChol" in str(f) for f in ctx.get("top_shap_features", []))
            or float(ctx.get("latest_labs", {}).get("Total Cholesterol", {}).get("value", 0) or 0) >= 200
        ),
        "guidance_text": "Cardiovascular wellness: Consider increasing soluble fiber intake (oats, legumes), limiting saturated fats, and reviewing lipid profiles periodically with a healthcare professional.",
        "reason": "Elevated cholesterol indicated in profile, laboratory record, or flagged by XAI SHAP analysis.",
        "source_note": "AHA/ACC Multi-Society Guideline on the Management of Blood Cholesterol",
    },
    {
        "rule_id": "REC_PHYSICAL_ACTIVITY",
        "category": "lifestyle",
        "condition": lambda ctx: (
            ctx.get("features", {}).get("PhysActivity") == 0
            or any("PhysActivity" in str(f) for f in ctx.get("top_shap_features", []))
        ),
        "guidance_text": "Physical activity: Aim for at least 150 minutes of moderate-intensity aerobic exercise (such as brisk walking) per week, plus muscle-strengthening activities on 2 or more days, as tolerated.",
        "reason": "Low physical activity identified in profile or XAI explanation.",
        "source_note": "U.S. Department of Health and Human Services - Physical Activity Guidelines for Americans (2nd ed.)",
    },
    {
        "rule_id": "REC_BMI_NUTRITION",
        "category": "lifestyle",
        "condition": lambda ctx: (
            float(ctx.get("features", {}).get("BMI", 0) or 0) >= 25.0
            or any("BMI" in str(f) for f in ctx.get("top_shap_features", []))
        ),
        "guidance_text": "Weight and nutrition: Emphasize balanced portion sizes, whole grains, lean proteins, and plenty of vegetables and fruits. Consult a registered dietitian for tailored nutritional planning.",
        "reason": "BMI indicator (>= 25) or high BMI importance highlighted in XAI feature impact.",
        "source_note": "WHO Guidelines on Physical Activity and Sedentary Behaviour / CDC Healthy Weight Guidelines",
    },
    {
        "rule_id": "REC_SMOKING",
        "category": "lifestyle",
        "condition": lambda ctx: ctx.get("features", {}).get("Smoker") == 1,
        "guidance_text": "Tobacco cessation: Inquire with your healthcare provider about evidence-based smoking cessation resources, counseling, and cessation support programs.",
        "reason": "Smoking history reported in profile.",
        "source_note": "U.S. Preventive Services Task Force (USPSTF) Tobacco Smoking Cessation in Adults Recommendation",
    },
    {
        "rule_id": "REC_GLYCEMIC_LAB",
        "category": "laboratory",
        "condition": lambda ctx: (
            float(ctx.get("latest_labs", {}).get("HbA1c", {}).get("value", 0) or 0) >= 5.7
            or float(ctx.get("latest_labs", {}).get("Fasting Blood Glucose", {}).get("value", 0) or 0) >= 100
        ),
        "guidance_text": "Glycemic monitoring: Prioritize discussing recent elevated glycemic indicators (HbA1c or fasting glucose) with your physician for formal assessment and glycemic health management.",
        "reason": "Laboratory records show HbA1c >= 5.7% or Fasting Glucose >= 100 mg/dL.",
        "source_note": "American Diabetes Association (ADA) Standards of Care in Diabetes (2024)",
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
    },
    {
        "rule_id": "REC_ROUTINE_CHECKUP",
        "category": "preventive",
        "condition": lambda ctx: ctx.get("risk_level") == "Low" and not ctx.get("imaging_finding", {}).get("finding") == "PNEUMONIA",
        "guidance_text": "Preventive maintenance: Maintain balanced nutrition, adequate hydration, regular restorative sleep, and schedule annual routine wellness visits.",
        "reason": "Profile demonstrates baseline low risk across primary indicators.",
        "source_note": "CDC Preventive Health Care Guidelines for Adults",
    },
]
