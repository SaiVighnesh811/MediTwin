# Recommendation Rules Review — Clinical & Expert Review Guide

> **NOTICE**: These rules are **NOT clinically validated**.
> They represent an educational baseline derived from general public-health publications (CDC, AHA, WHO, ADA).
> They must undergo formal clinical evaluation and review by certified medical professionals prior to any clinical deployment.

---

## Safety Constraints Enforced by System
1. **Never a Diagnosis**: Statements describing conditions must frame them as automated screening findings or self-reported factors, not confirmed medical diagnoses.
2. **Never Prescription or Dosage Advice**: The system is strictly prohibited from mentioning drug dosages, suggesting start/stop decisions, or prescribing substances.
3. **Always Includes Disclaimer**: The mandatory educational disclaimer is appended to every single generated payload.
4. **Mandatory Physician Consultation**: Any `High` risk classification or abnormal screening finding automatically triggers an unequivocal recommendation to consult a qualified healthcare provider.
5. **Screening Calibration Transparency**: `Medium` risk notices clearly inform the user that moderate classification serves as a sensitive screening flag with moderate precision (~3% in BRFSS survey data).

---

## Complete Rule Base Table

| Rule ID | Category | Trigger Condition | Educational Guidance Provided | Clinical / Public Health Source Note |
|:---|:---|:---|:---|:---|
| `REC_RISK_HIGH` | Overall Risk | `risk_level == "High"` | Elevated risk profile detected. Prioritize scheduling an in-person clinical assessment and comprehensive metabolic panel with your primary care physician. | CDC Division of Diabetes Translation - Screening Guidelines (2024) |
| `REC_RISK_MEDIUM` | Overall Risk | `risk_level == "Medium"` | Moderate risk screening indicator observed. Note: The moderate classification is a sensitive screening flag with moderate precision (~3% in population BRFSS calibration). Discuss preventive lifestyle strategies with your doctor during your next scheduled visit. | BRFSS Public Health Screening Framework & MediTwin calibration note |
| `REC_BP_HIGH` | Cardiovascular | Profile `HighBP == 1` OR SHAP feature includes `HighBP` | Blood pressure management: Adopt sodium reduction, DASH dietary patterns (rich in vegetables, fruits, and low-fat dairy), and regular blood pressure monitoring under medical supervision. | AHA/ACC Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure |
| `REC_CHOLESTEROL` | Cardiovascular | Profile `HighChol == 1` OR SHAP `HighChol` OR Lab `Total Cholesterol >= 200` | Cardiovascular wellness: Consider increasing soluble fiber intake (oats, legumes), limiting saturated fats, and reviewing lipid profiles periodically with a healthcare professional. | AHA/ACC Multi-Society Guideline on the Management of Blood Cholesterol |
| `REC_PHYSICAL_ACTIVITY` | Lifestyle | Profile `PhysActivity == 0` OR SHAP `PhysActivity` | Physical activity: Aim for at least 150 minutes of moderate-intensity aerobic exercise (such as brisk walking) per week, plus muscle-strengthening activities on 2 or more days, as tolerated. | U.S. Dept. of Health and Human Services - Physical Activity Guidelines for Americans (2nd ed.) |
| `REC_BMI_NUTRITION` | Lifestyle | Profile `BMI >= 25` OR SHAP `BMI` | Weight and nutrition: Emphasize balanced portion sizes, whole grains, lean proteins, and plenty of vegetables and fruits. Consult a registered dietitian for tailored nutritional planning. | WHO Guidelines on Physical Activity and Sedentary Behaviour / CDC Healthy Weight Guidelines |
| `REC_SMOKING` | Lifestyle | Profile `Smoker == 1` | Tobacco cessation: Inquire with your healthcare provider about evidence-based smoking cessation resources, counseling, and cessation support programs. | USPSTF Tobacco Smoking Cessation in Adults Recommendation |
| `REC_GLYCEMIC_LAB` | Laboratory | Lab `HbA1c >= 5.7%` OR `Fasting Glucose >= 100 mg/dL` | Glycemic monitoring: Prioritize discussing recent elevated glycemic indicators (HbA1c or fasting glucose) with your physician for formal assessment and glycemic health management. | American Diabetes Association (ADA) Standards of Care in Diabetes (2024) |
| `REC_IMAGING_ABNORMAL` | Imaging | Model finding `PNEUMONIA` OR probability `>= 0.5` | Chest imaging screening: Automated chest screening model flagged features suggestive of pulmonary opacity/pneumonia. Urgent clinical radiological review and clinical correlation are strongly advised. | American College of Radiology (ACR) Appropriateness Criteria - Acute Respiratory Illness |
| `REC_ROUTINE_CHECKUP` | Preventive | `risk_level == "Low"` AND normal imaging | Preventive maintenance: Maintain balanced nutrition, adequate hydration, regular restorative sleep, and schedule annual routine wellness visits. | CDC Preventive Health Care Guidelines for Adults |

---
*Document prepared for clinical advisory board and expert review.*

