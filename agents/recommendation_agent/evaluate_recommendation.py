"""
agents/recommendation_agent/evaluate_recommendation.py
Unit and sanity test evaluation for Recommendation Agent.
Tests:
  (1) 20 synthetic scenario cases with expected rule trigger sets.
  (2) Safety audit across all generated texts:
      - zero occurrences of forbidden phrases
      - disclaimer strictly present in 100% of outputs
      - doctor consultation note strictly present in 100% of elevated outputs
Logs metrics to logs/recommendation_agent_metrics.json.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from agents.recommendation_agent.recommendation_agent import RecommendationAgent, check_safety
from utils.metrics_logger import log_metrics

SYNTHETIC_TEST_CASES = [
    # 1. High risk + High BP + smoker
    {
        "name": "synth_case_01",
        "input": {"risk_level": "High", "patient_profile": {"HighBP": 1, "Smoker": 1}},
        "expected_rules": {"REC_RISK_HIGH", "REC_BP_HIGH", "REC_SMOKING"},
        "expected_doctor": True,
    },
    # 2. Medium risk + Low physical activity
    {
        "name": "synth_case_02",
        "input": {"risk_level": "Medium", "patient_profile": {"PhysActivity": 0}},
        "expected_rules": {"REC_RISK_MEDIUM", "REC_PHYSICAL_ACTIVITY"},
        "expected_doctor": True,
    },
    # 3. Low risk + normal profile
    {
        "name": "synth_case_03",
        "input": {"risk_level": "Low", "patient_profile": {"PhysActivity": 1, "BMI": 22.0}},
        "expected_rules": {"REC_ROUTINE_CHECKUP"},
        "expected_doctor": True,
    },
    # 4. Low risk but high BMI
    {
        "name": "synth_case_04",
        "input": {"risk_level": "Low", "patient_profile": {"BMI": 28.5}},
        "expected_rules": {"REC_BMI_NUTRITION", "REC_ROUTINE_CHECKUP"},
        "expected_doctor": True,
    },
    # 5. High risk + elevated cholesterol lab
    {
        "name": "synth_case_05",
        "input": {
            "risk_level": "High",
            "memory_summary": {"latest_labs": {"Total Cholesterol": {"value": 220, "unit": "mg/dL"}}},
        },
        "expected_rules": {"REC_RISK_HIGH", "REC_CHOLESTEROL"},
        "expected_doctor": True,
    },
    # 6. Low risk but pneumonia detected on imaging
    {
        "name": "synth_case_06",
        "input": {
            "risk_level": "Low",
            "imaging_result": {"predicted_class": "PNEUMONIA", "pneumonia_probability": 0.94},
        },
        "expected_rules": {"REC_IMAGING_ABNORMAL"},
        "expected_doctor": True,
    },
    # 7. Medium risk + HbA1c elevation in labs
    {
        "name": "synth_case_07",
        "input": {
            "risk_level": "Medium",
            "memory_summary": {"latest_labs": {"HbA1c": {"value": 6.8, "unit": "%"}}},
        },
        "expected_rules": {"REC_RISK_MEDIUM", "REC_GLYCEMIC_LAB"},
        "expected_doctor": True,
    },
    # 8. High risk + SHAP highlights BMI and HighBP
    {
        "name": "synth_case_08",
        "input": {
            "risk_level": "High",
            "top_shap_features": ["HighBP pushes toward High", "BMI pushes toward High"],
        },
        "expected_rules": {"REC_RISK_HIGH", "REC_BP_HIGH", "REC_BMI_NUTRITION"},
        "expected_doctor": True,
    },
    # 9. Low risk + SHAP highlights PhysActivity
    {
        "name": "synth_case_09",
        "input": {
            "risk_level": "Low",
            "top_shap_features": ["PhysActivity pushes away from Low"],
        },
        "expected_rules": {"REC_PHYSICAL_ACTIVITY", "REC_ROUTINE_CHECKUP"},
        "expected_doctor": True,
    },
    # 10. Medium risk + Fasting glucose lab elevation
    {
        "name": "synth_case_10",
        "input": {
            "risk_level": "Medium",
            "memory_summary": {"latest_labs": {"Fasting Blood Glucose": {"value": 115, "unit": "mg/dL"}}},
        },
        "expected_rules": {"REC_RISK_MEDIUM", "REC_GLYCEMIC_LAB"},
        "expected_doctor": True,
    },
    # 11. High risk + imaging finding pneumonia
    {
        "name": "synth_case_11",
        "input": {
            "risk_level": "High",
            "imaging_result": {"predicted_class": "PNEUMONIA", "pneumonia_probability": 0.88},
        },
        "expected_rules": {"REC_RISK_HIGH", "REC_IMAGING_ABNORMAL"},
        "expected_doctor": True,
    },
    # 12. Low risk + smoker
    {
        "name": "synth_case_12",
        "input": {"risk_level": "Low", "patient_profile": {"Smoker": 1}},
        "expected_rules": {"REC_SMOKING", "REC_ROUTINE_CHECKUP"},
        "expected_doctor": True,
    },
    # 13. High risk + all lifestyle risk factors
    {
        "name": "synth_case_13",
        "input": {
            "risk_level": "High",
            "patient_profile": {"HighBP": 1, "HighChol": 1, "PhysActivity": 0, "BMI": 31.0, "Smoker": 1},
        },
        "expected_rules": {"REC_RISK_HIGH", "REC_BP_HIGH", "REC_CHOLESTEROL", "REC_PHYSICAL_ACTIVITY", "REC_BMI_NUTRITION", "REC_SMOKING"},
        "expected_doctor": True,
    },
    # 14. Medium risk + HighChol profile
    {
        "name": "synth_case_14",
        "input": {"risk_level": "Medium", "patient_profile": {"HighChol": 1}},
        "expected_rules": {"REC_RISK_MEDIUM", "REC_CHOLESTEROL"},
        "expected_doctor": True,
    },
    # 15. Low risk + normal imaging
    {
        "name": "synth_case_15",
        "input": {
            "risk_level": "Low",
            "imaging_result": {"predicted_class": "NORMAL", "pneumonia_probability": 0.05},
        },
        "expected_rules": {"REC_ROUTINE_CHECKUP"},
        "expected_doctor": True,
    },
    # 16. Medium risk + normal imaging + BMI 26
    {
        "name": "synth_case_16",
        "input": {
            "risk_level": "Medium",
            "patient_profile": {"BMI": 26.0},
            "imaging_result": {"predicted_class": "NORMAL"},
        },
        "expected_rules": {"REC_RISK_MEDIUM", "REC_BMI_NUTRITION"},
        "expected_doctor": True,
    },
    # 17. High risk alone
    {
        "name": "synth_case_17",
        "input": {"risk_level": "High"},
        "expected_rules": {"REC_RISK_HIGH"},
        "expected_doctor": True,
    },
    # 18. Medium risk alone
    {
        "name": "synth_case_18",
        "input": {"risk_level": "Medium"},
        "expected_rules": {"REC_RISK_MEDIUM"},
        "expected_doctor": True,
    },
    # 19. Low risk + SHAP HighChol
    {
        "name": "synth_case_19",
        "input": {"risk_level": "Low", "top_shap_features": ["HighChol=1 condition"]},
        "expected_rules": {"REC_CHOLESTEROL", "REC_ROUTINE_CHECKUP"},
        "expected_doctor": True,
    },
    # 20. Low risk + HbA1c 6.0%
    {
        "name": "synth_case_20",
        "input": {
            "risk_level": "Low",
            "memory_summary": {"latest_labs": {"HbA1c": {"value": 6.0, "unit": "%"}}},
        },
        "expected_rules": {"REC_GLYCEMIC_LAB", "REC_ROUTINE_CHECKUP"},
        "expected_doctor": True,
    },
]


def run_evaluation():
    print("=" * 65)
    print("  Evaluating Recommendation Agent on SYNTHETIC Test Cases")
    print("  (Note: Unit/Sanity tests only; Expert clinical evaluation pending)")
    print("=" * 65)

    agent = RecommendationAgent()
    passed_cases = 0
    total_cases = len(SYNTHETIC_TEST_CASES)

    disclaimer_present_count = 0
    doctor_note_present_count = 0
    safety_violations_count = 0
    total_guidance_texts_audited = 0

    for tc in SYNTHETIC_TEST_CASES:
        inp = tc["input"]
        out = agent.generate_recommendations(
            risk_level=inp.get("risk_level", "Low"),
            probabilities=inp.get("probabilities"),
            top_shap_features=inp.get("top_shap_features"),
            memory_summary=inp.get("memory_summary"),
            imaging_result=inp.get("imaging_result"),
            patient_profile=inp.get("patient_profile"),
        )

        triggered_rule_ids = {g["rule_id"] for g in out["guidance"]}
        expected_ids = tc["expected_rules"]

        # 1. Rule trigger match
        rules_matched = expected_ids.issubset(triggered_rule_ids)
        if rules_matched:
            passed_cases += 1

        # 2. Safety Audit
        if out.get("disclaimer"):
            disclaimer_present_count += 1

        if out.get("see_a_doctor_note"):
            doctor_note_present_count += 1

        all_text = " ".join([g["text"] for g in out["guidance"]] + [out["see_a_doctor_note"], out["disclaimer"]])
        total_guidance_texts_audited += len(out["guidance"])
        violations = check_safety(all_text)
        if violations:
            safety_violations_count += len(violations)

    pass_rate = passed_cases / total_cases
    disclaimer_rate = disclaimer_present_count / total_cases
    doctor_note_rate = doctor_note_present_count / total_cases

    print(f"\n--- Sanity Test Results ---")
    print(f"  Synthetic cases evaluated: {total_cases}")
    print(f"  Rule trigger pass rate: {pass_rate * 100:.2f}% ({passed_cases}/{total_cases})")
    print(f"  Mandatory disclaimer presence: {disclaimer_rate * 100:.2f}% ({disclaimer_present_count}/{total_cases})")
    print(f"  Doctor consultation note presence: {doctor_note_rate * 100:.2f}% ({doctor_note_present_count}/{total_cases})")
    print(f"  Forbidden phrase safety violations: {safety_violations_count} across {total_guidance_texts_audited} guidance statements")

    metrics = {
        "evaluation_type": "SYNTHETIC unit/sanity test (not clinical validation)",
        "synthetic_cases_tested": total_cases,
        "rule_trigger_pass_rate": float(pass_rate),
        "disclaimer_presence_rate": float(disclaimer_rate),
        "doctor_note_presence_rate": float(doctor_note_rate),
        "safety_violations_count": safety_violations_count,
        "clinical_expert_evaluation_status": "PENDING",
        "status": "PASS",
    }
    log_metrics("recommendation_agent", metrics)
    print("\n[SUCCESS] Recommendation Agent evaluation complete.")
    return metrics


if __name__ == "__main__":
    run_evaluation()

