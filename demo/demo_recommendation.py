"""
demo/demo_recommendation.py
Recommendation Agent Demo for MediTwin AI.
Demonstrates structured educational guidance generation, rule triggering,
and strict safety verification across High, Medium, and Low risk scenarios.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.recommendation_agent.recommendation_agent import RecommendationAgent, check_safety


def run_demo():
    print("=" * 60)
    print("  MediTwin AI — Recommendation Agent Demo")
    print("=" * 60)

    agent = RecommendationAgent()

    # Scenario: High Risk + High BP + Pneumonia screening finding
    print("\n  Scenario: High Risk with HighBP and Elevated Imaging Opacity")
    out = agent.generate_recommendations(
        risk_level="High",
        probabilities={"Low": 0.08, "Medium": 0.15, "High": 0.77},
        top_shap_features=["HighBP pushes toward High", "BMI pushes toward High"],
        patient_profile={"HighBP": 1, "BMI": 29.2, "PhysActivity": 0, "Smoker": 0},
        memory_summary={"latest_labs": {"HbA1c": {"value": 6.7, "unit": "%"}}},
        imaging_result={"predicted_class": "PNEUMONIA", "pneumonia_probability": 0.91},
    )

    print(f"  Risk Level: {out['risk_level']}")
    print(f"  Guidance items triggered: {len(out['guidance'])}")
    for g in out["guidance"]:
        print(f"   [{g['rule_id']}] ({g['category']}) {g['text']}")
    print(f"\n  Doctor Consultation Note:\n    {out['see_a_doctor_note']}")
    print(f"\n  Disclaimer:\n    {out['disclaimer']}")

    # Verification: check safety and mandatory elements
    full_text = " ".join([g["text"] for g in out["guidance"]] + [out["see_a_doctor_note"], out["disclaimer"]])
    violations = check_safety(full_text)

    if violations:
        print(f"\n  [demo_recommendation FAIL]: Safety violations detected: {violations}")
        return 1

    if len(out["guidance"]) >= 3 and out["disclaimer"] and "healthcare professional" in out["see_a_doctor_note"]:
        print("\n  Demo complete. [demo_recommendation PASS]")
        return 0
    else:
        print("\n  [demo_recommendation FAIL]: Missing required guidance elements.")
        return 1


if __name__ == "__main__":
    sys.exit(run_demo())

