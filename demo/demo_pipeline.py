"""
demo/demo_pipeline.py
MediTwin AI — Multi-Agent Coordination Pipeline End-to-End Demo (Chapter 8)

Executes a full multi-modal pipeline run:
- Patient Profile (21 BRFSS survey risk features)
- Prescription text (OCR parsing & adapter)
- Lab text (OCR parsing & adapter)
- Clinical notes (Biomedical NER NLP agent)
- Chest X-Ray image (Medical Image ResNet18 screening)
- Patient Memory Agent (Timeline & event aggregation)
- Risk Prediction & XAI Agent (XGBoost + SHAP & LIME local explainability)
- Recommendation Agent (Rules-based educational guidance & safety audit)
"""

import os
import sys
import glob
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.coordination.orchestrator import run_case, BRFSS_FEATURES

DEMO_MEM_DIR = os.path.join(PROJECT_ROOT, "data", "demo_pipeline_scratch")
TEST_XRAY_DIR = os.path.expanduser(
    "~/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/test"
)


def main():
    print("=" * 70)
    print("  MediTwin AI — Chapter 8 Multi-Agent Pipeline End-to-End Demo")
    print("  Hardware: CPU Only | Batch Size 1 | Sequential INFERENCE_LOCK")
    print("=" * 70)

    # Pick an existing test X-ray image
    xrays = glob.glob(os.path.join(TEST_XRAY_DIR, "NORMAL", "*.jpeg"))
    if not xrays:
        xrays = glob.glob(os.path.join(TEST_XRAY_DIR, "**", "*.jpeg"), recursive=True)
    sample_xray = xrays[0] if xrays else None

    # Synthetic multi-modal case
    demo_case = {
        "patient_id": "DEMO_PATIENT_CH8",
        "patient_profile": {
            "HighBP": 1.0,
            "HighChol": 1.0,
            "CholCheck": 1.0,
            "BMI": 32.4,
            "Smoker": 0.0,
            "Stroke": 0.0,
            "HeartDiseaseorAttack": 0.0,
            "PhysActivity": 1.0,
            "Fruits": 1.0,
            "Veggies": 1.0,
            "HvyAlcoholConsump": 0.0,
            "AnyHealthcare": 1.0,
            "NoDocbcCost": 0.0,
            "GenHlth": 3.0,
            "MentHlth": 2.0,
            "PhysHlth": 4.0,
            "DiffWalk": 0.0,
            "Sex": 1.0,
            "Age": 9.0,
            "Education": 5.0,
            "Income": 7.0,
        },
        "prescription_text": "Rx: Metformin 500 mg PO BID with meals. Atorvastatin 20 mg PO QHS.",
        "lab_text": "HbA1c: 7.2 % (Ref: 4.0 - 5.6). Fasting Blood Glucose: 142 mg/dL.",
        "clinical_notes": "Patient presents with progressive fatigue and occasional bilateral lower extremity edema.",
        "chest_xray_path": sample_xray,
        "timestamp": "2026-11-01T10:00:00Z",
    }

    t0 = time.time()
    result = run_case(demo_case, memory_storage_dir=DEMO_MEM_DIR)
    elapsed = time.time() - t0

    print("\n--- Pipeline Execution Status ---")
    for agent, status in result["agent_status"].items():
        timing = result["agent_timing_seconds"].get(agent, 0.0)
        print(f"  • {agent:<24}: {status} ({timing:.2f}s)")
    print(f"\nTotal Pipeline Time: {result['total_time_seconds']:.2f}s (measured: {elapsed:.2f}s)")
    print(f"Peak RSS Memory:     {result['peak_rss_mb']:.1f} MB")

    print("\n--- Case Clinical Summary Output ---")
    print(result["summary"])

    # Clean up scratch storage
    if os.path.exists(DEMO_MEM_DIR):
        import shutil
        shutil.rmtree(DEMO_MEM_DIR)

    # Verification: must have completed without error
    if (
        result["agent_status"].get("risk_agent") == "ok"
        and result["agent_status"].get("xai_agent") == "ok"
        and result["agent_status"].get("recommendation_agent") == "ok"
        and result["agent_status"].get("patient_memory_agent") == "ok"
        and result.get("summary")
    ):
        print("\n[demo_pipeline PASS]")
        return 0
    else:
        print("\n[demo_pipeline FAIL] Pipeline did not produce expected complete outputs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

