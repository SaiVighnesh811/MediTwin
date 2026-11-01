"""
demo/demo_memory.py
Patient Memory Agent Demo for MediTwin AI.
Demonstrates storing and querying events, adapting agent outputs,
checking trend retrieval, and generating a patient clinical summary.
"""

import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.patient_memory_agent.patient_memory_agent import (
    PatientMemoryAgent,
    adapt_ocr_result,
    adapt_nlp_result,
    adapt_risk_result,
    adapt_medical_image_result,
)


def run_demo():
    print("=" * 60)
    print("  MediTwin AI — Patient Memory Agent Demo")
    print("=" * 60)

    demo_dir = os.path.join(PROJECT_ROOT, "data", "patient_memory_demo")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    mem = PatientMemoryAgent(storage_dir=demo_dir)
    pid = "DEMO_PATIENT_001"

    # 1. Ingest adapted OCR results
    rx_ocr = {"type": "prescription", "medicine": "Metformin", "dosage": "500 mg", "frequency": "twice daily", "duration": "30 days"}
    lab_ocr = {"type": "lab", "test": "HbA1c", "value": "7.2", "unit": "%", "reference": "4.0-5.6"}
    adapt_ocr_result(pid, rx_ocr, timestamp="2026-01-10T09:00:00Z", memory_agent=mem)
    adapt_ocr_result(pid, lab_ocr, timestamp="2026-01-10T09:15:00Z", memory_agent=mem)

    # 2. Ingest adapted NLP entities
    nlp_out = {
        "diseases": ["type 2 diabetes mellitus"],
        "symptoms": ["fatigue", "frequent urination"],
        "medications": ["Metformin"],
        "procedures": ["fasting blood glucose test"],
    }
    adapt_nlp_result(pid, nlp_out, timestamp="2026-01-10T09:30:00Z", memory_agent=mem)

    # 3. Ingest Risk Assessment
    risk_out = {"predicted_class": "High", "probabilities": {"Low": 0.12, "Medium": 0.18, "High": 0.70}}
    adapt_risk_result(pid, risk_out, timestamp="2026-01-10T10:00:00Z", memory_agent=mem)

    # 4. Ingest Medical Image Finding
    img_out = {
        "predicted_class": "NORMAL",
        "pneumonia_probability": 0.08,
        "normal_probability": 0.92,
        "heatmap_path": "logs/gradcam_demo.png",
        "disclaimer": "SCREENING AID ONLY",
    }
    adapt_medical_image_result(pid, img_out, timestamp="2026-01-10T10:30:00Z", memory_agent=mem)

    # 5. Later follow-up lab
    lab_ocr2 = {"type": "lab", "test": "HbA1c", "value": "6.8", "unit": "%", "reference": "4.0-5.6"}
    adapt_ocr_result(pid, lab_ocr2, timestamp="2026-04-12T11:00:00Z", memory_agent=mem)

    # Queries
    timeline = mem.get_timeline(pid)
    hba1c_trend = mem.get_trend(pid, "HbA1c")
    summary = mem.summary(pid)

    print(f"  Patient ID: {pid}")
    print(f"  Total Timeline Events: {len(timeline)}")
    print(f"  Diagnoses: {summary['diagnoses']}")
    print(f"  Symptoms: {summary['symptoms']}")
    print(f"  Medications: {summary['medications']}")
    print(f"  HbA1c Trend Points: {len(hba1c_trend)} (Values: {[t['value'] for t in hba1c_trend]})")
    print(f"  Latest Risk Assessment: {summary['latest_risk']['risk_level']}")
    print(f"  Latest Imaging Finding: {summary['imaging_findings'][-1]['finding']}")

    # Validation check: PASS only if events really stored and retrieved
    if len(timeline) >= 6 and len(hba1c_trend) == 2 and summary["latest_risk"] is not None:
        print("\n  Demo complete. [demo_memory PASS]")
        # Keep demo dir clean
        shutil.rmtree(demo_dir)
        return 0
    else:
        print("\n  Demo failed: timeline verification failed. [demo_memory FAIL]")
        return 1


if __name__ == "__main__":
    sys.exit(run_demo())

