"""
agents/coordination/orchestrator.py
MediTwin AI — Multi-Agent Coordination Orchestrator (Chapter 8)

Coordinates multi-modal pipeline execution:
1. OCR Agent (prescription & lab report text extraction)
2. NLP Agent (biomedical named entity recognition)
3. Medical Image Agent (chest X-ray pneumonia screening)
4. Patient Memory Agent (longitudinal event store & history aggregation)
5. Risk Prediction Agent (BRFSS 21-feature metabolic risk classification)
6. XAI Agent (SHAP & LIME local explainability)
7. Recommendation Agent (educational health & lifestyle guidance)

CRITICAL DESIGN & ARCHITECTURE RULES:
- Never run two heavy models at once (guarded via INFERENCE_LOCK).
- Lazy model loading: models load once on demand.
- Risk Prediction model strictly uses the 21 BRFSS survey features from patient_profile.
  Extracted lab and medication entities feed into Patient Memory and Recommendation,
  NOT into the Risk model (design separation between epidemiological risk survey & clinical findings).
- Robust partial execution: if an input is missing or an agent fails, the pipeline
  continues with remaining agents, marking individual agent status and partial output.
"""

import os
import sys
import time
import json
import traceback
from typing import Any, Dict, List, Optional
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.inference_lock import INFERENCE_LOCK

BRFSS_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]


def _get_peak_rss_mb() -> float:
    """Reads VmRSS from /proc/self/status on Linux."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def run_case(case: Dict[str, Any], memory_storage_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes the multi-agent pipeline for a single clinical case.
    
    Expected case fields:
      - patient_id: str (default 'P_UNKNOWN')
      - patient_profile: dict with 21 BRFSS features (for Risk & XAI)
      - prescription_text or prescription_image: str
      - lab_text or lab_image: str
      - clinical_notes: str (for NLP)
      - chest_xray_path: str (for Medical Image)
      - timestamp: optional ISO timestamp
    """
    t_start = time.time()
    patient_id = case.get("patient_id", "P_UNKNOWN")
    timestamp = case.get("timestamp")

    agent_status: Dict[str, str] = {}
    agent_timing: Dict[str, float] = {}
    errors: Dict[str, str] = {}

    context: Dict[str, Any] = {
        "case_input": case,
        "ocr": {},
        "nlp": {},
        "medical_image": {},
        "patient_memory": {},
        "risk": {},
        "xai": {},
        "recommendation": {},
    }

    # -------------------------------------------------------------
    # 1. Patient Memory Agent Init
    # -------------------------------------------------------------
    from agents.patient_memory_agent.patient_memory_agent import (
        PatientMemoryAgent,
        adapt_ocr_result,
        adapt_nlp_result,
        adapt_risk_result,
        adapt_medical_image_result,
    )
    mem_agent = PatientMemoryAgent(storage_dir=memory_storage_dir)

    # -------------------------------------------------------------
    # 2. OCR Agent
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        from agents.ocr_agent.ocr_agent import extract_fields, read_printed
        ocr_results = []

        # Handle prescription input
        rx_text = case.get("prescription_text")
        if not rx_text and case.get("prescription_image"):
            rx_text = read_printed(case["prescription_image"])
        if rx_text:
            parsed_rx = extract_fields(rx_text)
            ocr_results.append(parsed_rx)
            adapt_ocr_result(patient_id, parsed_rx, timestamp=timestamp, memory_agent=mem_agent, raw_text=rx_text)

        # Handle lab input
        lab_text = case.get("lab_text")
        if not lab_text and case.get("lab_image"):
            lab_text = read_printed(case["lab_image"])
        if lab_text:
            parsed_lab = extract_fields(lab_text)
            ocr_results.append(parsed_lab)
            adapt_ocr_result(patient_id, parsed_lab, timestamp=timestamp, memory_agent=mem_agent, raw_text=lab_text)

        if ocr_results:
            context["ocr"] = {"extracted": ocr_results}
            agent_status["ocr_agent"] = "ok"
        else:
            agent_status["ocr_agent"] = "skipped (no input)"
    except Exception as e:
        agent_status["ocr_agent"] = "failed"
        errors["ocr_agent"] = str(e)
    agent_timing["ocr_agent"] = round(time.time() - t0, 3)

    # -------------------------------------------------------------
    # 3. NLP Agent
    # -------------------------------------------------------------
    t0 = time.time()
    notes = case.get("clinical_notes")
    if notes:
        try:
            from agents.nlp_agent.nlp_agent import extract_entities
            nlp_res = extract_entities(notes)
            context["nlp"] = nlp_res
            adapt_nlp_result(patient_id, nlp_res, timestamp=timestamp, memory_agent=mem_agent)
            agent_status["nlp_agent"] = "ok"
        except Exception as e:
            agent_status["nlp_agent"] = "failed"
            errors["nlp_agent"] = str(e)
    else:
        agent_status["nlp_agent"] = "skipped (no input)"
    agent_timing["nlp_agent"] = round(time.time() - t0, 3)

    # -------------------------------------------------------------
    # 4. Medical Image Agent
    # -------------------------------------------------------------
    t0 = time.time()
    xray_path = case.get("chest_xray_path")
    if xray_path and os.path.exists(xray_path):
        try:
            from agents.medical_image_agent.medical_image_agent import predict_image
            img_res = predict_image(xray_path)
            context["medical_image"] = img_res
            adapt_medical_image_result(patient_id, img_res, timestamp=timestamp, memory_agent=mem_agent)
            agent_status["medical_image_agent"] = "ok"
        except Exception as e:
            agent_status["medical_image_agent"] = "failed"
            errors["medical_image_agent"] = str(e)
    elif xray_path:
        agent_status["medical_image_agent"] = "failed (file not found)"
        errors["medical_image_agent"] = f"Path does not exist: {xray_path}"
    else:
        agent_status["medical_image_agent"] = "skipped (no input)"
    agent_timing["medical_image_agent"] = round(time.time() - t0, 3)

    # -------------------------------------------------------------
    # 5. Patient Memory Summary Refresh (Before Risk so Assembler has memory)
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        mem_summary = mem_agent.summary(patient_id)
        context["patient_memory"] = mem_summary
        agent_status["patient_memory_agent"] = "ok"
    except Exception as e:
        agent_status["patient_memory_agent"] = "failed"
        errors["patient_memory_agent"] = str(e)
    agent_timing["patient_memory_agent"] = round(time.time() - t0, 3)

    # -------------------------------------------------------------
    # 6. Feature Assembler, Risk Prediction & XAI Agents
    # -------------------------------------------------------------
    t0 = time.time()
    profile = case.get("patient_profile")
    try:
        from agents.coordination.feature_assembler import assemble_features
        from agents.xai_agent.xai_agent import explain_with_shap, explain_with_lime

        # Assemble the 21 features with priority: provided > extracted > assumed
        assembler_res = assemble_features(
            provided_profile=profile if isinstance(profile, dict) else {},
            memory_summary=context.get("patient_memory", {}),
            memory_agent=mem_agent,
            patient_id=patient_id,
            default_strategy="training_median"
        )
        context["feature_assembly"] = assembler_res

        patient_row = {f: float(assembler_res["features"][f]) for f in BRFSS_FEATURES}
        patient_df = pd.DataFrame([patient_row])

        shap_res = explain_with_shap(patient_df)
        lime_res = explain_with_lime(patient_df)

        # Annotate assumed features in SHAP and LIME
        assumed_set = set(assembler_res.get("assumed_features", []))
        top_shap_sentences = []
        for f in shap_res.get("top5_features", []):
            s_text = f["sentence"]
            if f["feature"] in assumed_set:
                s_text += " [assumed value]"
            top_shap_sentences.append(s_text)

        top_lime_sentences = []
        for f in lime_res.get("top5_features", []):
            l_text = f["sentence"]
            # Check if any assumed feature name is in the condition string
            if any(af in f.get("feature_condition", "") for af in assumed_set):
                l_text += " [assumed value]"
            top_lime_sentences.append(l_text)

        risk_pred = {
            "predicted_class": shap_res["predicted_class"],
            "probabilities": shap_res["probabilities"],
            "features_used": 21,
            "completeness": assembler_res["completeness_message"],
            "known_condition": assembler_res["known_condition"],
        }
        context["risk"] = risk_pred
        context["xai"] = {
            "shap": shap_res,
            "lime": lime_res,
            "top_shap_features": top_shap_sentences,
            "top_lime_features": top_lime_sentences,
        }
        adapt_risk_result(patient_id, risk_pred, timestamp=timestamp, memory_agent=mem_agent)
        agent_status["risk_agent"] = "ok"
        agent_status["xai_agent"] = "ok"
    except Exception as e:
        agent_status["risk_agent"] = "failed"
        agent_status["xai_agent"] = "failed"
        errors["risk_and_xai"] = str(e)
    agent_timing["risk_and_xai"] = round(time.time() - t0, 3)

    # -------------------------------------------------------------
    # 7. Recommendation Agent
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        from agents.recommendation_agent.recommendation_agent import RecommendationAgent
        rec_agent = RecommendationAgent()
        risk_level = context["risk"].get("predicted_class", "Low")
        rec_res = rec_agent.generate_recommendations(
            risk_level=risk_level,
            probabilities=context["risk"].get("probabilities", {}),
            top_shap_features=context["xai"].get("top_shap_features", []),
            memory_summary=context["patient_memory"],
            imaging_result=context["medical_image"],
            patient_profile=context.get("feature_assembly", {}).get("features"),
            assumed_features=context.get("feature_assembly", {}).get("assumed_features", []),
        )
        context["recommendation"] = rec_res
        agent_status["recommendation_agent"] = "ok"
    except Exception as e:
        agent_status["recommendation_agent"] = "failed"
        errors["recommendation_agent"] = str(e)
    agent_timing["recommendation_agent"] = round(time.time() - t0, 3)

    total_time = round(time.time() - t_start, 3)
    peak_rss = round(_get_peak_rss_mb(), 2)

    # Summary synthesis
    is_partial = any(s.startswith("failed") or s.startswith("skipped") for s in agent_status.values())
    summary_text = _build_case_summary(patient_id, context, agent_status, is_partial)

    return {
        "patient_id": patient_id,
        "is_partial": is_partial,
        "agent_status": agent_status,
        "agent_timing_seconds": agent_timing,
        "total_time_seconds": total_time,
        "peak_rss_mb": peak_rss,
        "errors": errors,
        "context": context,
        "summary": summary_text,
    }


def _build_case_summary(patient_id: str, ctx: Dict[str, Any], status: Dict[str, str], is_partial: bool) -> str:
    """Builds a concise, structured clinical plain-English overview."""
    risk = ctx.get("risk", {}).get("predicted_class", "Not assessed")
    completeness = ctx.get("risk", {}).get("completeness", "")
    known_cond = ctx.get("risk", {}).get("known_condition", False)
    img = ctx.get("medical_image", {}).get("predicted_class", "Not assessed")
    labs = list(ctx.get("patient_memory", {}).get("latest_labs", {}).keys())
    meds = ctx.get("patient_memory", {}).get("medications", [])
    guidance_count = len(ctx.get("recommendation", {}).get("guidance", []))

    status_tag = "[PARTIAL OUTPUT]" if is_partial else "[COMPLETE PIPELINE]"

    lines = [
        f"MediTwin AI Case Summary for {patient_id} {status_tag}",
        f"• Risk Assessment: {risk} (Epidemiological metabolic risk via 21 BRFSS features; {completeness})",
    ]
    if known_cond:
        lines.append("  (Note: Documented diabetes diagnosis/medication identified as patient clinical condition)")
    lines.extend([
        f"• Chest Imaging Screening: {img}",
        f"• Patient Memory Record: {len(labs)} recent lab tests ({', '.join(labs) if labs else 'none'}), {len(meds)} recorded medications",
        f"• Educational Guidance: {guidance_count} targeted preventive health recommendations generated",
    ])
    if is_partial:
        skipped_or_failed = [f"{k}: {v}" for k, v in status.items() if v != "ok"]
        lines.append(f"• Incomplete Agents: {'; '.join(skipped_or_failed)}")

    return "\n".join(lines)
