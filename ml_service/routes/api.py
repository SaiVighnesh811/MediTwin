"""
FastAPI route handlers for all ML endpoints.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status
from pydantic import ValidationError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_service.config import DISCLAIMER_TEXT, LIMITATIONS
from ml_service.security import verify_api_key, validate_and_save_upload
from ml_service.schemas import RiskRequest, NLPRequest, BRFSSFeatures
from ml_service.inference import check_loaded_models, run_heavy_inference, is_warming, ANALYZE_TIMEOUT_SECONDS

router = APIRouter()


def sanitize_numpy(obj: Any) -> Any:
    """Recursively converts numpy numbers and arrays to pure Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [sanitize_numpy(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_numpy(x) for x in obj]
    return obj


def compute_reliability(provided_or_extracted_count: int, predicted_class: str) -> Dict[str, Any]:
    """
    Computes reliability level ('low' | 'medium' | 'high') with plain-English reasons.
    - Low: < 9 of 21 features provided/extracted, OR predicted class is 'Medium' (unreliable class).
    - Medium: 9 to 17 provided/extracted, and class is not 'Medium'.
    - High: >= 18 provided/extracted, and class is not 'Medium'.
    """
    reasons = []
    is_medium_class = (predicted_class or "").lower() == "medium"

    if is_medium_class:
        reasons.append(
            "Predicted class is 'Medium' (Prediabetes), which has low precision and high uncertainty due to severe training class imbalance (1.8% prevalence in BRFSS data)."
        )

    if provided_or_extracted_count < 9:
        reasons.append(
            f"Only {provided_or_extracted_count} of 21 features were provided or extracted (fewer than 9). Empirical tests demonstrate High-class recall drops from ~62.6% to ~30.8% with 9 features."
        )
        level = "low"
    elif is_medium_class:
        level = "low"
    elif 9 <= provided_or_extracted_count <= 17:
        reasons.append(
            f"{provided_or_extracted_count} of 21 features were provided or extracted. Missing features rely on training set medians, introducing moderate uncertainty."
        )
        level = "medium"
    else:
        reasons.append(
            f"{provided_or_extracted_count} of 21 features were provided or extracted, giving high confidence in feature completeness without prediabetes class uncertainty."
        )
        level = "high"

    return {"level": level, "reasons": reasons}


# ─────────────────────────────────────────────────────────────
# 1. GET /health
# ─────────────────────────────────────────────────────────────
@router.get("/health")
def health_check():
    """
    Public health check endpoint.
    Reports service status, currently loaded models, and model descriptions from MODEL_CREDITS.md.
    """
    loaded_models = check_loaded_models()
    return {
        "status": "healthy",
        "service": "MediTwin ML Service",
        "version": "1.0.0",
        "warming": is_warming(),
        "loaded_models": loaded_models,
        "models_info": {
            "risk_prediction": {
                "name": "Class-Weighted XGBoost v2",
                "dataset": "BRFSS 2015 Diabetes Survey (253,680 rows, 21 features)",
                "classes": ["Low", "Medium", "High"],
            },
            "medical_image": {
                "name": "ResNet18 Binary Chest X-Ray Classifier",
                "source": "luanacarolina/pneumonia-chest-xray-classifier (HuggingFace)",
                "license": "CC BY 4.0",
                "architecture": "torchvision.models.resnet18",
                "dataset": "Kermany et al. pediatric chest X-ray dataset",
            },
            "nlp": {
                "name": "Biomedical NER All",
                "source": "d4data/biomedical-ner-all (HuggingFace)",
                "license": "Apache-2.0",
                "architecture": "DistilBertForTokenClassification",
                "dataset": "MACCROBAT biomedical annotations",
            },
            "ocr": {
                "printed": "EasyOCR (JaidedAI, PyTorch)",
                "handwriting": "Custom ResNet18 BD medicine classifier (78 classes)",
            },
            "xai": {
                "methods": ["Permutation SHAP (shap.Explainer)", "Local Interpretable Model-agnostic Explanations (LIME)"],
            },
            "recommendation": {
                "type": "Rule-based preventive lifestyle guidelines (CDC/AHA/WHO)",
            }
        },
        "disclaimer": DISCLAIMER_TEXT,
    }


# ─────────────────────────────────────────────────────────────
# 2. POST /v1/risk
# ─────────────────────────────────────────────────────────────
@router.post("/v1/risk", dependencies=[Depends(verify_api_key)])
async def predict_risk(req: RiskRequest):
    """
    Evaluates metabolic risk given any subset of the 21 BRFSS survey features.
    Assembles provided values first, then defaults for missing ones.
    """
    def _execute():
        from agents.coordination.feature_assembler import assemble_features, BRFSS_FEATURES
        from agents.xai_agent import xai_agent

        xai_agent._ensure_loaded()
        provided_dict = req.model_dump(exclude_none=True)
        assembled = assemble_features(provided_profile=provided_dict)
        
        df = pd.DataFrame([assembled["features"]])[BRFSS_FEATURES]
        proba = xai_agent._model.predict_proba(df)[0]
        pred_idx = int(np.argmax(proba))
        pred_class = xai_agent.CLASS_NAMES[pred_idx]

        probabilities = {c: round(float(proba[i]), 4) for i, c in enumerate(xai_agent.CLASS_NAMES)}
        reliability = compute_reliability(assembled["provided_or_extracted_count"], pred_class)

        return {
            "predicted_class": pred_class,
            "probabilities": probabilities,
            "completeness": assembled["completeness_message"],
            "provided_or_extracted_count": assembled["provided_or_extracted_count"],
            "assumed_count": assembled["assumed_count"],
            "assumed_features": assembled["assumed_features"],
            "known_condition": assembled["known_condition"],
            "features_used": 21,
            "reliability": reliability,
            "disclaimer": DISCLAIMER_TEXT,
            "limitations": LIMITATIONS["risk"],
        }

    return await run_heavy_inference(_execute)


# ─────────────────────────────────────────────────────────────
# 3. POST /v1/explain
# ─────────────────────────────────────────────────────────────
@router.post("/v1/explain", dependencies=[Depends(verify_api_key)])
async def explain_risk(req: RiskRequest):
    """
    Generates SHAP and LIME feature attribution with plain-English sentences for the case.
    """
    def _execute():
        from agents.coordination.feature_assembler import assemble_features, BRFSS_FEATURES
        from agents.xai_agent.xai_agent import explain_with_shap, explain_with_lime

        provided_dict = req.model_dump(exclude_none=True)
        assembled = assemble_features(provided_profile=provided_dict)
        patient_row = {f: float(assembled["features"][f]) for f in BRFSS_FEATURES}
        patient_df = pd.DataFrame([patient_row])

        shap_res = explain_with_shap(patient_df)
        lime_res = explain_with_lime(patient_df)

        assumed_set = set(assembled.get("assumed_features", []))

        # Annotate plain English sentences if assumed
        top_shap_features = []
        for f in shap_res.get("top5_features", []):
            item = dict(f)
            if item["feature"] in assumed_set:
                item["sentence"] += " [assumed value]"
            item["is_assumed"] = item["feature"] in assumed_set
            top_shap_features.append(item)

        top_lime_features = []
        for f in lime_res.get("top5_features", []):
            item = dict(f)
            cond_str = item.get("feature_condition", "")
            if any(af in cond_str for af in assumed_set):
                item["sentence"] += " [assumed value]"
            item["is_assumed"] = any(af in cond_str for af in assumed_set)
            top_lime_features.append(item)

        return sanitize_numpy({
            "predicted_class": shap_res["predicted_class"],
            "probabilities": shap_res["probabilities"],
            "shap_top5": top_shap_features,
            "lime_top5": top_lime_features,
            "completeness": assembled["completeness_message"],
            "assumed_features": assembled["assumed_features"],
            "runtime_seconds": {
                "shap": shap_res.get("runtime_seconds"),
                "lime": lime_res.get("runtime_seconds"),
            },
            "disclaimer": DISCLAIMER_TEXT,
            "limitations": LIMITATIONS["xai"],
        })

    return await run_heavy_inference(_execute)


# ─────────────────────────────────────────────────────────────
# 4. POST /v1/image
# ─────────────────────────────────────────────────────────────
@router.post("/v1/image", dependencies=[Depends(verify_api_key)])
async def analyze_xray(
    file: UploadFile = File(...),
    age_years: Optional[int] = Form(None),
):
    """
    Screens uploaded chest X-ray image for signs of pneumonia.
    Strictly accepts JPEG/PNG, max 8MB.
    age_years is required; routes <18 to pediatric Kermany ResNet18,
    >=18 to adult TorchXRayVision DenseNet121-res224-all.
    """
    if age_years is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "age_years is required for chest X-ray analysis"},
        )
    if not (0 <= age_years <= 120):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "age_years must be between 0 and 120"},
        )

    temp_path, _ = await validate_and_save_upload(file)
    try:
        from ml_service.xray_check import validate_xray_plausibility
        validate_xray_plausibility(temp_path)

        def _execute():
            if age_years < 18:
                # Pediatric path: Kermany ResNet18
                from agents.medical_image_agent.medical_image_agent import predict_image
                result = predict_image(temp_path, save_heatmap=False)
                return sanitize_numpy({
                    "model_used": "pediatric_kermany_resnet18",
                    "population": (
                        "Pediatric chest X-rays "
                        "(Guangzhou Women and Children's Medical Center)"
                    ),
                    "predicted_class": result["predicted_class"],
                    "pneumonia_probability": round(float(result["pneumonia_probability"]), 4),
                    "normal_probability": round(float(result["normal_probability"]), 4),
                    "threshold_used": result["threshold_used"],
                    "disclaimer": DISCLAIMER_TEXT,
                    "limitations": LIMITATIONS["image"],
                })
            else:
                # Adult path: TorchXRayVision DenseNet121-res224-all
                from agents.medical_image_agent.adult_xray_agent import predict_adult_xray
                result = predict_adult_xray(temp_path)
                return sanitize_numpy({
                    "model_used": result["model_used"],
                    "population": result["population"],
                    "pneumonia_score": result["pneumonia_score"],
                    "pneumonia_op_thresh": result["pneumonia_op_thresh"],
                    "pneumonia_op_thresh_source": result["pneumonia_op_thresh_source"],
                    "related_findings": result["related_findings"],
                    "all_scores": result["all_scores"],
                    "output_range_semantics": result["output_range_semantics"],
                    "inference_time_seconds": result["inference_time_seconds"],
                    "disclaimer": DISCLAIMER_TEXT,
                    "limitations": result["limitations"],
                })

        return await run_heavy_inference(_execute, timeout_seconds=ANALYZE_TIMEOUT_SECONDS)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────
# 5. POST /v1/ocr
# ─────────────────────────────────────────────────────────────
@router.post("/v1/ocr", dependencies=[Depends(verify_api_key)])
async def ocr_document(
    file: UploadFile = File(...),
    handwriting: bool = Form(False)
):
    """
    Extracts text and structured prescription / lab entities from document image.
    Optionally runs the 78-class handwriting classifier.
    """
    temp_path, _ = await validate_and_save_upload(file)
    try:
        def _execute():
            from agents.ocr_agent.ocr_agent import process_document
            res = process_document(temp_path, run_handwriting=handwriting)
            hw_candidates = res.get("handwriting_candidates", [])
            handwriting_top3 = []
            if handwriting and hw_candidates:
                first = hw_candidates[0]
                if first.get("top3"):
                    handwriting_top3 = first["top3"]
                elif first.get("name") != "Unclear":
                    handwriting_top3 = [{"rank": 1, "medicine_name": first["name"], "probability": first["probability"]}]

            return sanitize_numpy({
                "printed_text": res.get("raw_text", ""),
                "extracted_fields": res,
                "handwriting_candidates": hw_candidates,
                "handwriting_top3": handwriting_top3,
                "disclaimer": DISCLAIMER_TEXT,
                "limitations": LIMITATIONS["ocr"],
            })

        return await run_heavy_inference(_execute, timeout_seconds=ANALYZE_TIMEOUT_SECONDS)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────
# 6. POST /v1/nlp
# ─────────────────────────────────────────────────────────────
@router.post("/v1/nlp", dependencies=[Depends(verify_api_key)])
async def nlp_entities(req: NLPRequest):
    """
    Extracts biomedical entities (diseases, symptoms, medications, procedures) from clinical text.
    """
    def _execute():
        from agents.nlp_agent.nlp_agent import extract_entities
        res = extract_entities(req.text)
        return sanitize_numpy({
            "entities": res.get("entities", []),
            "diseases": res.get("diseases", []),
            "symptoms": res.get("symptoms", []),
            "medications": res.get("medications", []),
            "procedures": res.get("procedures", []),
            "history": res.get("history", []),
            "family_history": res.get("family_history", []),
            "disclaimer": DISCLAIMER_TEXT,
            "limitations": LIMITATIONS["nlp"],
        })

    return await run_heavy_inference(_execute)


# ─────────────────────────────────────────────────────────────
# 7. POST /v1/analyze
# ─────────────────────────────────────────────────────────────
@router.post("/v1/analyze", dependencies=[Depends(verify_api_key)])
async def run_full_analysis(
    features: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    age_years: Optional[int] = Form(None),
    use_memory: Optional[Union[bool, str]] = Form(True),
    document_image: Optional[UploadFile] = File(None),
    xray_image: Optional[UploadFile] = File(None),
):
    """
    Executes orchestrator run_case across multi-modal clinical inputs.
    Gracefully handles partial input and partial agent failures.
    """
    parsed_profile = {}
    if features:
        try:
            raw_feats = json.loads(features)
            validated = BRFSSFeatures.model_validate(raw_feats)
            parsed_profile = validated.model_dump(exclude_none=True)
        except (json.JSONDecodeError, ValidationError) as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": f"Invalid features JSON or schema violation: {str(e)}"}
            )

    doc_path = None
    xray_path = None

    if document_image and document_image.filename:
        doc_path, _ = await validate_and_save_upload(document_image)

    if xray_image and xray_image.filename:
        if age_years is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "age_years is required when an X-ray image is provided"},
            )
        if not (0 <= age_years <= 120):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "age_years must be between 0 and 120"},
            )
        xray_path, _ = await validate_and_save_upload(xray_image)

    # Minimum inputs rule: if fewer than 4 features provided and no document uploaded, reject upfront
    if len(parsed_profile) < 4 and not doc_path:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Please answer at least 4 questions or upload a lab report so a risk estimate is meaningful."}
        )

    try:
        if xray_path:
            from ml_service.xray_check import validate_xray_plausibility
            validate_xray_plausibility(xray_path)

        use_mem_bool = True
        if use_memory is not None:
            if isinstance(use_memory, bool):
                use_mem_bool = use_memory
            elif isinstance(use_memory, str):
                use_mem_bool = use_memory.lower() not in ("false", "0", "no", "off")

        def _execute():
            from agents.coordination.orchestrator import run_case
            case_data: Dict[str, Any] = {
                "patient_id": patient_id or "P_WEB_CASE",
                "patient_profile": parsed_profile,
                "clinical_notes": text,
                "use_memory": use_mem_bool,
            }
            if doc_path:
                case_data["prescription_image"] = doc_path
            if xray_path:
                case_data["chest_xray_path"] = xray_path
                if age_years is not None:
                    case_data["patient_age_years"] = age_years

            case_result = run_case(case_data)

            # Extract structured response
            ctx = case_result.get("context", {})
            risk_info = ctx.get("risk", {})
            feat_assembly = ctx.get("feature_assembly", {})
            count_prov = feat_assembly.get("provided_or_extracted_count", 0)

            # Minimum inputs rule: if fewer than 4 features provided or extracted, reject with 422
            if count_prov < 4:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={"error": "Please answer at least 4 questions or upload a lab report so a risk estimate is meaningful."}
                )

            pred_cls = risk_info.get("predicted_class", "")
            reliability = compute_reliability(count_prov, pred_cls)

            return sanitize_numpy({
                "patient_id": case_result.get("patient_id"),
                "is_partial": case_result.get("is_partial", False),
                "agent_status": case_result.get("agent_status", {}),
                "agent_timing_seconds": case_result.get("agent_timing_seconds", {}),
                "total_time_seconds": case_result.get("total_time_seconds"),
                "errors": case_result.get("errors", {}),
                "agent_errors": case_result.get("agent_errors", {}),
                "risk_assessment": risk_info,
                "sources": feat_assembly.get("sources", {}),
                "saved_records": feat_assembly.get("saved_records", []),
                "reliability": reliability,
                "explainability": {
                    "top_shap_features": ctx.get("xai", {}).get("top_shap_features", []),
                    "top_lime_features": ctx.get("xai", {}).get("top_lime_features", []),
                },
                "imaging": ctx.get("medical_image", {}),
                "ocr_findings": ctx.get("ocr", {}),
                "nlp_entities": ctx.get("nlp", {}),
                "patient_memory": {
                    "events_count": len(ctx.get("patient_memory", {}).get("timeline", [])),
                    "medications": ctx.get("patient_memory", {}).get("medications", []),
                    "conditions": ctx.get("patient_memory", {}).get("conditions", []),
                    "latest_labs": ctx.get("patient_memory", {}).get("latest_labs", {}),
                },
                "recommendations": ctx.get("recommendation", {}).get("guidance", []),
                "summary": case_result.get("summary", ""),
                "disclaimer": DISCLAIMER_TEXT,
                "limitations": LIMITATIONS["pipeline"],
            })

        return await run_heavy_inference(_execute, timeout_seconds=ANALYZE_TIMEOUT_SECONDS)
    finally:
        for p in (doc_path, xray_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
