"""
Lazy model loader registry and execution wrapper for heavy ML tasks.
Ensures single uvicorn worker safety and serializes heavy inferences with a 120s timeout.
"""
import sys
import os
import asyncio
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import HTTPException, status

# Asyncio lock for ML inference requests
_INFERENCE_SEMAPHORE = asyncio.Semaphore(1)
INFERENCE_TIMEOUT_SECONDS = 120.0


def check_loaded_models() -> Dict[str, bool]:
    """Inspects which agents currently have initialized model references in memory."""
    status_map = {}
    
    # Risk & XAI
    try:
        from agents.xai_agent import xai_agent
        status_map["risk_prediction_xgboost"] = xai_agent._model is not None
        status_map["xai_shap_explainer"] = xai_agent._shap_explainer is not None
        status_map["xai_lime_explainer"] = xai_agent._lime_explainer is not None
    except Exception:
        status_map["risk_prediction_xgboost"] = False
        status_map["xai_shap_explainer"] = False
        status_map["xai_lime_explainer"] = False

    # Medical Image ResNet18
    try:
        from agents.medical_image_agent import medical_image_agent
        status_map["medical_image_resnet18"] = medical_image_agent._model is not None
    except Exception:
        status_map["medical_image_resnet18"] = False

    # OCR models
    try:
        from agents.ocr_agent import ocr_agent
        status_map["ocr_printed_easyocr"] = ocr_agent._ocr_reader is not None
        status_map["ocr_handwriting_resnet18"] = ocr_agent._hw_model is not None
    except Exception:
        status_map["ocr_printed_easyocr"] = False
        status_map["ocr_handwriting_resnet18"] = False

    # NLP DistilBERT
    try:
        from agents.nlp_agent import nlp_agent
        status_map["nlp_distilbert_ner"] = nlp_agent._model is not None
    except Exception:
        status_map["nlp_distilbert_ner"] = False

    return status_map


async def run_heavy_inference(sync_func, *args, **kwargs) -> Any:
    """
    Acquires the inference semaphore and runs the synchronous heavy ML function
    in the default threadpool executor with a 120-second timeout.
    If the timeout is exceeded, raises HTTP 503.
    """
    try:
        async with _INFERENCE_SEMAPHORE:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, lambda: sync_func(*args, **kwargs))
            return await asyncio.wait_for(future, timeout=INFERENCE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ML Service busy: Inference request timed out after 120 seconds. Please try again shortly.",
                "timeout_seconds": INFERENCE_TIMEOUT_SECONDS
            }
        )

