"""
Lazy model loader registry and execution wrapper for heavy ML tasks.
Ensures single uvicorn worker safety and serializes heavy inferences.

Lock-wait timeout  → HTTP 503 "busy with another analysis"
Execution timeout  → HTTP 504 "took too long / models still loading"
"""
import sys
import os
import asyncio
import logging
import time
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import HTTPException, status

logger = logging.getLogger("meditwin.ml_service")

# Asyncio lock for ML inference requests (semaphore = 1 → one at a time)
_INFERENCE_SEMAPHORE: Optional[asyncio.Semaphore] = None
_SEMAPHORE_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _get_semaphore() -> asyncio.Semaphore:
    """Returns the single inference semaphore, re-binding to current loop if needed."""
    global _INFERENCE_SEMAPHORE, _SEMAPHORE_LOOP
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        if _INFERENCE_SEMAPHORE is None or _SEMAPHORE_LOOP is not loop:
            _INFERENCE_SEMAPHORE = asyncio.Semaphore(1)
            _SEMAPHORE_LOOP = loop
    elif _INFERENCE_SEMAPHORE is None:
        _INFERENCE_SEMAPHORE = asyncio.Semaphore(1)
    return _INFERENCE_SEMAPHORE


# Default execution timeout (seconds) for /v1/analyze (heavy); other routes use 120s
ANALYZE_TIMEOUT_SECONDS = 240.0
DEFAULT_TIMEOUT_SECONDS = 120.0

# Lock-wait timeout: how long to wait to acquire the semaphore before giving 503
_LOCK_WAIT_TIMEOUT = float(os.environ.get("ML_LOCK_WAIT_TIMEOUT", "60"))

# Warm-up state
_warming: bool = False
_warmup_task: Optional[asyncio.Task] = None


def get_process_rss_mb() -> float:
    """Reads VmRSS from /proc/self/status (Linux). Returns 0.0 elsewhere."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def is_warming() -> bool:
    return _warming


def check_loaded_models() -> Dict[str, bool]:
    """Inspects which agents currently have initialized model references in memory."""
    status_map: Dict[str, bool] = {}

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

    # Medical Image – Pediatric ResNet18
    try:
        from agents.medical_image_agent import medical_image_agent
        status_map["medical_image_pediatric_resnet18"] = medical_image_agent._model is not None
    except Exception:
        status_map["medical_image_pediatric_resnet18"] = False

    # Medical Image – Adult DenseNet121
    try:
        from agents.medical_image_agent import adult_xray_agent
        status_map["medical_image_adult_densenet121"] = adult_xray_agent._model is not None
    except Exception:
        status_map["medical_image_adult_densenet121"] = False

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


async def _warmup_sequence():
    """
    Loads all models one at a time in a background task.
    Acquires and immediately releases the semaphore between each load
    so normal requests can still proceed (they grab the lock themselves).
    """
    global _warming
    _warming = True
    logger.info("[WARMUP] Starting background model warm-up sequence")

    steps = [
        ("risk_xai",         _load_risk_and_xai),
        ("nlp",              _load_nlp),
        ("ocr_printed",      _load_ocr_printed),
        ("ocr_handwriting",  _load_ocr_handwriting),
        ("image_pediatric",  _load_image_pediatric),
        ("image_adult",      _load_image_adult),
    ]

    loop = asyncio.get_running_loop()
    for name, loader_fn in steps:
        t0 = time.time()
        try:
            # Acquire semaphore so warmup is serialized with real requests
            sem = _get_semaphore()
            async with sem:
                await loop.run_in_executor(None, loader_fn)
            elapsed = round(time.time() - t0, 2)
            rss = round(get_process_rss_mb(), 1)
            logger.info(f"[WARMUP] {name} loaded in {elapsed}s — RSS {rss} MB")
        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            logger.warning(f"[WARMUP] {name} failed after {elapsed}s: {exc}")
        # Small yield so the event loop can handle other requests
        await asyncio.sleep(0)

    _warming = False
    logger.info("[WARMUP] All models warm-up and forward passes complete")


def _load_risk_and_xai():
    from agents.xai_agent import xai_agent
    xai_agent._ensure_loaded()
    t_pass = time.time()
    try:
        import pandas as pd
        from agents.coordination.feature_assembler import load_training_defaults
        base_df = pd.DataFrame([load_training_defaults()])
        if xai_agent._model is not None:
            xai_agent._model.predict_proba(base_df)
        elapsed = round(time.time() - t_pass, 3)
        logger.info(f"[WARMUP] risk_xai forward pass completed in {elapsed}s")
    except Exception as e:
        logger.warning(f"[WARMUP] risk_xai forward pass warning: {e}")


def _load_nlp():
    from agents.nlp_agent import nlp_agent as _nlp
    _nlp._load()
    t_pass = time.time()
    try:
        if _nlp._pipeline is not None:
            _nlp._pipeline("warm up")
        elapsed = round(time.time() - t_pass, 3)
        logger.info(f"[WARMUP] nlp forward pass completed in {elapsed}s")
    except Exception as e:
        logger.warning(f"[WARMUP] nlp forward pass warning: {e}")


def _load_ocr_printed():
    from agents.ocr_agent import ocr_agent as _ocr
    _ocr._load_ocr_reader()
    t_pass = time.time()
    try:
        import numpy as np
        blank_img = np.zeros((100, 100, 3), dtype=np.uint8)
        if _ocr._ocr_reader is not None:
            _ocr._ocr_reader.readtext(blank_img)
        elapsed = round(time.time() - t_pass, 3)
        logger.info(f"[WARMUP] ocr_printed forward pass completed in {elapsed}s")
    except Exception as e:
        logger.warning(f"[WARMUP] ocr_printed forward pass warning: {e}")


def _load_ocr_handwriting():
    from agents.ocr_agent import ocr_agent as _ocr
    _ocr._load_hw_model()
    t_pass = time.time()
    try:
        from PIL import Image
        blank_crop = Image.new("RGB", (64, 64), color="white")
        _ocr.classify_handwritten_medicine(blank_crop)
        elapsed = round(time.time() - t_pass, 3)
        logger.info(f"[WARMUP] ocr_handwriting forward pass completed in {elapsed}s")
    except Exception as e:
        logger.warning(f"[WARMUP] ocr_handwriting forward pass warning: {e}")


def _load_image_pediatric():
    from agents.medical_image_agent import medical_image_agent as _mia
    _mia._load_model()
    t_pass = time.time()
    try:
        import torch
        dummy = torch.zeros(1, 3, 224, 224).to(_mia.DEVICE)
        with torch.no_grad():
            if _mia._model is not None:
                _mia._model(dummy)
        elapsed = round(time.time() - t_pass, 3)
        logger.info(f"[WARMUP] image_pediatric forward pass completed in {elapsed}s")
    except Exception as e:
        logger.warning(f"[WARMUP] image_pediatric forward pass warning: {e}")


def _load_image_adult():
    from agents.medical_image_agent import adult_xray_agent as _axr
    _axr._load_model()
    t_pass = time.time()
    try:
        import torch
        dummy = torch.zeros(1, 1, 224, 224).to(_axr.DEVICE)
        with torch.no_grad():
            if _axr._model is not None:
                _axr._model(dummy)
        elapsed = round(time.time() - t_pass, 3)
        logger.info(f"[WARMUP] image_adult forward pass completed in {elapsed}s")
    except Exception as e:
        logger.warning(f"[WARMUP] image_adult forward pass warning: {e}")


def start_warmup_task(loop: Optional[asyncio.AbstractEventLoop] = None):
    """Schedule the warm-up coroutine as a background asyncio task."""
    global _warmup_task
    try:
        _warmup_task = asyncio.ensure_future(_warmup_sequence())
    except RuntimeError:
        # No running loop yet (e.g. during import) — will be called from lifespan
        logger.warning("[WARMUP] Could not schedule warmup task (no running loop)")


async def run_heavy_inference(
    sync_func,
    *args,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    **kwargs,
) -> Any:
    """
    Acquires the inference semaphore (raises HTTP 503 on lock-wait timeout)
    then runs the synchronous heavy ML function in the threadpool executor
    (raises HTTP 504 on execution timeout).

    Parameters
    ----------
    timeout_seconds : float
        Wall-clock execution budget for the actual inference call.
        Use ANALYZE_TIMEOUT_SECONDS (240s) for /v1/analyze, default 120s elsewhere.
    """
    # Phase 1: Wait to acquire the lock
    sem = _get_semaphore()
    try:
        acquired = await asyncio.wait_for(
            sem.acquire(),
            timeout=_LOCK_WAIT_TIMEOUT
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "The analysis service is busy with another analysis. Please retry shortly.",
            },
        )

    # Phase 2: Run the inference with an execution-time budget
    try:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, lambda: sync_func(*args, **kwargs))
        return await asyncio.wait_for(future, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error": "The analysis took too long, probably because the models are still loading. Please retry in a minute.",
                "timeout_seconds": timeout_seconds,
            },
        )
    finally:
        sem.release()
