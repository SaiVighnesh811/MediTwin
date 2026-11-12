"""
Chapter 12: ML Service Integration and Security Tests.
Tests:
(a) /health works without a key
(b) a protected route without a key returns 401
(c) wrong file type or an 9 MB file is rejected
(d) risk with all 21 features returns a class
(e) risk with only 6 features returns completeness "6 of 21" and the assumed list
(f) analyze with a synthetic case returns per-agent results
(g) analyze with one agent forced to fail still returns partial output
(h) a malformed feature value returns 422
"""
import io
import os
import sys
import json
import pytest
from PIL import Image
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_service.main import app
from ml_service.config import INTERNAL_API_KEY


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def auth_headers():
    return {"X-Internal-Key": INTERNAL_API_KEY}


# Helper to generate in-memory dummy image
def make_dummy_image(format="JPEG", size=(100, 100), color=(128, 128, 128)):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


# -------------------------------------------------------------
# Test (a): /health works without a key
# -------------------------------------------------------------
def test_a_health_works_without_key(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "loaded_models" in data
    assert "models_info" in data
    assert "disclaimer" in data


# -------------------------------------------------------------
# Test (b): a protected route without a key returns 401
# -------------------------------------------------------------
def test_b_protected_route_without_key_returns_401(client):
    # No header
    res = client.post("/v1/risk", json={"BMI": 25.0})
    assert res.status_code == 401
    assert "Unauthorized" in res.json().get("error", "")

    # Invalid header
    res2 = client.post("/v1/risk", json={"BMI": 25.0}, headers={"X-Internal-Key": "wrong-key"})
    assert res2.status_code == 401


# -------------------------------------------------------------
# Test (c): wrong file type or a 9 MB file is rejected
# -------------------------------------------------------------
def test_c_upload_validation_rejects_wrong_type_and_oversize(client, auth_headers):
    # 1. Wrong file type: text file disguised or with bad extension/bytes
    # age_years is now required; include it so the age check is not what triggers 400
    bad_bytes = b"This is a plain text file, not an image."
    res_bad_type = client.post(
        "/v1/image",
        data={"age_years": "30"},
        files={"file": ("fake.png", bad_bytes, "image/png")},
        headers=auth_headers
    )
    assert res_bad_type.status_code == 400, res_bad_type.text
    assert "not a valid jpeg or png image" in res_bad_type.json().get("error", "").lower()

    # 2. Oversize file: 9 MB (limit is 8 MB)
    oversized_bytes = b"0" * (9 * 1024 * 1024)
    res_oversize = client.post(
        "/v1/image",
        data={"age_years": "30"},
        files={"file": ("huge.jpg", oversized_bytes, "image/jpeg")},
        headers=auth_headers
    )
    assert res_oversize.status_code == 400, res_oversize.text
    assert "exceeds 8 mb" in res_oversize.json().get("error", "").lower()


# -------------------------------------------------------------
# Test (d): risk with all 21 features returns a class
# -------------------------------------------------------------
def test_d_risk_with_all_21_features_returns_class(client, auth_headers):
    payload = {
        "HighBP": 1.0,
        "HighChol": 1.0,
        "CholCheck": 1.0,
        "BMI": 28.5,
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
        "PhysHlth": 1.0,
        "DiffWalk": 0.0,
        "Sex": 1.0,
        "Age": 9.0,
        "Education": 5.0,
        "Income": 6.0
    }
    res = client.post("/v1/risk", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] in ["Low", "Medium", "High"]
    assert "probabilities" in data
    assert data["provided_or_extracted_count"] == 21
    assert data["assumed_count"] == 0
    assert "21 of 21 provided or extracted, 0 assumed" in data["completeness"]
    assert "disclaimer" in data
    assert len(data["limitations"]) > 0
    assert "reliability" in data
    assert data["reliability"]["level"] in ["low", "medium", "high"]
    assert len(data["reliability"]["reasons"]) > 0


# -------------------------------------------------------------
# Test (e): risk with only 6 features returns completeness "6 of 21" and assumed list
# -------------------------------------------------------------
def test_e_risk_with_6_features_returns_completeness(client, auth_headers):
    payload = {
        "HighBP": 1.0,
        "HighChol": 0.0,
        "BMI": 32.0,
        "Smoker": 1.0,
        "Age": 8.0,
        "Sex": 0.0
    }
    res = client.post("/v1/risk", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] in ["Low", "Medium", "High"]
    assert data["provided_or_extracted_count"] == 6
    assert data["assumed_count"] == 15
    assert "6 of 21 provided or extracted, 15 assumed" in data["completeness"]
    assert len(data["assumed_features"]) == 15
    assert "GenHlth" in data["assumed_features"]
    assert "DiffWalk" in data["assumed_features"]
    assert "reliability" in data
    assert data["reliability"]["level"] == "low"
    assert any("fewer than 9" in r for r in data["reliability"]["reasons"])


# -------------------------------------------------------------
# Test (f): analyze with a synthetic case returns per-agent results
# -------------------------------------------------------------
def test_f_analyze_synthetic_case_returns_per_agent(client, auth_headers):
    profile = {
        "HighBP": 1.0,
        "BMI": 31.0,
        "Age": 9.0,
        "Sex": 1.0
    }
    notes = "Patient reports history of hypertension and takes Amlodipine 5mg once daily."

    res = client.post(
        "/v1/analyze",
        data={
            "patient_id": "TEST_SYNTH_01",
            "features": json.dumps(profile),
            "text": notes,
        },
        headers=auth_headers
    )
    assert res.status_code == 200
    data = res.json()
    assert data["patient_id"] == "TEST_SYNTH_01"
    assert "agent_status" in data
    assert "agent_timing_seconds" in data
    assert "risk_assessment" in data
    assert "recommendations" in data
    assert "summary" in data
    assert "disclaimer" in data
    assert "reliability" in data
    assert data["reliability"]["level"] in ["low", "medium", "high"]
    assert data["agent_status"].get("risk_agent") == "ok"
    assert data["agent_status"].get("nlp_agent") == "ok"


# -------------------------------------------------------------
# Test (g): analyze with one agent forced to fail still returns partial output
# -------------------------------------------------------------
def test_g_analyze_with_forced_failure_returns_partial(client, auth_headers, monkeypatch):
    from agents.nlp_agent import nlp_agent

    def mock_extract_entities_fail(text):
        raise RuntimeError("Simulated NLP Model OOM Crash")

    monkeypatch.setattr(nlp_agent, "extract_entities", mock_extract_entities_fail)

    res = client.post(
        "/v1/analyze",
        data={
            "patient_id": "TEST_PARTIAL_FAIL_02",
            "text": "Some text that triggers NLP",
            "features": json.dumps({"BMI": 26.0, "Age": 5.0, "GenHlth": 2.0, "Sex": 1.0})
        },
        headers=auth_headers
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_partial"] is True
    assert data["agent_status"]["nlp_agent"] == "failed"
    assert "Simulated NLP Model OOM Crash" in data["errors"].get("nlp_agent", "")
    # Other agents still succeeded
    assert data["agent_status"]["risk_agent"] == "ok"
    assert "risk_assessment" in data
    assert "recommendations" in data


# -------------------------------------------------------------
# Test (h): a malformed feature value returns 422
# -------------------------------------------------------------
def test_h_malformed_feature_value_returns_422(client, auth_headers):
    # 1. BMI out of range (> 98)
    res1 = client.post("/v1/risk", json={"BMI": 150.0}, headers=auth_headers)
    assert res1.status_code == 422
    assert "BMI" in res1.text

    # 2. Unknown field forbidden
    res2 = client.post("/v1/risk", json={"UnknownFeature": 123}, headers=auth_headers)
    assert res2.status_code == 422
    assert "extra_forbidden" in res2.text or "UnknownFeature" in res2.text

    # 3. HighBP must be 0 or 1 (le=1.0)
    res3 = client.post("/v1/risk", json={"HighBP": 5.0}, headers=auth_headers)
    assert res3.status_code == 422


# =============================================================
# Chapter 3b: Adult X-ray model and age routing tests
# =============================================================

# Helper to generate a grayscale PNG in memory
def _make_grayscale_png(w=256, h=256) -> bytes:
    import numpy as np
    arr = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    img = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


# (i) /v1/image without age_years returns 422
def test_i_image_missing_age_returns_422(client, auth_headers):
    png = _make_grayscale_png()
    res = client.post(
        "/v1/image",
        files={"file": ("test.png", png, "image/png")},
        headers=auth_headers,
    )
    assert res.status_code == 422, res.text
    assert "age_years" in res.text.lower()


# (j) /v1/image with age=62 returns adult model keys
def test_j_image_adult_routing(client, auth_headers):
    png = _make_grayscale_png()
    res = client.post(
        "/v1/image",
        data={"age_years": "62"},
        files={"file": ("test.png", png, "image/png")},
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["model_used"] == "adult_torchxrayvision_densenet121_all"
    assert "pneumonia_score" in data
    assert isinstance(data["pneumonia_score"], float)
    assert "related_findings" in data
    assert "limitations" in data


# (k) /v1/image with age=8 stays on pediatric model
def test_k_image_pediatric_routing(client, auth_headers):
    png = _make_grayscale_png()
    res = client.post(
        "/v1/image",
        data={"age_years": "8"},
        files={"file": ("test.png", png, "image/png")},
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["model_used"] == "pediatric_kermany_resnet18"
    assert "predicted_class" in data
    assert data["predicted_class"] in ["NORMAL", "PNEUMONIA"]


# (l) /v1/image with out-of-range age returns 422
def test_l_image_invalid_age_returns_422(client, auth_headers):
    png = _make_grayscale_png()
    res = client.post(
        "/v1/image",
        data={"age_years": "200"},
        files={"file": ("test.png", png, "image/png")},
        headers=auth_headers,
    )
    assert res.status_code == 422, res.text


# (m) /v1/analyze with xray_image but no age_years returns 422
def test_m_analyze_xray_missing_age_returns_422(client, auth_headers):
    png = _make_grayscale_png()
    res = client.post(
        "/v1/analyze",
        data={"patient_id": "TEST_AGE_REQ"},
        files={"xray_image": ("chest.png", png, "image/png")},
        headers=auth_headers,
    )
    assert res.status_code == 422, res.text
    assert "age_years" in res.text.lower()

# (n) /health reports warming flag and loaded_models dict
def test_n_health_reports_warming_flag(client):
    """
    /health must include 'warming' (bool) and 'loaded_models' (dict).
    During tests (TestClient) the warmup task is not started, so warming=False.
    """
    res = client.get("/health")
    assert res.status_code == 200, res.text
    data = res.json()
    assert "warming" in data, "/health must include 'warming' key"
    assert isinstance(data["warming"], bool), "'warming' must be a bool"
    assert "loaded_models" in data, "/health must include 'loaded_models' key"
    # All known model keys should be present
    expected_keys = {
        "risk_prediction_xgboost",
        "xai_shap_explainer",
        "xai_lime_explainer",
        "medical_image_pediatric_resnet18",
        "medical_image_adult_densenet121",
        "ocr_printed_easyocr",
        "ocr_handwriting_resnet18",
        "nlp_distilbert_ner",
    }
    for k in expected_keys:
        assert k in data["loaded_models"], f"Missing '{k}' in loaded_models"
        assert isinstance(data["loaded_models"][k], bool), f"'{k}' must be bool"


# (o) run_heavy_inference raises HTTP 503 when lock-wait times out
def test_o_run_heavy_inference_raises_503_on_lock_wait_timeout():
    """
    If the semaphore is held and ML_LOCK_WAIT_TIMEOUT is very short, the caller
    should receive HTTP 503 with the 'busy' message.
    """
    import asyncio
    import os
    from fastapi import HTTPException

    # Patch lock-wait timeout to 0.01 s so it expires immediately
    import ml_service.inference as inf
    original = inf._LOCK_WAIT_TIMEOUT
    inf._LOCK_WAIT_TIMEOUT = 0.01

    async def _run():
        # Hold the semaphore manually so the next acquire will block
        sem = inf._get_semaphore()
        await sem.acquire()
        try:
            with pytest.raises(HTTPException) as exc_info:
                await inf.run_heavy_inference(lambda: None, timeout_seconds=5.0)
            assert exc_info.value.status_code == 503
            assert "busy" in str(exc_info.value.detail).lower()
        finally:
            sem.release()
            inf._LOCK_WAIT_TIMEOUT = original

    asyncio.run(_run())


# (p) run_heavy_inference raises HTTP 504 when execution times out
def test_p_run_heavy_inference_raises_504_on_execution_timeout():
    """
    When inference takes longer than timeout_seconds, the caller gets HTTP 504.
    """
    import asyncio
    import time
    from fastapi import HTTPException
    import ml_service.inference as inf

    def slow_fn():
        time.sleep(5)

    async def _run():
        with pytest.raises(HTTPException) as exc_info:
            await inf.run_heavy_inference(slow_fn, timeout_seconds=0.05)
        assert exc_info.value.status_code == 504
        assert "too long" in str(exc_info.value.detail).lower()

    asyncio.run(_run())


# (q) Orchestrator logs [START], [END], and [FAIL] lines to console and returns agent_errors
def test_q_orchestrator_logs_start_end_and_failure(client, auth_headers, monkeypatch, caplog):
    """
    Verify orchestrator logs [START] and [END] lines and [FAIL] on failure,
    and returns agent_errors in the response.
    """
    import logging
    from agents.nlp_agent import nlp_agent

    def mock_extract_entities_fail(text):
        raise RuntimeError("Simulated agent failure for caplog test")

    monkeypatch.setattr(nlp_agent, "extract_entities", mock_extract_entities_fail)

    with caplog.at_level(logging.INFO):
        res = client.post(
            "/v1/analyze",
            data={
                "patient_id": "TEST_LOG_01",
                "text": "Some text triggering nlp",
                "features": json.dumps({"BMI": 25.0, "Age": 6.0, "GenHlth": 2.0, "Sex": 1.0}),
            },
            headers=auth_headers,
        )
    assert res.status_code == 200
    data = res.json()
    assert "agent_errors" in data
    assert "nlp_agent" in data["agent_errors"]
    assert "RuntimeError" in data["agent_errors"]["nlp_agent"]

    log_text = caplog.text
    assert "[START] nlp_agent" in log_text
    assert "[END] nlp_agent in" in log_text
    assert "[FAIL] nlp_agent" in log_text
    assert "RuntimeError" in log_text


# (r) NLP extraction keeps "cholesterol" as one single entity
def test_r_nlp_cholesterol_is_one_entity(client, auth_headers):
    """
    Verify NLP extraction treats 'cholesterol' as a single entity without word-piece splitting.
    """
    test_sent = (
        "62-year-old male with hypertension and high cholesterol presents with fatigue. "
        "Takes Amlodipine 5 mg once daily. Denies chest pain."
    )
    res = client.post("/v1/nlp", json={"text": test_sent}, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    entities = data["entities"]

    matching = [e for e in entities if e["text"].lower() == "cholesterol"]
    assert len(matching) == 1, f"Expected exactly 1 'cholesterol' entity, found {len(matching)}: {matching}"
    assert matching[0]["text"] == "cholesterol"

    split_pieces = [e for e in entities if e["text"].lower() in ("cho", "les", "terol", "##les", "##terol")]
    assert len(split_pieces) == 0, f"Found split word pieces: {split_pieces}"


# (s) OCR agent handles _classes as a list when OR-ed with EXTRA_MEDS set
def test_s_ocr_classes_list_merged_with_extra_meds():
    """
    Verify OCR agent line 154: set(_classes or []) | EXTRA_MEDS
    works when _classes is a list of strings, raising no exception
    and including both the list items and EXTRA_MEDS.
    """
    import agents.ocr_agent.ocr_agent as ocr_mod

    saved_classes = ocr_mod._classes
    try:
        ocr_mod._classes = ["CustomMedAlpha", "CustomMedBeta"]
        result = ocr_mod.extract_fields("Rx: CustomMedAlpha 500mg once daily")
        assert result["type"] == "prescription"
        assert result["medicine"] == "CustomMedAlpha"

        all_meds = set(ocr_mod._classes or []) | ocr_mod.EXTRA_MEDS
        assert "CustomMedAlpha" in all_meds
        assert "CustomMedBeta" in all_meds
        for extra in list(ocr_mod.EXTRA_MEDS)[:5]:
            assert extra in all_meds
    finally:
        ocr_mod._classes = saved_classes


# (t) OCR line-by-line parser on 5-line sample document
def test_t_ocr_parser_five_lines():
    """
    Chapter 15d unit test on exact 5-line sample prescription expecting:
    - Metformin 500 mg twice daily 30 days
    - Amlodipine 5 mg once daily 30 days
    - Fasting Blood Glucose 110 mg/dL (above printed 70-100)
    - Total Cholesterol 195 mg/dL (printed 'less than 200', within/not above)
    """
    from agents.ocr_agent.ocr_agent import extract_fields

    sample_text = (
        "SAMPLE PRESCRIPTION (test data, not a real patient)\n"
        "Tab Metformin 500 mg twice daily for 30 days\n"
        "Tab Amlodipine 5 mg once daily for 30 days\n"
        "Fasting Blood Glucose 110 mg/dL Ref 70-100\n"
        "Total Cholesterol 195 mg/dL Ref less than 200"
    )

    parsed = extract_fields(sample_text)
    assert len(parsed["medicines"]) == 2
    assert len(parsed["labs"]) == 2

    # Medicine 1: Metformin
    m1 = parsed["medicines"][0]
    assert m1["medicine"] == "Metformin"
    assert m1["dosage"] == "500 mg"
    assert m1["frequency"] == "twice daily"
    assert m1["duration"] == "30 days"

    # Medicine 2: Amlodipine
    m2 = parsed["medicines"][1]
    assert m2["medicine"] == "Amlodipine"
    assert m2["dosage"] == "5 mg"
    assert m2["frequency"] == "once daily"
    assert m2["duration"] == "30 days"

    # Lab 1: Fasting Blood Glucose
    l1 = parsed["labs"][0]
    assert l1["test"] == "Fasting Blood Glucose"
    assert l1["value"] == "110"
    assert l1["unit"] == "mg/dL"
    assert l1["reference"] == "70-100"
    assert l1["flag"] == "above"

    # Lab 2: Total Cholesterol
    l2 = parsed["labs"][1]
    assert l2["test"] == "Total Cholesterol"
    assert l2["value"] == "195"
    assert l2["unit"] == "mg/dL"
    assert l2["reference"] == "less than 200"
    assert l2["flag"] == "within"


# (u) OCR line without dose remains null and does not borrow from next line
def test_u_ocr_line_without_dose_remains_null():
    """
    Chapter 15d: A line with a medicine but no dose followed by a line with a dose
    must keep the first line's dosage as None (never borrowing from the next line).
    """
    from agents.ocr_agent.ocr_agent import extract_fields

    text = (
        "Tab Metformin\n"
        "Tab Amlodipine 5 mg once daily for 30 days"
    )

    parsed = extract_fields(text)
    assert len(parsed["medicines"]) == 2
    assert parsed["medicines"][0]["medicine"] == "Metformin"
    assert parsed["medicines"][0]["dosage"] is None
    assert parsed["medicines"][1]["medicine"] == "Amlodipine"
    assert parsed["medicines"][1]["dosage"] == "5 mg"


# (v) Patient memory stores clean unique names from dict entities and skips negated
def test_v_patient_memory_dict_entities_and_deduplication(tmp_path):
    """
    Chapter 15d: Patient memory adapter and summary must:
    - store clean unique list of names from dict-shaped entities
    - de-duplicate case-insensitively
    - skip negated entities
    """
    from agents.patient_memory_agent.patient_memory_agent import PatientMemoryAgent, adapt_nlp_result

    mem = PatientMemoryAgent(storage_dir=str(tmp_path))
    patient_id = "TEST_P_CLEAN_01"

    nlp_result = {
        "medications": [
            {"text": "Amlodipine", "start": 14, "end": 24, "negated": False},
            {"text": "amlodipine", "start": 30, "end": 40, "negated": False},
            {"text": "Metformin", "start": 50, "end": 59, "negated": False},
            {"text": "Aspirin", "start": 70, "end": 77, "negated": True},
        ],
        "diseases": [
            {"text": "Hypertension", "start": 0, "end": 12, "negated": False},
            {"text": "hypertension", "start": 20, "end": 32, "negated": False},
            {"text": "Lung Cancer", "start": 40, "end": 51, "negated": True},
        ],
        "symptoms": [
            {"text": "Fatigue", "start": 0, "end": 7, "negated": False},
            {"text": "fatigue", "start": 10, "end": 17, "negated": False},
            {"text": "Chest Pain", "start": 20, "end": 30, "negated": True},
        ],
    }

    adapt_nlp_result(patient_id, nlp_result, memory_agent=mem)
    summary = mem.summary(patient_id)

    assert summary["medications"] == ["Amlodipine", "Metformin"]
    assert summary["diagnoses"] == ["Hypertension"]
    assert summary["symptoms"] == ["Fatigue"]
    assert "Aspirin" not in summary["medications"]
    assert "Lung Cancer" not in summary["diagnoses"]
    assert "Chest Pain" not in summary["symptoms"]
    for med in summary["medications"]:
        assert isinstance(med, str)
        assert "{" not in med


# (w) Chapter 15d-2 3a & 3c: EasyOCR mock tuples (ordered & shuffled) reach OCR parser & patient memory
def test_w_ocr_mock_readtext_boxes_and_memory(tmp_path, monkeypatch):
    """
    Chapter 15d-2:
    3a) Mock EasyOCR readtext to return the 5 (bbox, text, conf) tuples with realistic bboxes.
        Run read_printed + extract_fields, assert 2 medicines, 2 labs, and latest_labs reaches memory.
    3c) Boxes returned out of order (shuffled y) still sort top-to-bottom and give the exact same results.
    """
    import random
    from unittest.mock import MagicMock
    import agents.ocr_agent.ocr_agent as ocr_mod
    from agents.patient_memory_agent.patient_memory_agent import PatientMemoryAgent, adapt_ocr_result

    sample_boxes = [
        ([[10, 20], [300, 20], [300, 45], [10, 45]], "SAMPLE PRESCRIPTION (test data, not a real patient)", 0.99),
        ([[10, 60], [320, 60], [320, 85], [10, 85]], "Tab Metformin 500 mg twice daily for 30 days", 0.97),
        ([[10, 100], [310, 100], [310, 125], [10, 125]], "Tab Amlodipine 5 mg once daily for 30 days", 0.96),
        ([[10, 140], [330, 140], [330, 165], [10, 165]], "Fasting Blood Glucose 110 mg/dL Ref 70-100", 0.95),
        ([[10, 180], [340, 180], [340, 205], [10, 205]], "Total Cholesterol 195 mg/dL Ref less than 200", 0.94),
    ]

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = sample_boxes
    monkeypatch.setattr(ocr_mod, "_ocr_reader", mock_reader)

    # 3a: Run public OCR function (read_printed + extract_fields)
    lines_text = ocr_mod.read_printed("dummy_rx_sample.png")
    parsed = ocr_mod.extract_fields(lines_text)

    # Assert 2 medicines
    assert len(parsed["medicines"]) == 2
    assert parsed["medicines"][0]["medicine"] == "Metformin"
    assert parsed["medicines"][0]["dosage"] == "500 mg"
    assert parsed["medicines"][0]["frequency"] == "twice daily"
    assert parsed["medicines"][0]["duration"] == "30 days"

    assert parsed["medicines"][1]["medicine"] == "Amlodipine"
    assert parsed["medicines"][1]["dosage"] == "5 mg"
    assert parsed["medicines"][1]["frequency"] == "once daily"
    assert parsed["medicines"][1]["duration"] == "30 days"

    # Assert 2 labs
    assert len(parsed["labs"]) == 2
    assert parsed["labs"][0]["test"] == "Fasting Blood Glucose"
    assert parsed["labs"][0]["value"] == "110"
    assert parsed["labs"][0]["unit"] == "mg/dL"
    assert parsed["labs"][0]["reference"] == "70-100"
    assert parsed["labs"][0]["flag"] == "above"

    assert parsed["labs"][1]["test"] == "Total Cholesterol"
    assert parsed["labs"][1]["value"] == "195"
    assert parsed["labs"][1]["unit"] == "mg/dL"
    assert parsed["labs"][1]["reference"] == "less than 200"
    assert parsed["labs"][1]["flag"] == "within"

    # Assert latest_labs reaches patient memory
    mem = PatientMemoryAgent(storage_dir=str(tmp_path))
    patient_id = "TEST_P_15D2_OCR"
    adapt_ocr_result(patient_id, parsed, timestamp="2026-11-11T08:00:00Z", memory_agent=mem, raw_text=lines_text)
    summary = mem.summary(patient_id)
    assert "Fasting Blood Glucose" in summary["latest_labs"]
    assert summary["latest_labs"]["Fasting Blood Glucose"]["value"] == "110"
    assert summary["latest_labs"]["Fasting Blood Glucose"]["flag"] == "above"
    assert summary["latest_labs"]["Fasting Blood Glucose"]["reference"] == "70-100"
    assert "Total Cholesterol" in summary["latest_labs"]
    assert summary["latest_labs"]["Total Cholesterol"]["value"] == "195"
    assert summary["latest_labs"]["Total Cholesterol"]["flag"] == "within"
    assert summary["latest_labs"]["Total Cholesterol"]["reference"] == "less than 200"

    # 3c: Out of order boxes (shuffled y) still give the exact same result
    shuffled_boxes = list(sample_boxes)
    random.seed(12345)
    random.shuffle(shuffled_boxes)
    mock_reader.readtext.return_value = shuffled_boxes

    shuffled_text = ocr_mod.read_printed("dummy_rx_sample.png")
    parsed_shuffled = ocr_mod.extract_fields(shuffled_text)

    assert len(parsed_shuffled["medicines"]) == 2
    assert parsed_shuffled["medicines"][0]["medicine"] == "Metformin"
    assert parsed_shuffled["medicines"][0]["dosage"] == "500 mg"
    assert parsed_shuffled["medicines"][1]["medicine"] == "Amlodipine"
    assert parsed_shuffled["medicines"][1]["dosage"] == "5 mg"

    assert len(parsed_shuffled["labs"]) == 2
    assert parsed_shuffled["labs"][0]["test"] == "Fasting Blood Glucose"
    assert parsed_shuffled["labs"][0]["flag"] == "above"
    assert parsed_shuffled["labs"][1]["test"] == "Total Cholesterol"
    assert parsed_shuffled["labs"][1]["flag"] == "within"


# (x) Chapter 15d-2 3b: Fallback for single-line joined raw text string
def test_x_ocr_fallback_single_line_joined_string():
    """
    Chapter 15d-2 3b: Feed the EXACT joined single-line raw_text string to the fallback
    and assert the same 2 medicines and 2 labs.
    """
    from agents.ocr_agent.ocr_agent import extract_fields

    exact_joined_string = (
        "SAMPLE PRESCRIPTION (test data, not a real patient) "
        "Tab Metformin 500 mg twice daily for 30 days "
        "Tab Amlodipine 5 mg once daily for 30 days "
        "Fasting Blood Glucose 110 mg/dL Ref 70-100 "
        "Total Cholesterol 195 mg/dL Ref less than 200"
    )

    parsed = extract_fields(exact_joined_string)

    assert len(parsed["medicines"]) == 2
    assert parsed["medicines"][0]["medicine"] == "Metformin"
    assert parsed["medicines"][0]["dosage"] == "500 mg"
    assert parsed["medicines"][0]["frequency"] == "twice daily"
    assert parsed["medicines"][0]["duration"] == "30 days"

    assert parsed["medicines"][1]["medicine"] == "Amlodipine"
    assert parsed["medicines"][1]["dosage"] == "5 mg"
    assert parsed["medicines"][1]["frequency"] == "once daily"
    assert parsed["medicines"][1]["duration"] == "30 days"

    assert len(parsed["labs"]) == 2
    assert parsed["labs"][0]["test"] == "Fasting Blood Glucose"
    assert parsed["labs"][0]["value"] == "110"
    assert parsed["labs"][0]["unit"] == "mg/dL"
    assert parsed["labs"][0]["reference"] == "70-100"
    assert parsed["labs"][0]["flag"] == "above"

    assert parsed["labs"][1]["test"] == "Total Cholesterol"
    assert parsed["labs"][1]["value"] == "195"
    assert parsed["labs"][1]["unit"] == "mg/dL"
    assert parsed["labs"][1]["reference"] == "less than 200"
    assert parsed["labs"][1]["flag"] == "within"


# (y) Chapter 16a: 5-line sample prescription tilted 0, 3, and 6 degrees with y jitter
def test_y_ocr_tilted_sample_prescription():
    """
    Chapter 16a Test 1:
    The 5-line sample prescription tilted 0, 3 and 6 degrees with y jitter
    must give 2 medicines (Metformin 500 mg twice daily 30 days; Amlodipine 5 mg once daily 30 days)
    and 2 labs (glucose 110 above, cholesterol 195 within).
    """
    import math
    from agents.ocr_agent.ocr_agent import extract_fields

    def rotate_point(x, y, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return [x * cos_a - y * sin_a, x * sin_a + y * cos_a]

    lines = [
        "SAMPLE PRESCRIPTION (test data, not a real patient)",
        "Tab Metformin 500 mg twice daily for 30 days",
        "Tab Amlodipine 5 mg once daily for 30 days",
        "Fasting Blood Glucose 110 mg/dL Ref 70-100",
        "Total Cholesterol 195 mg/dL Ref less than 200",
    ]
    jitters = [1, -2, 2, -1, 3]

    for deg in [0, 3, 6]:
        rad = math.radians(deg)
        boxes = []
        for i, line in enumerate(lines):
            y_top = 30 + i * 40 + jitters[i]
            y_bot = y_top + 20
            pts = [[10, y_top], [350, y_top], [350, y_bot], [10, y_bot]]
            rot_pts = [rotate_point(x, y, rad) for x, y in pts]
            boxes.append((rot_pts, line, 0.95))

        parsed = extract_fields(boxes)

        assert len(parsed["medicines"]) == 2, f"Failed at deg={deg}"
        assert parsed["medicines"][0]["medicine"] == "Metformin"
        assert parsed["medicines"][0]["dosage"] == "500 mg"
        assert parsed["medicines"][0]["frequency"] == "twice daily"
        assert parsed["medicines"][0]["duration"] == "30 days"

        assert parsed["medicines"][1]["medicine"] == "Amlodipine"
        assert parsed["medicines"][1]["dosage"] == "5 mg"
        assert parsed["medicines"][1]["frequency"] == "once daily"
        assert parsed["medicines"][1]["duration"] == "30 days"

        assert len(parsed["labs"]) == 2, f"Failed at deg={deg}"
        assert parsed["labs"][0]["test"] == "Fasting Blood Glucose"
        assert parsed["labs"][0]["value"] == "110"
        assert parsed["labs"][0]["unit"] == "mg/dL"
        assert parsed["labs"][0]["flag"] == "above"

        assert parsed["labs"][1]["test"] == "Total Cholesterol"
        assert parsed["labs"][1]["value"] == "195"
        assert parsed["labs"][1]["unit"] == "mg/dL"
        assert parsed["labs"][1]["flag"] == "within"


# (z) Chapter 16a: 7-row table lab report as cell boxes
def test_z_ocr_table_report_seven_rows():
    """
    Chapter 16a Test 2:
    The 7-row table lab report as cell boxes must give 7 labs with correct
    value, unit, reference and flag.
    """
    from agents.ocr_agent.ocr_agent import extract_fields

    rows_data = [
        ("Fasting Blood Glucose", "110", "mg/dL", "70 - 100", "above"),
        ("Total Cholesterol", "195", "mg/dL", "< 200", "within"),
        ("HDL Cholesterol", "38", "mg/dL", "> 40", "below"),
        ("Triglycerides", "160", "mg/dL", "< 150", "above"),
        ("HbA1c", "6.8", "%", "4.0 - 5.6", "above"),
        ("Serum Creatinine", "1.0", "mg/dL", "0.7 - 1.3", "within"),
        ("BUN", "18", "mg/dL", "7 - 20", "within"),
    ]

    table_boxes = []
    for row_idx, (t_name, val, unit, ref, _) in enumerate(rows_data):
        y_top = 50 + row_idx * 35
        y_bot = y_top + 22
        # cell 1: name
        table_boxes.append(([[20, y_top], [180, y_top], [180, y_bot], [20, y_bot]], t_name, 0.99))
        # cell 2: value
        table_boxes.append(([[190, y_top], [230, y_top], [230, y_bot], [190, y_bot]], val, 0.98))
        # cell 3: unit
        table_boxes.append(([[240, y_top], [290, y_top], [290, y_bot], [240, y_bot]], unit, 0.97))
        # cell 4: reference
        table_boxes.append(([[300, y_top], [400, y_top], [400, y_bot], [300, y_bot]], ref, 0.96))

    parsed = extract_fields(table_boxes)

    assert len(parsed["labs"]) == 7
    for i, (exp_name, exp_val, exp_unit, exp_ref, exp_flag) in enumerate(rows_data):
        lab = parsed["labs"][i]
        assert lab["test"] == exp_name
        assert lab["value"] == exp_val
        assert lab["unit"] == exp_unit
        assert lab["reference"] == exp_ref
        assert lab["flag"] == exp_flag


# (aa) Chapter 16a: Multi-format boxes, split rows, unclear dose/unit safety, fuzzy matching
def test_aa_ocr_multi_format_and_safety():
    """
    Chapter 16a Tests 3-9:
    - 'Tab Amlodipine' and 'Smg OD x 30 days' on one row give dose unclear, once daily, 30 days
    - 'Tab Metformin 500 ma twice daily for 30' plus a 'days' box gives dose unclear, twice daily, 30 days
    - 'Metfommin 500 mg' is matched to Metformin
    - 'HbAlc 6.8 % Ref 4.0-5.6' gives HbA1c above
    - 'HDL Cholesterol 38 mg/dL Ref more than 40' gives below
    - 'Tab Losartan 50 mg once daily' is found
    - a medicine line with no dose still gives a null dose
    """
    from agents.ocr_agent.ocr_agent import extract_fields

    # 1. Row split: 'Tab Amlodipine' and 'Smg OD x 30 days'
    boxes_amlodipine = [
        ([[10, 50], [150, 50], [150, 70], [10, 70]], "Tab Amlodipine", 0.99),
        ([[160, 50], [350, 50], [350, 70], [160, 70]], "Smg OD x 30 days", 0.98),
    ]
    p1 = extract_fields(boxes_amlodipine)
    assert len(p1["medicines"]) == 1
    m1 = p1["medicines"][0]
    assert m1["medicine"] == "Amlodipine"
    assert m1["dosage"] is None
    assert m1["dose_unclear"] is True
    assert m1["frequency"] == "once daily"
    assert m1["duration"] == "30 days"

    # 2. Row split: 'Tab Metformin 500 ma twice daily for 30' + 'days'
    boxes_metformin = [
        ([[10, 90], [280, 90], [280, 110], [10, 110]], "Tab Metformin 500 ma twice daily for 30", 0.99),
        ([[290, 90], [350, 90], [350, 110], [290, 110]], "days", 0.98),
    ]
    p2 = extract_fields(boxes_metformin)
    assert len(p2["medicines"]) == 1
    m2 = p2["medicines"][0]
    assert m2["medicine"] == "Metformin"
    assert m2["dosage"] is None
    assert m2["dose_unclear"] is True
    assert m2["frequency"] == "twice daily"
    assert m2["duration"] == "30 days"

    # 3. Fuzzy matching: 'Metfommin 500 mg'
    p3 = extract_fields("Metfommin 500 mg")
    assert len(p3["medicines"]) == 1
    m3 = p3["medicines"][0]
    assert m3["medicine"] == "Metformin"
    assert m3["raw_medicine"] == "Metfommin"
    assert m3["dosage"] == "500 mg"

    # 4. Lab alias: 'HbAlc 6.8 % Ref 4.0-5.6'
    p4 = extract_fields("HbAlc 6.8 % Ref 4.0-5.6")
    assert len(p4["labs"]) == 1
    l4 = p4["labs"][0]
    assert l4["test"] == "HbA1c"
    assert l4["value"] == "6.8"
    assert l4["unit"] == "%"
    assert l4["flag"] == "above"

    # 5. Lab comparison: 'HDL Cholesterol 38 mg/dL Ref more than 40'
    p5 = extract_fields("HDL Cholesterol 38 mg/dL Ref more than 40")
    assert len(p5["labs"]) == 1
    l5 = p5["labs"][0]
    assert l5["test"] == "HDL Cholesterol"
    assert l5["value"] == "38"
    assert l5["unit"] == "mg/dL"
    assert l5["flag"] == "below"

    # 6. Added medicine: 'Tab Losartan 50 mg once daily'
    p6 = extract_fields("Tab Losartan 50 mg once daily")
    assert len(p6["medicines"]) == 1
    m6 = p6["medicines"][0]
    assert m6["medicine"] == "Losartan"
    assert m6["dosage"] == "50 mg"
    assert m6["frequency"] == "once daily"

    # 7. Medicine with no dose gives null dose and dose_unclear False
    p7 = extract_fields("Tab Metformin twice daily")
    assert len(p7["medicines"]) == 1
    m7 = p7["medicines"][0]
    assert m7["medicine"] == "Metformin"
    assert m7["dosage"] is None
    assert m7["dose_unclear"] is False
    assert m7["frequency"] == "twice daily"


# (ab) Chapter 16e: Handwriting classifier wiring, thresholding, and memory safety
def test_ab_ocr_handwriting_classifier_wiring_and_safety(tmp_path, monkeypatch):
    """
    Chapter 16e:
    1. A box that parses as a dose is never classified
    2. A below-threshold prediction gives 'Unclear'
    3. An image with two unknown boxes gives two candidates
    4. A candidate never appears in confirmed medicines or patient memory
    """
    import agents.ocr_agent.ocr_agent as ocr_mod
    from agents.patient_memory_agent.patient_memory_agent import PatientMemoryAgent, adapt_ocr_result

    classified_calls = []

    def mock_classify(image_input):
        classified_calls.append(image_input)
        if len(classified_calls) == 1:
            return [
                {"rank": 1, "medicine_name": "Napa", "probability": 0.85},
                {"rank": 2, "medicine_name": "Aceta", "probability": 0.08},
                {"rank": 3, "medicine_name": "Parpy", "probability": 0.03},
            ]
        else:
            return [
                {"rank": 1, "medicine_name": "Amlodipine", "probability": 0.22},
                {"rank": 2, "medicine_name": "Metformin", "probability": 0.18},
                {"rank": 3, "medicine_name": "Losartan", "probability": 0.15},
            ]

    monkeypatch.setattr(ocr_mod, "classify_handwritten_medicine", mock_classify)

    # 4 boxes:
    # 1. "Tab Amlodipine" (known medicine) -> skipped
    # 2. "500 mg" (dose) -> skipped
    # 3. "scribble_unknown_one" (unknown, conf 0.40 < HW_GATE) -> classified (returns 0.85 -> "Napa")
    # 4. "scribble_unknown_two" (unknown, conf 0.35 < HW_GATE) -> classified (returns 0.22 -> "Unclear")
    boxes = [
        ([[10, 20], [120, 20], [120, 40], [10, 40]], "Tab Amlodipine", 0.98),
        ([[130, 20], [200, 20], [200, 40], [130, 40]], "500 mg", 0.97),
        ([[10, 60], [150, 60], [150, 80], [10, 80]], "scribble_unknown_one", 0.40),
        ([[10, 100], [150, 100], [150, 120], [10, 120]], "scribble_unknown_two", 0.35),
    ]

    res = ocr_mod.process_document(boxes, run_handwriting=True)

    # 1. Check exactly 2 unknown boxes classified (dose and known med skipped)
    assert len(classified_calls) == 2, f"Expected 2 classifier calls, got {len(classified_calls)}"

    # 2. Check candidates
    cands = res["handwriting_candidates"]
    assert len(cands) == 2

    # Above threshold (0.85 >= 0.3) -> name is "Napa"
    assert cands[0]["name"] == "Napa"
    assert cands[0]["probability"] == 0.85
    assert cands[0]["source"] == "handwriting_model"
    assert cands[0]["ocr_confidence"] == 0.40

    # Below threshold (0.22 < 0.3) -> name is "Unclear"
    assert cands[1]["name"] == "Unclear"
    assert cands[1]["probability"] == 0.22
    assert cands[1]["source"] == "handwriting_model"
    assert len(cands[1]["top3"]) == 3
    assert cands[1]["ocr_confidence"] == 0.35

    # 3. Check confirmed medicines: candidate NEVER appears in medicines
    confirmed_names = [m["medicine"] for m in res["medicines"]]
    assert "Amlodipine" in confirmed_names
    assert "Napa" not in confirmed_names
    assert "Unclear" not in confirmed_names

    # 4. Check patient memory: candidate NEVER reaches patient memory
    mem = PatientMemoryAgent(storage_dir=str(tmp_path))
    patient_id = "TEST_P_16E_HW"
    adapt_ocr_result(patient_id, res, memory_agent=mem)
    summary = mem.summary(patient_id)
    assert "Amlodipine" in summary["medications"]
    assert "Napa" not in summary["medications"]
    assert "Unclear" not in summary["medications"]


# (ac) Chapter 16f: HW gate, table headers, blank/oversized image gate, cap of 5, table flags, cwice daily
def test_ac_ocr_hw_gate_table_and_frequency(monkeypatch):
    """
    Chapter 16f:
    - a printed box at confidence 0.9 is never classified
    - a low-confidence unknown box is classified
    - a table header or 'Triglycerides' is never classified
    - a blank image and a 1300x460 image never use the whole-image path
    - the cap of 5 handwriting candidates
    - the table fixture gives flag_printed H for glucose and unclear references for '200 ;'
    - 'cwice daily' gives frequency_unclear: True, frequency: None
    """
    from PIL import Image
    import agents.ocr_agent.ocr_agent as ocr_mod

    # 1. Frequency misread 'cwice daily'
    rx_res = ocr_mod.extract_fields("Tab Metformin 500 mg cwice daily for 30 days")
    assert len(rx_res["medicines"]) == 1
    m0 = rx_res["medicines"][0]
    assert m0["medicine"] == "Metformin"
    assert m0["dosage"] == "500 mg"
    assert m0["frequency"] is None
    assert m0["frequency_unclear"] is True
    assert m0["duration"] == "30 days"

    # 2. Table row parsing with flag_printed and unclear references
    lab_a = ocr_mod.parse_line_lab("Fasting Blood Glucose 110 mg/dL 70 100 H")
    assert lab_a is not None
    assert lab_a["test"] == "Fasting Blood Glucose"
    assert lab_a["value"] == "110"
    assert lab_a["reference"] == "70 100"
    assert lab_a["reference_inferred"] is True
    assert lab_a["reference_unclear"] is False
    assert lab_a["flag"] == "above"
    assert lab_a["flag_printed"] == "H"

    lab_b = ocr_mod.parse_line_lab("Total Cholesterol 195 mg/dL 200 ;")
    assert lab_b is not None
    assert lab_b["test"] == "Total Cholesterol"
    assert lab_b["value"] == "195"
    assert lab_b["reference_unclear"] is True
    assert lab_b["flag"] is None

    # 3. Handwriting Gate & Filter checks
    classified_texts = []
    def mock_hw(img_or_crop):
        classified_texts.append("called")
        return [
            {"rank": 1, "medicine_name": f"Med_{len(classified_texts)}", "probability": 0.80 - 0.05 * len(classified_texts)},
            {"rank": 2, "medicine_name": "Alt", "probability": 0.10},
            {"rank": 3, "medicine_name": "Alt2", "probability": 0.05},
        ]
    monkeypatch.setattr(ocr_mod, "classify_handwritten_medicine", mock_hw)

    boxes = [
        # Box 1: high confidence (0.90) unknown word -> must NOT be classified (conf >= HW_GATE)
        ([[10, 10], [100, 10], [100, 30], [10, 30]], "unknown_high_conf", 0.90),
        # Box 2: table header 'Result' -> must NOT be classified
        ([[10, 40], [100, 40], [100, 60], [10, 60]], "Result", 0.20),
        # Box 3: known lab test 'Triglycerides' -> must NOT be classified
        ([[10, 70], [100, 70], [100, 90], [10, 90]], "Triglycerides", 0.20),
        # Box 4: pure numbers/symbols '70 - 100' -> must NOT be classified
        ([[10, 100], [100, 100], [100, 120], [10, 120]], "70 - 100", 0.15),
        # Boxes 5-11: 7 low-confidence unknown boxes -> should be classified, then capped at 5
        ([[10, 130], [100, 130], [100, 150], [10, 150]], "scribble_1", 0.30),
        ([[10, 160], [100, 160], [100, 180], [10, 180]], "scribble_2", 0.32),
        ([[10, 190], [100, 190], [100, 210], [10, 210]], "scribble_3", 0.28),
        ([[10, 220], [100, 220], [100, 240], [10, 240]], "scribble_4", 0.25),
        ([[10, 250], [100, 250], [100, 270], [10, 270]], "scribble_5", 0.35),
        ([[10, 280], [100, 280], [100, 300], [10, 300]], "scribble_6", 0.33),
        ([[10, 310], [100, 310], [100, 330], [10, 330]], "scribble_7", 0.31),
    ]

    p = ocr_mod.process_document(boxes, run_handwriting=True)
    assert len(classified_texts) == 7
    assert len(p["handwriting_candidates"]) == 5

    # 4. Whole-image path rejection on blank image and 1300x460 image
    classified_texts.clear()
    img_large = Image.new("RGB", (1300, 460), color="white")
    p_large = ocr_mod.process_document(img_large, run_handwriting=True)
    assert len(p_large["handwriting_candidates"]) == 0
    assert len(classified_texts) == 0

    img_blank = Image.new("RGB", (400, 300), color="white")
    p_blank = ocr_mod.process_document(img_blank, run_handwriting=True)
    assert len(p_blank["handwriting_candidates"]) == 0
    assert len(classified_texts) == 0


# (ad) Chapter 16b: NLP corner cases (negation, family history, rules, filter, dedupe)
def test_ad_nlp_corner_cases_negation_family_rules_filter_and_dedupe(monkeypatch):
    """
    Chapter 16b: Tests NLP corner cases:
    1. 'No chest pain but has cough and mild fever.' -> chest pain negated, cough & fever not negated.
    2. 'Patient denies chest pain and shortness of breath. No fever.' -> fever found in 'No fever' and negated.
    3. 'Father had type 2 diabetes. Mother has hypertension. Patient has no history of diabetes.'
       -> 'Father had hypertension' gives family history and does not set HighBP.
    4. 'Pt c/o fatigue. Known HTN and T2DM, on metformin 500mg bd.' -> HTN gives hypertension (rule), Pt dropped.
    5. 'Pateint has hypertention and diabetis, takes amlodipin 5 mg.' -> Pateint is not a medication.
    6. 'Severe headache for 3 days with nausea.' -> nausea found.
    7. 150 repeated sentences give at most 30 entities.
    8. 'On 12/03/2025 BP was 150/95.' -> BP dropped.
    """
    import re
    import agents.nlp_agent.nlp_agent as nlp_mod
    from agents.coordination.feature_assembler import extract_features_from_memory
    from agents.patient_memory_agent.patient_memory_agent import adapt_nlp_result, PatientMemoryAgent

    def mock_pipeline_call(text):
        ents = []
        for m in re.finditer(r"\b(hypertension|type 2 diabetes|diabetes)\b", text, re.I):
            ents.append({"entity_group": "Disease_disorder", "score": 0.95, "word": m.group(0), "start": m.start(), "end": m.end()})
        if "chest pain" in text:
            idx = text.index("chest pain")
            ents.append({"entity_group": "Sign_symptom", "score": 0.95, "word": "chest pain", "start": idx, "end": idx + 10})
        if "cough" in text:
            idx = text.index("cough")
            ents.append({"entity_group": "Sign_symptom", "score": 0.92, "word": "cough", "start": idx, "end": idx + 5})
        if "Pt" in text:
            idx = text.index("Pt")
            ents.append({"entity_group": "Diagnostic_procedure", "score": 0.70, "word": "Pt", "start": idx, "end": idx + 2})
        if "/" in text and "150/95" not in text:
            idx = text.index("/")
            ents.append({"entity_group": "Diagnostic_procedure", "score": 0.60, "word": "/", "start": idx, "end": idx + 1})
        if "Pateint" in text:
            idx = text.index("Pateint")
            ents.append({"entity_group": "Medication", "score": 0.85, "word": "Pateint", "start": idx, "end": idx + 7})
        if "BP" in text:
            idx = text.index("BP")
            ents.append({"entity_group": "Diagnostic_procedure", "score": 0.80, "word": "BP", "start": idx, "end": idx + 2})
        if "metformin" in text:
            idx = text.index("metformin")
            ents.append({"entity_group": "Medication", "score": 0.96, "word": "metformin", "start": idx, "end": idx + 9})
        if "amlodipin" in text:
            idx = text.index("amlodipin")
            ents.append({"entity_group": "Medication", "score": 0.94, "word": "amlodipin", "start": idx, "end": idx + 9})
        return ents

    monkeypatch.setattr(nlp_mod, "_pipeline", mock_pipeline_call)

    # 1. 'No chest pain but has cough and mild fever.'
    res1 = nlp_mod.extract_entities("No chest pain but has cough and mild fever.")
    cp = next(e for e in res1["entities"] if e["text"].lower() == "chest pain")
    assert cp["negated"] is True, "chest pain must be negated"
    cough = next(e for e in res1["entities"] if e["text"].lower() == "cough")
    assert cough["negated"] is False, "cough must NOT be negated"
    fever = next(e for e in res1["entities"] if e["text"].lower() == "fever")
    assert fever["negated"] is False, "fever must NOT be negated"

    # 2. 'Patient denies chest pain and shortness of breath. No fever.'
    res2 = nlp_mod.extract_entities("Patient denies chest pain and shortness of breath. No fever.")
    fever2 = next(e for e in res2["entities"] if e["text"].lower() == "fever")
    assert fever2["negated"] is True, "fever must be found in 'No fever' and negated"

    # 3. Family history and HighBP
    res_fam = nlp_mod.extract_entities("Father had hypertension.")
    assert len(res_fam["family_history"]) >= 1, "Father had hypertension gives family history"
    assert any(e["text"].lower() == "hypertension" and e["subject"] == "family" for e in res_fam["family_history"])
    assert len(res_fam["diseases"]) == 0, "family history entity must not be in patient diseases"

    mem_agent = PatientMemoryAgent(storage_dir="/tmp/test_mem_ch16b")
    events = adapt_nlp_result("TEST_PATIENT_FH", res_fam, memory_agent=mem_agent)
    assert len([ev for ev in events if ev.get("event_type") == "diagnosis"]) == 0
    mem_summary = mem_agent.summary("TEST_PATIENT_FH")
    assert "hypertension" not in [str(c).lower() for c in mem_summary.get("conditions", [])]
    extracted_feats, known_cond = extract_features_from_memory(mem_summary)
    assert extracted_feats.get("HighBP", 0.0) == 0.0, "Family history does not set HighBP"

    res3 = nlp_mod.extract_entities("Father had type 2 diabetes. Mother has hypertension. Patient has no history of diabetes.")
    fh_names = [e["text"].lower() for e in res3["family_history"]]
    assert "type 2 diabetes mellitus" in fh_names or "type 2 diabetes" in fh_names
    assert "hypertension" in fh_names
    pt_diab = [e for e in res3["entities"] if e["subject"] == "patient" and "diabet" in e["text"].lower()]
    assert len(pt_diab) >= 1 and pt_diab[0]["negated"] is True

    # 4. 'Pt c/o fatigue. Known HTN and T2DM, on metformin 500mg bd.'
    res4 = nlp_mod.extract_entities("Pt c/o fatigue. Known HTN and T2DM, on metformin 500mg bd.")
    assert not any(e["text"] == "Pt" for e in res4["entities"]), "Pt must be dropped"
    assert not any(e["text"] == "/" for e in res4["entities"]), "/ must be dropped"
    htn_ent = next(e for e in res4["entities"] if e["text"].lower() == "hypertension")
    assert htn_ent["source"] == "rule", "HTN gives hypertension (rule)"
    t2dm_ent = next(e for e in res4["entities"] if e["text"].lower() == "type 2 diabetes mellitus")
    assert t2dm_ent["source"] == "rule"

    # 5. 'Pateint has hypertention and diabetis, takes amlodipin 5 mg.'
    res5 = nlp_mod.extract_entities("Pateint has hypertention and diabetis, takes amlodipin 5 mg.")
    assert not any(e["text"].lower() == "pateint" and e["group"] == "MEDICATION" for e in res5["entities"]), "Pateint is not a medication"
    assert any(e["group"] == "MEDICATION" and "amlodip" in e["text"].lower() for e in res5["entities"])

    # 6. 'Severe headache for 3 days with nausea.'
    res6 = nlp_mod.extract_entities("Severe headache for 3 days with nausea.")
    assert any(e["text"].lower() == "nausea" for e in res6["entities"]), "nausea found"

    # 7. 150 repeated sentences give at most 30 entities
    repeated_text = "Pt c/o fatigue. Known HTN and T2DM, on metformin 500mg bd. " * 150
    res7 = nlp_mod.extract_entities(repeated_text)
    assert len(res7["entities"]) <= 30, f"150 repeated sentences must give at most 30 entities, got {len(res7['entities'])}"
    htn_rep = next(e for e in res7["entities"] if e["text"].lower() == "hypertension")
    assert htn_rep["count"] == 150, f"Expected count 150, got {htn_rep['count']}"

    # 8. 'On 12/03/2025 BP was 150/95.'
    res8 = nlp_mod.extract_entities("On 12/03/2025 BP was 150/95.")
    assert not any(e["text"] == "BP" for e in res8["entities"]), "BP alone is not an entity"



# ── Chapter 16c: X-Ray plausibility gate ─────────────────────────────────────

def test_ae_xray_plausibility_gate(client, auth_headers, monkeypatch):
    """
    Chapter 16c – reject images that are clearly not a chest X-ray.

    Synthetic arrays:
      1. Colour gradient 256x256 RGB  → REJECT (sat > SAT_MAX)
      2. Mostly white 1300x460        → REJECT (white_frac, std, aspect)
      3. Blank 500x500 (all 255)      → REJECT (white_frac, std)
      4. Grayscale noise 1024x1024    → ACCEPT
      5. Portrait 800x1000 noise      → ACCEPT
      6. Route POST /v1/image with colour gradient → 422, no model called
    """
    import numpy as np
    from PIL import Image as PILImage
    from ml_service.xray_check import (
        check_xray_plausibility,
        XRAY_REJECT_MESSAGE,
        SAT_MAX,
        WHITE_MAX,
        STD_MIN,
        ASPECT_RANGE,
    )

    # 1. Colour gradient 256x256 RGB – high saturation
    arr_colour = np.zeros((256, 256, 3), dtype=np.uint8)
    arr_colour[:, :, 0] = np.arange(256, dtype=np.uint8).reshape(1, 256)  # red gradient
    img_colour = PILImage.fromarray(arr_colour, mode="RGB")
    ok, reason, m = check_xray_plausibility(img_colour)
    assert not ok, f"Colour gradient must be rejected, got ok=True, metrics={m}"
    assert m["sat"] > SAT_MAX, f"Colour gradient sat {m['sat']} should exceed {SAT_MAX}"

    # 2. Mostly white 1300x460 – high white_frac, low std, extreme aspect
    arr_white_wide = np.full((460, 1300), 250, dtype=np.uint8)
    img_white_wide = PILImage.fromarray(arr_white_wide, mode="L")
    ok2, reason2, m2 = check_xray_plausibility(img_white_wide)
    assert not ok2, f"Mostly-white 1300x460 must be rejected, got ok=True, metrics={m2}"

    # 3. Blank 500x500 – all white
    arr_blank = np.full((500, 500), 255, dtype=np.uint8)
    img_blank = PILImage.fromarray(arr_blank, mode="L")
    ok3, reason3, m3 = check_xray_plausibility(img_blank)
    assert not ok3, f"Blank page must be rejected, got ok=True, metrics={m3}"
    assert m3["white_frac"] > WHITE_MAX
    assert m3["std"] < STD_MIN

    # 4. Grayscale noise 1024x1024 – should pass
    rng = np.random.RandomState(99)
    arr_noise = rng.randint(20, 200, size=(1024, 1024), dtype=np.uint8)
    img_noise = PILImage.fromarray(arr_noise, mode="L")
    ok4, reason4, m4 = check_xray_plausibility(img_noise)
    assert ok4, f"Grayscale noise 1024x1024 must be accepted, reason={reason4}, metrics={m4}"

    # 5. Portrait 800x1000 grayscale noise – should pass
    arr_portrait = rng.randint(20, 200, size=(1000, 800), dtype=np.uint8)
    img_portrait = PILImage.fromarray(arr_portrait, mode="L")
    ok5, reason5, m5 = check_xray_plausibility(img_portrait)
    assert ok5, f"Portrait noise 800x1000 must be accepted, reason={reason5}, metrics={m5}"

    # 6. Route test – POST /v1/image with colour gradient → 422, no model called
    import io

    buf = io.BytesIO()
    img_colour.save(buf, format="PNG")
    buf.seek(0)

    model_called = {"predict_image": False, "predict_adult_xray": False}

    def fake_predict_image(*a, **kw):
        model_called["predict_image"] = True
        return {"predicted_class": "NORMAL", "probabilities": {"NORMAL": 0.9, "PNEUMONIA": 0.1}}

    def fake_predict_adult(*a, **kw):
        model_called["predict_adult_xray"] = True
        return {"pneumonia_score": 0.1, "findings": {}}

    monkeypatch.setattr("agents.medical_image_agent.medical_image_agent.predict_image", fake_predict_image)
    monkeypatch.setattr("agents.medical_image_agent.adult_xray_agent.predict_adult_xray", fake_predict_adult)

    resp = client.post(
        "/v1/image",
        data={"age_years": "30"},
        files={"file": ("test_colour.png", buf, "image/png")},
        headers=auth_headers,
    )
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert XRAY_REJECT_MESSAGE in (body.get("error", "") or body.get("detail", {}).get("error", "")), (
        f"422 body must contain reject message, got {body}"
    )
    assert not model_called["predict_image"], "predict_image must NOT be called for rejected image"
    assert not model_called["predict_adult_xray"], "predict_adult_xray must NOT be called for rejected image"


# ── Chapter 16 Final: NLP Filter, Memory Transparency, Min Inputs ─────────────

def test_af_chapter16_final_nlp_filter_memory_transparency_min_inputs(client, auth_headers, monkeypatch, tmp_path):
    """
    Chapter 16 Final tests:
    - ITEM 1: Mocked NER output for dropped OTHER, dose attached to MEDICATION, single words dropped,
              dates/digits dropped, blood pressure dropped, family subject and never stored in memory.
    - ITEM 2: Memory transparency (source='saved_record', saved_records with saved_text & date, use_memory=False).
    - ITEM 3: Minimum inputs rule (fewer than 4 features returns 422 with exact message; empty {} too).
    """
    import json
    from agents.nlp_agent import nlp_agent
    from agents.patient_memory_agent.patient_memory_agent import PatientMemoryAgent, adapt_nlp_result
    from agents.coordination.feature_assembler import assemble_features

    test_text = (
        "Patient seen on 12/03/2025. BP was 150/95. Father has severe hypertension. "
        "Prescribed Metformin with dose 500 mg daily. Standalone 10 ml in separate sentence. "
        "Mother and sister report mild fever."
    )

    words = [
        ("Date", "12/03/2025"),
        ("Diagnostic_procedure", "BP"),
        ("Biological_structure", "150/95"),
        ("Sign_symptom", "Father"),
        ("Sign_symptom", "severe"),
        ("Disease_disorder", "hypertension"),
        ("Medication", "Metformin"),
        ("Dosage", "500 mg daily"),
        ("Dosage", "10 ml"),
        ("Sign_symptom", "Mother"),
        ("Sign_symptom", "sister"),
        ("Sign_symptom", "mild"),
        ("Sign_symptom", "fever"),
    ]
    mock_raw_output = []
    for grp, w in words:
        s = test_text.find(w)
        e = s + len(w)
        mock_raw_output.append({"entity_group": grp, "score": 0.95, "start": s, "end": e, "word": w})

    monkeypatch.setattr(nlp_agent, "_load", lambda: None)
    monkeypatch.setattr(nlp_agent, "_pipeline", lambda t: mock_raw_output)

    res = nlp_agent.extract_entities(test_text)

    # 1. OTHER groups are NOT in entities
    assert not any(e.get("group") == "OTHER" for e in res["entities"]), "OTHER group must not be in entities"

    # 2. Dose attached to nearest MEDICATION in same sentence
    met_ent = next((e for e in res["entities"] if e["text"].lower() == "metformin"), None)
    assert met_ent is not None, "Metformin must be in entities"
    assert met_ent.get("dose") == "500 mg daily", f"Dose must be attached to Metformin, got {met_ent.get('dose')}"

    # 3. Standalone dose in separate sentence must be dropped (not in entities)
    assert not any("10 ml" in e["text"] for e in res["entities"]), "Standalone dose must be dropped"

    # 4. Dropped single words: Father, Mother, Sister, Brother, mild, moderate, severe
    for word in ["father", "mother", "sister", "brother", "mild", "moderate", "severe"]:
        assert not any(e["text"].lower() == word for e in res["entities"]), f"'{word}' must be dropped"

    # 5. Dropped date and digits/slashes
    assert not any(e["text"] in ["12/03/2025", "150/95"] for e in res["entities"]), "Date and digits/slashes dropped"
    assert not any(e["text"].upper() == "BP" for e in res["entities"]), "BP dropped"

    # 6. Family subject on entities when sentence mentions family
    htn = next((e for e in res["entities"] if e["text"].lower() == "hypertension"), None)
    assert htn is not None and htn["subject"] == "family", "Hypertension must have subject: 'family'"
    fever = next((e for e in res["entities"] if e["text"].lower() == "fever"), None)
    assert fever is not None and fever["subject"] == "family", "Fever must have subject: 'family'"

    # 7. Family entities never stored in patient memory
    mem_dir = str(tmp_path / "mem_test_16_final")
    mem_agent = PatientMemoryAgent(storage_dir=mem_dir)
    events = adapt_nlp_result("P_TEST_FH", res, memory_agent=mem_agent)
    assert not any(ev.get("data", {}).get("diagnosis") == "hypertension" for ev in events), "Family diagnosis not stored"
    assert not any(ev.get("data", {}).get("symptom") == "fever" for ev in events), "Family symptom not stored"
    summary = mem_agent.summary("P_TEST_FH")
    assert "hypertension" not in [str(c).lower() for c in summary.get("conditions", [])]
    assert "fever" not in [str(s).lower() for s in summary.get("symptoms", [])]

    # ITEM 2: Memory transparency
    mem_summary = {
        "conditions": ["Hypertension"],
        "medications": ["Amlodipine 5mg"],
        "latest_labs": {"Total Cholesterol": {"value": 240.0, "unit": "mg/dL", "timestamp": "2025-05-01T10:00:00"}},
    }
    # With use_memory=True
    assembled_mem_on = assemble_features(
        provided_profile={"BMI": 26.0, "Age": 6.0, "GenHlth": 2.0, "Sex": 1.0},
        memory_summary=mem_summary,
        use_memory=True,
    )
    assert assembled_mem_on["sources"]["HighBP"] == "saved_record"
    assert assembled_mem_on["sources"]["HighChol"] == "saved_record"
    assert len(assembled_mem_on["saved_records"]) >= 2
    bp_rec = next(r for r in assembled_mem_on["saved_records"] if r["feature"] == "HighBP")
    assert "Hypertension" in bp_rec["saved_text"] or "Amlodipine" in bp_rec["saved_text"]
    chol_rec = next(r for r in assembled_mem_on["saved_records"] if r["feature"] == "HighChol")
    assert "240" in chol_rec["saved_text"]
    assert chol_rec["date"] == "2025-05-01T10:00:00"

    # With use_memory=False
    assembled_mem_off = assemble_features(
        provided_profile={"BMI": 26.0, "Age": 6.0, "GenHlth": 2.0, "Sex": 1.0},
        memory_summary=mem_summary,
        use_memory=False,
        current_doc_features={"HighChol": 1.0},  # current document upload
    )
    assert assembled_mem_off["sources"]["HighBP"] == "assumed", "HighBP should be assumed when use_memory=False"
    assert assembled_mem_off["sources"]["HighChol"] == "extracted", "Current document upload stays extracted"
    assert len(assembled_mem_off["saved_records"]) == 0

    # ITEM 3: Minimum inputs rule in /v1/analyze
    # (a) Empty features {} and no document -> 422
    resp_empty = client.post(
        "/v1/analyze",
        data={"features": "{}"},
        headers=auth_headers,
    )
    assert resp_empty.status_code == 422
    assert "Please answer at least 4 questions or upload a lab report" in resp_empty.text

    # (b) Fewer than 4 features (e.g. 2 features) and no document -> 422
    resp_two = client.post(
        "/v1/analyze",
        data={"features": json.dumps({"BMI": 24.0, "Age": 5.0})},
        headers=auth_headers,
    )
    assert resp_two.status_code == 422
    assert "Please answer at least 4 questions or upload a lab report" in resp_two.text


# -------------------------------------------------------------
# Test (ag): Chapter 17 - Medicine-Aware Guidance, Source Labels & Notes
# -------------------------------------------------------------
def test_ag_chapter17_medicine_guidance_and_source_labels(client, auth_headers):
    from agents.recommendation_agent.recommendation_agent import RecommendationAgent
    from agents.coordination.feature_assembler import assemble_features

    agent = RecommendationAgent()

    # 1. Glycemic card absent without lab/diabetes trigger
    out_no_glycemic = agent.generate_recommendations(
        risk_level="Low",
        patient_profile={"BMI": 22.0, "HighBP": 0.0, "HighChol": 0.0, "PhysActivity": 1.0},
        memory_summary={"latest_labs": {}, "diagnoses": [], "medications": []},
    )
    assert not any(c["rule_id"] == "REC_GLYCEMIC_LAB" for c in out_no_glycemic["guidance"]), (
        "Glycemic card must be absent when no lab or diabetes trigger is present"
    )

    # 2. Glycemic card present with elevated lab (HbA1c >= 5.7)
    out_lab_glycemic = agent.generate_recommendations(
        risk_level="Low",
        patient_profile={"BMI": 22.0, "HighBP": 0.0, "HighChol": 0.0},
        memory_summary={"latest_labs": {"HbA1c": {"value": 6.2, "timestamp": "2026-11-10"}}},
    )
    gly_card = next((c for c in out_lab_glycemic["guidance"] if c["rule_id"] == "REC_GLYCEMIC_LAB"), None)
    assert gly_card is not None
    assert "recent elevated glycemic indicators" in gly_card["text"]
    assert "6.2" in gly_card["text"]
    assert "2026-11-10" in gly_card["text"]
    assert "saved lab record: HbA1c 6.2% (2026-11-10)" in gly_card["based_on"]

    # 3. Glycemic card present with diabetes answer but no elevated lab -> general nutrition wording
    out_answer_glycemic = agent.generate_recommendations(
        risk_level="Low",
        patient_profile={"Diabetes_binary": 1.0, "BMI": 22.0},
        memory_summary={"latest_labs": {}},
    )
    gly_card_ans = next((c for c in out_answer_glycemic["guidance"] if c["rule_id"] == "REC_GLYCEMIC_LAB"), None)
    assert gly_card_ans is not None
    assert "recent elevated glycemic indicators" not in gly_card_ans["text"], (
        "Must not claim recent elevated readings without elevated lab"
    )
    assert "Glycemic and nutrition guidance" in gly_card_ans["text"]
    assert gly_card_ans["based_on"] == "your answer: diabetes = yes"

    # 4. Every card has a Based on line
    for c in out_lab_glycemic["guidance"]:
        assert "based_on" in c and len(c["based_on"].strip()) > 0, f"Card {c['rule_id']} missing based_on"

    # 5. Medicine card shows "dose unclear" when missing and small built-in list source_note
    out_med_unclear = agent.generate_recommendations(
        risk_level="Low",
        document_medicines=[{"medicine": "Omeprazole"}],
    )
    omep_card = next((c for c in out_med_unclear["guidance"] if "OMEPRAZOLE" in c["rule_id"]), None)
    assert omep_card is not None
    assert "dose unclear" in omep_card["text"]
    assert "stomach acid" in omep_card["text"]
    assert omep_card["title"] == "Medicines found in your document"
    assert "Extracted from your document - check it against the original" in omep_card["text"]
    assert omep_card["source_note"] == "MediTwin small built-in medicine list (6 names, not a verified drug database)"

    # 6. Duplicate note: "listed more than once; please check"
    out_med_dup = agent.generate_recommendations(
        risk_level="Low",
        document_medicines=[
            {"medicine": "Metformin", "dose": "500 mg"},
            {"medicine": "Metformin", "dose": "850 mg"},
        ],
    )
    met_cards = [c for c in out_med_dup["guidance"] if "METFORMIN" in c["rule_id"]]
    assert len(met_cards) == 2
    for mc in met_cards:
        assert "listed more than once; please check" in mc["text"]

    # 7. Chapter 17b Fix 2: Contradiction note ONLY on source "provided"
    # Case 7a: Skipped HighBP (assumed 0) + Amlodipine on document -> NO contradiction note
    out_med_skipped = agent.generate_recommendations(
        risk_level="Low",
        patient_profile={"HighBP": 0.0},
        sources={"HighBP": "assumed"},
        document_medicines=[{"medicine": "Amlodipine", "dose": "5 mg"}],
    )
    aml_skipped_card = next((c for c in out_med_skipped["guidance"] if "AMLODIPINE" in c["rule_id"]), None)
    assert aml_skipped_card is not None
    assert not any("answered No to blood pressure" in n for n in aml_skipped_card.get("notes", [])), (
        "Skipped/assumed HighBP must NEVER trigger a contradiction note"
    )

    # Case 7b: HighBP explicitly answered 0 (source "provided") + Amlodipine -> note shown
    out_med_contradiction = agent.generate_recommendations(
        risk_level="Low",
        patient_profile={"HighBP": 0.0},
        sources={"HighBP": "provided"},
        document_medicines=[{"medicine": "Amlodipine", "dose": "5 mg"}],
    )
    aml_card = next((c for c in out_med_contradiction["guidance"] if "AMLODIPINE" in c["rule_id"]), None)
    assert aml_card is not None
    assert (
        "You listed a medicine commonly used for blood pressure but answered No to blood pressure; please check your answers."
        in aml_card["text"]
    )

    # 8. Chapter 17b Fix 3: based_on and current upload vs saved record
    # (a) Lab from CURRENT upload -> "uploaded document: <lab> <value>"
    out_cur_upload = agent.generate_recommendations(
        risk_level="Low",
        document_labs=[
            {"test": "HbA1c", "value": "6.5", "unit": "%"},
            {"test": "Total Cholesterol", "value": "240", "unit": "mg/dL"},
            {"test": "Blood Pressure", "value": "140/90"},
        ],
    )
    gly_cur = next(c for c in out_cur_upload["guidance"] if c["rule_id"] == "REC_GLYCEMIC_LAB")
    assert gly_cur["based_on"] == "uploaded document: HbA1c 6.5%"
    assert "uploaded document" in gly_cur["text"]

    chol_cur = next(c for c in out_cur_upload["guidance"] if c["rule_id"] == "REC_CHOLESTEROL")
    assert chol_cur["based_on"] == "uploaded document: Total Cholesterol 240 mg/dL"

    bp_cur = next(c for c in out_cur_upload["guidance"] if c["rule_id"] == "REC_BP_HIGH")
    assert bp_cur["based_on"] == "uploaded document: Blood Pressure 140/90"

    # (b) BP & Lipid cards: when no personal trigger -> general advice
    out_general = agent.generate_recommendations(
        risk_level="Low",
        patient_profile={"HighBP": 0.0, "HighChol": 0.0},
        sources={"HighBP": "provided", "HighChol": "provided"},
        top_shap_features=["HighBP pushes toward High", "HighChol pushes toward High"],
    )
    bp_gen = next((c for c in out_general["guidance"] if c["rule_id"] == "REC_BP_HIGH"), None)
    assert bp_gen is not None
    assert bp_gen["based_on"] == "general advice (no personal data used)"
    assert "wellness: Maintain a balanced diet" in bp_gen["text"]

    chol_gen = next((c for c in out_general["guidance"] if c["rule_id"] == "REC_CHOLESTEROL"), None)
    assert chol_gen is not None
    assert chol_gen["based_on"] == "general advice (no personal data used)"
    assert "wellness: Support healthy cholesterol levels" in chol_gen["text"]

    # (c) BP & Lipid cards: when HighBP/HighChol provided/saved = yes -> claims personal data
    out_yes = agent.generate_recommendations(
        risk_level="Low",
        patient_profile={"HighBP": 1.0, "HighChol": 1.0},
        sources={"HighBP": "provided", "HighChol": "saved_record"},
    )
    bp_yes = next(c for c in out_yes["guidance"] if c["rule_id"] == "REC_BP_HIGH")
    assert bp_yes["based_on"] == "your answer: high blood pressure = yes"
    chol_yes = next(c for c in out_yes["guidance"] if c["rule_id"] == "REC_CHOLESTEROL")
    assert chol_yes["based_on"] == "saved record: high cholesterol = yes"

    # 8. Handwriting candidates never produce a medicine card
    out_hw_only = agent.generate_recommendations(
        risk_level="Low",
        document_medicines=[],
        ocr_result={"medicines": [], "handwriting_candidates": [{"name": "Metformin", "confidence": 0.95}]},
    )
    assert not any("METFORMIN" in c.get("rule_id", "") for c in out_hw_only["guidance"]), (
        "Handwriting candidates must never generate medicine cards"
    )

    # 9. Inferred label appears on medicine-inferred features in saved records
    assembled = assemble_features(
        provided_profile={"BMI": 25.0, "Age": 6.0, "GenHlth": 2.0, "Sex": 1.0},
        memory_summary={"conditions": [], "medications": ["Amlodipine 5mg"]},
    )
    assert assembled["sources"]["HighBP"] == "saved_record"
    bp_sr = next((r for r in assembled["saved_records"] if r["feature"] == "HighBP"), None)
    assert bp_sr is not None
    assert bp_sr.get("inferred_from_medicine") is True

