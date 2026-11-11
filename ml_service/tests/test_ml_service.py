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
            "features": json.dumps({"BMI": 26.0, "Age": 5.0})
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
                "features": json.dumps({"BMI": 25.0, "Age": 6.0}),
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
