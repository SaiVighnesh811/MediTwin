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
