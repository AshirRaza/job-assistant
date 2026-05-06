from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "checks" in body


def test_analyze_text_request():
    response = client.post(
        "/analyze",
        data={
            "cv_text": "Python FastAPI Docker AWS engineer",
            "jd_text": "Need Python FastAPI Kubernetes and AWS",
            "top_k": "4",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "match_score" in body
    assert "skill_match" in body
    assert "model_used" in body


def test_score_text_request():
    response = client.post(
        "/score",
        data={
            "cv_text": "Python FastAPI Docker AWS engineer",
            "jd_text": "Need Python FastAPI Kubernetes and AWS",
        },
    )
    assert response.status_code == 200
    assert "match_score" in response.json()


def test_analyze_bad_pdf_returns_ingestion_error():
    response = client.post(
        "/analyze",
        files={
            "cv_file": ("cv.pdf", b"not-a-real-pdf", "application/pdf"),
            "jd_file": ("jd.pdf", b"not-a-real-pdf", "application/pdf"),
        },
    )
    assert response.status_code == 400
    assert response.json().get("error") == "ingestion_error"
