from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "features": 17, "facies": 9}


def test_demo_list_and_prediction():
    wells = [w["id"] for w in client.get("/api/demo").json()]
    assert wells == ["synthetic_a", "synthetic_b"]
    result = client.get("/api/demo/synthetic_a").json()
    assert result["well"] == "SYNTHETIC A" and result["samples"] == 460


def test_unknown_demo_well():
    assert client.get("/api/demo/nowhere").status_code == 404


def test_upload_and_error_message():
    csv = client.get("/api/demo/synthetic_b/csv").content
    ok = client.post("/api/predict", files={"file": ("synthetic_b.csv", csv, "text/csv")})
    assert ok.status_code == 200 and ok.json()["scores"] is not None
    broken = csv.replace(b",PE,", b",XX,", 1)
    bad = client.post("/api/predict", files={"file": ("bad.csv", broken, "text/csv")})
    assert bad.status_code == 422 and "PE" in bad.json()["detail"]


def test_page_is_served():
    response = client.get("/")
    assert response.status_code == 200 and "Facies" in response.text
