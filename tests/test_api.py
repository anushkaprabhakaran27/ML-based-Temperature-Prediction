from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict_schema():
    payload = {
        "day_of_year": 150,
        "month": 6,
        "humidity": 70,
        "wind_speed": 15.0,
        "pressure": 1015.0,
        "cloud_cover": 50,
        "previous_temp": 28.0
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "predicted_temperature" in r.json()
