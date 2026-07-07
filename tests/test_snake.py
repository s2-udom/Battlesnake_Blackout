from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_index():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["apiversion"] == "1"

def test_move_returns_valid_direction():
    r = client.post("/move", json={"you": {}, "board": {}, "turn": 0, "game": {}})
    assert r.status_code == 200
    assert r.json()["move"] in ["up", "down", "left", "right"]
