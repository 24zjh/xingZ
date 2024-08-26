from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_process_text():
    response = client.post("/api/text/", json={"content": "你好，千帆"})
    assert response.status_code == 200
    assert "result" in response.json()
