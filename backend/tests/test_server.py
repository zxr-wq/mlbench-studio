import time

from fastapi.testclient import TestClient

from backend.server.app import app


client = TestClient(app)


def test_server_rejects_unknown_model_with_a_clear_400():
    response = client.post("/api/experiments", json={"dataset": "iris", "model": {"name": "not_a_model"}})
    assert response.status_code == 400
    assert "未注册" in response.json()["detail"]


def test_iris_knn_http_and_websocket_flow():
    created = client.post("/api/experiments", json={
        "dataset": "iris",
        "split": {"test_size": .2, "random_state": 42},
        "preprocessing": ["standard_scaler"],
        "model": {"name": "knn", "implementation": "scratch", "params": {"k": 5}},
        "metrics": ["accuracy", "f1"],
    })
    assert created.status_code == 202
    experiment_id = created.json()["id"]
    with client.websocket_connect(f"/api/ws/experiments/{experiment_id}") as socket:
        for _ in range(8):
            payload = socket.receive_json()["experiment"]
            if payload["status"] in {"completed", "failed"}:
                break
    if payload["status"] != "completed":
        for _ in range(10):
            payload = client.get(f"/api/experiments/{experiment_id}").json()
            if payload["status"] in {"completed", "failed"}:
                break
            time.sleep(.02)
    assert payload["status"] == "completed", payload.get("error")
    assert payload["result"]["metrics"]["accuracy"] > .85
