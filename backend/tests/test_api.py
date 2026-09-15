from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_catalog_endpoints() -> None:
    datasets = client.get("/api/datasets")
    algorithms = client.get("/api/algorithms")
    assert datasets.status_code == 200
    assert algorithms.status_code == 200
    assert len(datasets.json()) == 4
    assert len(algorithms.json()) == 5


def test_experiment_completes() -> None:
    with client:
        response = client.post(
            "/api/experiments",
            json={
                "dataset_id": "iris",
                "model_ids": ["naive_bayes", "knn"],
                "implementations": ["scratch", "sklearn"],
                "split_method": "stratified",
                "test_size": 0.2,
                "seed": 11,
            },
        )
        assert response.status_code == 202
        experiment_id = response.json()["id"]
        import time

        for _ in range(100):
            record = client.get(f"/api/experiments/{experiment_id}").json()
            if record["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert record["status"] == "completed", record
        assert len(record["results"]) == 4
        assert {item["implementation"] for item in record["results"]} == {"scratch", "sklearn"}
        assert all("training_time" in item and "inference_time" in item for item in record["results"])
        assert all("comparison" in item for item in record["results"])
        by_id = {item["id"]: item for item in record["results"]}
        assert by_id["knn:scratch"]["visualization"]["type"] == "knn_neighbors"
        assert by_id["naive_bayes:scratch"]["visualization"]["type"] == "naive_bayes_stats"
        assert by_id["knn:scratch"]["comparison"]["prediction_agreement"] >= 0.9
