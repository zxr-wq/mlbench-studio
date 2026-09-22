import asyncio

import pytest

from backend.server.app import execute, normalize_config, records


def _iris_knn_config():
    return {
        "dataset": "iris",
        "split": {"test_size": .2, "random_state": 42},
        "preprocessing": ["standard_scaler"],
        "model": {"name": "knn", "implementation": "scratch", "params": {"k": 5}},
        "metrics": ["accuracy", "f1"],
    }


def test_server_rejects_unknown_model_with_a_clear_error():
    with pytest.raises(KeyError, match="未注册"):
        normalize_config({"dataset": "iris", "model": {"name": "not_a_model"}})


def test_server_execution_path_runs_iris_knn_and_preserves_result_record():
    experiment_id = "test-iris-knn"
    records[experiment_id] = {
        "id": experiment_id, "status": "queued", "progress": 0,
        "stage": "queued", "config": _iris_knn_config(), "result": None, "error": None,
    }
    asyncio.run(execute(experiment_id, normalize_config(_iris_knn_config())))
    record = records.pop(experiment_id)
    assert record["status"] == "completed", record["error"]
    assert record["result"]["metrics"]["accuracy"] > .85
