"""Small end-to-end core test: Iris -> KNN -> unified Result."""

import backend.models  # noqa: F401
from backend.core.runner import run_experiment


def test_iris_knn_smoke_result_shape():
    result = run_experiment({
        "dataset": "iris",
        "split": {"test_size": 0.2, "random_state": 42},
        "preprocessing": ["standard_scaler"],
        "model": {"name": "knn", "implementation": "scratch", "params": {"k": 5}},
        "metrics": ["accuracy", "f1"],
    })
    assert result["model"] == "knn"
    assert result["task_type"] == "classification"
    assert result["metrics"]["accuracy"] > 0.85
    assert result["training_time"] >= 0
