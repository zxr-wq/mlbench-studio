"""Tests for repeated-seed and K-fold evaluation in the unified Runner."""

import pytest

import backend.models  # noqa: F401
from backend.core.runner import run_experiment
from backend.server.app import normalize_config


def _config(model="knn", dataset="iris"):
    return {
        "dataset": dataset,
        "split": {"test_size": 0.2, "random_state": 42},
        "preprocessing": ["standard_scaler"],
        "model": {
            "name": model,
            "implementation": "sklearn",
            "params": {"k": 5} if model == "knn" else {"k": 3, "random_state": 42},
        },
        "metrics": ["accuracy", "f1"] if model == "knn" else ["silhouette_score", "inertia"],
    }


def test_repeated_holdout_reports_mean_std_and_seed_trace():
    config = _config()
    config["evaluation"] = {"strategy": "holdout", "repetitions": 3}
    result = run_experiment(config)
    assert result["evaluation"]["total_runs"] == 3
    assert result["evaluation"]["seeds"] == [42, 43, 44]
    assert len(result["evaluation"]["per_run"]) == 3
    assert result["metrics"]["accuracy"] > 0.85
    assert result["metric_std"]["accuracy"] >= 0


def test_five_fold_evaluation_trains_every_fold():
    config = _config()
    config["evaluation"] = {"strategy": "kfold", "n_splits": 5, "repetitions": 1}
    result = run_experiment(config)
    assert result["evaluation"]["strategy"] == "kfold"
    assert result["evaluation"]["total_runs"] == 5
    assert {row["fold"] for row in result["evaluation"]["per_run"]} == {1, 2, 3, 4, 5}
    assert result["metrics"]["accuracy"] > 0.85


def test_cross_validation_rejects_unsupervised_models():
    config = _config("kmeans")
    config["evaluation"] = {"strategy": "kfold", "n_splits": 5, "repetitions": 1}
    with pytest.raises(ValueError, match="仅用于分类和回归"):
        run_experiment(config)


def test_server_preserves_evaluation_settings():
    config = _config()
    config["evaluation"] = {"strategy": "kfold", "n_splits": 5, "repetitions": 3}
    normalized = normalize_config(config)
    assert normalized["evaluation"] == {"strategy": "kfold", "repetitions": 3, "n_splits": 5}

