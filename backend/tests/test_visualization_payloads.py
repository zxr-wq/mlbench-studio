"""Regression tests for chart-ready, JSON-safe experiment results."""

import json

import backend.models  # noqa: F401
from backend.core.runner import run_experiment


def run(name, dataset, params=None):
    return run_experiment({
        "dataset": dataset,
        "split": {"test_size": .2, "random_state": 42},
        "preprocessing": ["standard_scaler"],
        "model": {"name": name, "implementation": "scratch", "params": params or {}},
        "metrics": [],
    })


def test_classification_payload_contains_real_evaluation_charts():
    result = run("knn", "iris", {"k": 5})
    charts = result["visualization"]["classification"]
    assert len(charts["confusion_matrix"]) == 3
    assert len(charts["points"]) == 30
    json.dumps(result)


def test_decision_tree_payload_contains_dataset_labelled_structure():
    result = run("decision_tree", "iris", {"max_depth": 3})
    tree = result["visualization"]["tree_structure"]
    assert tree["samples"] == 120
    assert tree["feature"] in result["visualization"]["feature_names"]
    assert tree["left"] is not None and tree["right"] is not None


def test_regression_payload_contains_fit_and_residual_charts():
    result = run("linear_regression", "diabetes")
    charts = result["visualization"]["regression"]
    assert charts["pairs"]
    assert charts["residual_histogram"]["counts"]
    assert result["visualization"]["feature_contributions"]["items"]
    json.dumps(result)


def test_unsupervised_payloads_are_task_appropriate():
    cluster = run("kmeans", "iris", {"k": 3, "n_init": 2})
    pca = run("pca", "iris", {"n_components": 2})
    assert len(cluster["visualization"]["clustering"]["points"]) == 120
    assert len(cluster["visualization"]["clustering"]["centroid_history"]) >= 2
    assert len(cluster["visualization"]["clustering"]["centroid_history"][0]) == 3
    assert len(pca["visualization"]["pca"]["points"]) == 30
    assert len(pca["metrics"]["explained_variance_ratio"]) == 2
    json.dumps(cluster)
    json.dumps(pca)


def test_binary_probability_model_returns_roc_auc_and_curve():
    result = run_experiment({
        "dataset": "moons",
        "split": {"test_size": .2, "random_state": 42},
        "preprocessing": ["standard_scaler"],
        "model": {"name": "logistic_regression", "implementation": "scratch", "params": {"max_iter": 300}},
        "metrics": ["accuracy", "roc_auc"],
    })
    assert .5 <= result["metrics"]["roc_auc"] <= 1
    assert result["visualization"]["classification"]["roc_curve"]["points"]
