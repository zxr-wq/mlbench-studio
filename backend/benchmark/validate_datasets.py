"""Run one fast, representative acceptance experiment for every built-in dataset.

This is intentionally not a leaderboard: each row checks that loading, splitting,
preprocessing, training, metrics and visualization payload generation all work.
"""

import json

import backend.models  # noqa: F401
from backend.core.runner import run_experiment


CASES = {
    "iris": ("knn", {"k": 5}, ["accuracy", "f1"]),
    "wine": ("logistic_regression", {"max_iter": 800, "l2": 0.01}, ["accuracy", "f1"]),
    "breast_cancer": ("logistic_regression", {"max_iter": 800, "l2": 0.01}, ["accuracy", "f1", "roc_auc"]),
    "digits": ("knn", {"k": 5}, ["accuracy", "f1"]),
    "moons": ("svm", {"C": 1.0, "kernel": "rbf"}, ["accuracy", "f1"]),
    "circles": ("svm", {"C": 1.0, "kernel": "rbf"}, ["accuracy", "f1"]),
    "diabetes": ("linear_regression", {"fit_intercept": True}, ["r2", "rmse", "mae"]),
    "friedman1": ("linear_regression", {"fit_intercept": True}, ["r2", "rmse", "mae"]),
}


def validate_all(repetitions=3):
    rows = []
    for dataset, (model, params, metrics) in CASES.items():
        result = run_experiment({
            "dataset": dataset,
            "split": {"test_size": 0.2, "random_state": 42},
            "preprocessing": ["standard_scaler"],
            "model": {"name": model, "implementation": "sklearn", "params": params},
            "metrics": metrics,
            "evaluation": {"strategy": "holdout", "repetitions": repetitions},
        })
        rows.append({
            "dataset": dataset,
            "model": model,
            "metrics": result["metrics"],
            "metric_std": result["metric_std"],
            "total_runs": result["evaluation"]["total_runs"],
            "visualization_sections": sorted(
                key for key in result["visualization"]
                if key in {"classification", "regression", "clustering", "pca", "feature_contributions", "loss_curve"}
            ),
        })
    return rows


if __name__ == "__main__":
    print(json.dumps(validate_all(), ensure_ascii=False, indent=2))

