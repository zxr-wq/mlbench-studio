import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import backend.models  # noqa: F401
from backend.core.factory import ModelFactory
from backend.core.runner import run_experiment


def test_linear_regression_matches_least_squares_reference():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=.2, random_state=42)
    scratch = ModelFactory.create("linear_regression").fit(X_train, y_train)
    reference = SklearnLinearRegression().fit(X_train, y_train)
    np.testing.assert_allclose(scratch.predict(X_test), reference.predict(X_test), atol=1e-8)


def test_logistic_regression_is_accurate_on_standardized_iris():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    model = ModelFactory.create("logistic_regression", {"learning_rate": .1, "max_iter": 1200, "l2": .001}).fit(scaler.transform(X_train), y_train)
    assert np.mean(model.predict(scaler.transform(X_test)) == y_test) >= .9
    assert len(model.get_visualization_data()["loss_history"]) > 1


def test_diabetes_runner_returns_regression_metrics_only():
    result = run_experiment({
        "dataset": "diabetes",
        "split": {"test_size": .2, "random_state": 42},
        "preprocessing": ["standard_scaler"],
        "model": {"name": "linear_regression", "implementation": "scratch", "params": {}},
        "metrics": ["mse", "rmse", "mae", "r2"],
    })
    assert result["task_type"] == "regression"
    assert set(result["metrics"]) == {"mse", "rmse", "mae", "r2"}
    assert result["metrics"]["r2"] > .3
