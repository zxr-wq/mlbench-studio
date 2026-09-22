import numpy as np
from sklearn.datasets import load_iris

import backend.models  # noqa: F401
from backend.core.factory import ModelFactory
from backend.core.registry import registered_names


def test_eight_scratch_models_are_registered():
    assert set(registered_names()) == {
        "decision_tree", "gradient_boosting", "kmeans", "knn",
        "naive_bayes", "pca", "random_forest", "svm",
    }


def test_tree_ensemble_models_train_and_predict():
    X, y = load_iris(return_X_y=True)
    for name, params in (
        ("decision_tree", {"max_depth": 4}),
        ("random_forest", {"n_estimators": 8, "max_depth": 4}),
        ("gradient_boosting", {"n_estimators": 10, "max_depth": 2}),
    ):
        model = ModelFactory.create(name, params).fit(X, y)
        assert np.mean(model.predict(X) == y) > .85


def test_knn_and_naive_bayes_match_the_shared_interface():
    X, y = load_iris(return_X_y=True)
    for name in ("knn", "naive_bayes"):
        model = ModelFactory.create(name).fit(X, y)
        assert len(model.predict(X[:5])) == 5
        assert isinstance(model.get_params(), dict)
        assert isinstance(model.get_visualization_data(), dict)
