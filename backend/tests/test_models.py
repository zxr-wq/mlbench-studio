from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.ml.models import (
    DecisionTreeClassifier,
    GaussianNBClassifier,
    KNNClassifier,
    LinearSVMClassifier,
    SoftmaxRegression,
)
from app.ml.sklearn_models import SklearnGaussianNBClassifier, SklearnKNNClassifier


def _iris_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = load_iris()
    rng = np.random.default_rng(7)
    train_parts = []
    test_parts = []
    for label in np.unique(dataset.target):
        indices = np.where(dataset.target == label)[0]
        rng.shuffle(indices)
        test_parts.append(indices[:15])
        train_parts.append(indices[15:])
    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)
    mean = dataset.data[train_indices].mean(axis=0)
    std = dataset.data[train_indices].std(axis=0)
    return (
        (dataset.data[train_indices] - mean) / std,
        dataset.target[train_indices],
        (dataset.data[test_indices] - mean) / std,
        dataset.target[test_indices],
    )


def test_all_scratch_classifiers_reach_useful_iris_accuracy() -> None:
    x_train, y_train, x_test, y_test = _iris_split()
    models = [
        KNNClassifier(),
        GaussianNBClassifier(),
        SoftmaxRegression(epochs=220),
        LinearSVMClassifier(epochs=220),
        DecisionTreeClassifier(max_depth=5),
    ]
    scores = []
    for model in models:
        model.fit(x_train, y_train)
        scores.append(float(np.mean(model.predict(x_test) == y_test)))
    assert min(scores) >= 0.80, scores


@pytest.mark.parametrize("loader", [load_iris, load_wine, load_breast_cancer])
@pytest.mark.parametrize(
    ("scratch_factory", "sklearn_factory"),
    [
        (lambda: KNNClassifier(k=5, distance="euclidean", weights="distance"), lambda: SklearnKNNClassifier(k=5)),
        (lambda: GaussianNBClassifier(), lambda: SklearnGaussianNBClassifier()),
    ],
)
def test_a_models_agree_with_sklearn(loader, scratch_factory, sklearn_factory) -> None:
    dataset = loader()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    scratch = scratch_factory().fit(x_train, y_train)
    reference = sklearn_factory().fit(x_train, y_train)
    scratch_prediction = scratch.predict(x_test)
    reference_prediction = reference.predict(x_test)
    scratch_accuracy = float(np.mean(scratch_prediction == y_test))
    reference_accuracy = float(np.mean(reference_prediction == y_test))
    agreement = float(np.mean(scratch_prediction == reference_prediction))
    assert abs(scratch_accuracy - reference_accuracy) <= 0.05
    assert agreement >= 0.90


def test_knn_supports_manhattan_uniform_and_visualization() -> None:
    x_train, y_train, x_test, _ = _iris_split()
    model = KNNClassifier(k=3, distance="manhattan", weights="uniform").fit(x_train, y_train)
    probabilities = model.predict_proba(x_test[:2])
    prediction = model.predict(x_test[:2])
    visualization = model.get_visualization_data()
    assert probabilities.shape == (2, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert prediction.shape == (2,)
    assert visualization["type"] == "knn_neighbors"
    assert len(visualization["neighbors"]) == 3


def test_naive_bayes_handles_zero_variance_and_returns_probabilities() -> None:
    x = np.array([[1.0, 0.0], [1.0, 0.2], [1.0, 1.0], [1.0, 1.2]])
    y = np.array([0, 0, 1, 1])
    model = GaussianNBClassifier().fit(x, y)
    probabilities = model.predict_proba(x)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.get_visualization_data()["type"] == "naive_bayes_stats"


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": 999}, {"distance": "cosine"}, {"weights": "bad"}])
def test_knn_rejects_invalid_configuration(kwargs) -> None:
    x_train, y_train, _, _ = _iris_split()
    with pytest.raises(ValueError):
        KNNClassifier(**kwargs).fit(x_train, y_train)


@pytest.mark.parametrize("model", [KNNClassifier(), GaussianNBClassifier()])
def test_a_models_require_fit_before_predict(model) -> None:
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 4)))
