from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from app.ml.base import Classifier


class SklearnKNNClassifier(Classifier):
    """Adapter that keeps the public parameter names identical to Scratch KNN."""

    def __init__(self, k: int = 5, distance: str = "euclidean", weights: str = "distance") -> None:
        super().__init__()
        self.k = int(k)
        self.distance = str(distance)
        self.weights = str(weights)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SklearnKNNClassifier":
        metric = {"euclidean": "euclidean", "manhattan": "manhattan"}.get(self.distance)
        if metric is None:
            raise ValueError("distance must be 'euclidean' or 'manhattan'")
        self.model_ = KNeighborsClassifier(n_neighbors=self.k, metric=metric, weights=self.weights)
        self.model_.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=int))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict")
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict")
        return np.asarray(self.model_.predict_proba(np.asarray(x, dtype=float)), dtype=float)


class SklearnGaussianNBClassifier(Classifier):
    def __init__(self, var_smoothing: float = 1e-9) -> None:
        super().__init__()
        self.var_smoothing = float(var_smoothing)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SklearnGaussianNBClassifier":
        self.model_ = GaussianNB(var_smoothing=self.var_smoothing)
        self.model_.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=int))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict")
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict")
        return np.asarray(self.model_.predict_proba(np.asarray(x, dtype=float)), dtype=float)
