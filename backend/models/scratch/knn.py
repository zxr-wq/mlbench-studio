"""Scratch KNN classifier registered in the shared MLBench model registry."""

import numpy as np

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


@register_model("knn")
class KNN(BaseModel):
    task_type = "classification"

    def __init__(self, k=5, distance="euclidean", weights="uniform"):
        self.k = int(k)
        self.distance = distance
        self.weights = weights

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("KNN is a classifier and requires y")
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        if self.X_train_.ndim != 2 or len(self.X_train_) != len(self.y_train_):
            raise ValueError("X must be 2D and contain one row per y value")
        if not 1 <= self.k <= len(self.X_train_):
            raise ValueError("k must be between 1 and the number of training samples")
        if self.distance not in {"euclidean", "manhattan"}:
            raise ValueError("distance must be euclidean or manhattan")
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be uniform or distance")
        self.classes_ = np.unique(self.y_train_)
        return self

    def _neighbors(self, X):
        if not hasattr(self, "X_train_"):
            raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        diff = X[:, None, :] - self.X_train_[None, :, :]
        distances = np.abs(diff).sum(axis=2) if self.distance == "manhattan" else np.sqrt((diff ** 2).sum(axis=2))
        indices = np.argpartition(distances, self.k - 1, axis=1)[:, : self.k]
        order = np.take_along_axis(distances, indices, axis=1).argsort(axis=1)
        return np.take_along_axis(indices, order, axis=1), np.take_along_axis(distances, indices, axis=1)

    def predict(self, X):
        indices, distances = self._neighbors(X)
        predictions = []
        for row, row_indices in enumerate(indices):
            labels = self.y_train_[row_indices]
            weights = np.ones(self.k) if self.weights == "uniform" else 1 / np.maximum(distances[row], 1e-12)
            scores = [weights[labels == label].sum() for label in self.classes_]
            predictions.append(self.classes_[int(np.argmax(scores))])
        self.last_neighbor_indices_ = indices
        self.last_neighbor_distances_ = distances
        return np.asarray(predictions)

    def predict_proba(self, X):
        indices, distances = self._neighbors(X)
        probabilities = np.zeros((len(indices), len(self.classes_)))
        for row, row_indices in enumerate(indices):
            labels = self.y_train_[row_indices]
            weights = np.ones(self.k) if self.weights == "uniform" else 1 / np.maximum(distances[row], 1e-12)
            for column, label in enumerate(self.classes_):
                probabilities[row, column] = weights[labels == label].sum()
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def get_params(self):
        return {"k": self.k, "distance": self.distance, "weights": self.weights}

    def get_visualization_data(self):
        if not hasattr(self, "last_neighbor_indices_") or not len(self.last_neighbor_indices_):
            return {}
        return {
            "type": "knn_neighbors",
            "neighbor_indices": self.last_neighbor_indices_[0].tolist(),
            "neighbor_distances": self.last_neighbor_distances_[0].tolist(),
        }
