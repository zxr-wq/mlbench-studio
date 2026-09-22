"""Scratch multiclass gradient boosting with shallow regression trees."""

from dataclasses import dataclass
import numpy as np

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


@dataclass
class _Node:
    feature: int | None = None
    threshold: float | None = None
    value: float = 0.0
    left: object = None
    right: object = None


class _RegressionTree:
    def __init__(self, max_depth=2, min_samples_leaf=1):
        self.max_depth, self.min_samples_leaf = int(max_depth), int(min_samples_leaf)

    def fit(self, X, residual):
        self.root_ = self._grow(np.asarray(X, dtype=float), np.asarray(residual, dtype=float), 0)
        return self

    def _grow(self, X, y, depth):
        node = _Node(value=float(y.mean()))
        if depth >= self.max_depth or len(y) <= 2 * self.min_samples_leaf: return node
        best = (float(np.var(y) * len(y)), None, None, None)
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            if len(values) > 32: values = np.quantile(values, np.linspace(.03, .97, 32))
            for threshold in values[:-1]:
                mask = X[:, feature] <= threshold
                if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf: continue
                score = float(mask.sum() * np.var(y[mask]) + (~mask).sum() * np.var(y[~mask]))
                if score < best[0]: best = (score, feature, float(threshold), mask)
        _, feature, threshold, mask = best
        if feature is None: return node
        node.feature, node.threshold = feature, threshold
        node.left, node.right = self._grow(X[mask], y[mask], depth + 1), self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X):
        def one(row):
            node = self.root_
            while node.left is not None: node = node.left if row[node.feature] <= node.threshold else node.right
            return node.value
        return np.asarray([one(row) for row in np.asarray(X, dtype=float)])


@register_model("gradient_boosting")
class GradientBoosting(BaseModel):
    task_type = "classification"

    def __init__(self, n_estimators=100, learning_rate=.1, max_depth=2, min_samples_leaf=1):
        self.n_estimators, self.learning_rate, self.max_depth, self.min_samples_leaf = int(n_estimators), float(learning_rate), int(max_depth), int(min_samples_leaf)

    @staticmethod
    def _softmax(scores):
        values = np.exp(scores - scores.max(axis=1, keepdims=True))
        return values / values.sum(axis=1, keepdims=True)

    def fit(self, X, y=None):
        if y is None: raise ValueError("GradientBoosting is a classifier and requires y")
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        targets = np.eye(len(self.classes_))[encoded]
        scores = np.zeros_like(targets, dtype=float)
        self.trees_, self.loss_history_ = [], []
        for _ in range(self.n_estimators):
            probabilities = self._softmax(scores)
            self.loss_history_.append(float(-np.mean(np.log(probabilities[np.arange(len(y)), encoded] + 1e-12))))
            stage = []
            for klass in range(len(self.classes_)):
                tree = _RegressionTree(self.max_depth, self.min_samples_leaf).fit(X, targets[:, klass] - probabilities[:, klass])
                scores[:, klass] += self.learning_rate * tree.predict(X)
                stage.append(tree)
            self.trees_.append(stage)
        self.loss_history_.append(float(-np.mean(np.log(self._softmax(scores)[np.arange(len(y)), encoded] + 1e-12))))
        return self

    def predict(self, X):
        if not hasattr(self, "trees_"): raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        scores = np.zeros((len(X), len(self.classes_)))
        for stage in self.trees_:
            for klass, tree in enumerate(stage): scores[:, klass] += self.learning_rate * tree.predict(X)
        return self.classes_[scores.argmax(axis=1)]

    def get_params(self):
        return {"n_estimators": self.n_estimators, "learning_rate": self.learning_rate, "max_depth": self.max_depth, "min_samples_leaf": self.min_samples_leaf}

    def get_visualization_data(self):
        return {"type": "gradient_boosting_loss", "loss_history": getattr(self, "loss_history_", []), "n_estimators": self.n_estimators, "learning_rate": self.learning_rate}
