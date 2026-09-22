"""CART classification tree with the shared visualization schema."""

from dataclasses import dataclass
import numpy as np

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


@dataclass
class TreeNode:
    feature: int | None = None
    threshold: float | None = None
    samples: int = 0
    value: list[int] | None = None
    left: object = None
    right: object = None


@register_model("decision_tree")
class DecisionTree(BaseModel):
    task_type = "classification"

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("DecisionTree is a classifier and requires y")
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.feature_names_ = [f"feature_{index}" for index in range(X.shape[1])]
        self.root_ = self._grow(X, encoded, 0)
        return self

    def _gini(self, y):
        if not len(y): return 0.0
        counts = np.bincount(y, minlength=len(self.classes_)) / len(y)
        return float(1 - (counts ** 2).sum())

    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=len(self.classes_)).tolist()
        node = TreeNode(samples=len(y), value=counts)
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return node
        best = (self._gini(y), None, None, None)
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            thresholds = (values[:-1] + values[1:]) / 2
            for threshold in thresholds:
                mask = X[:, feature] <= threshold
                left_n, right_n = mask.sum(), len(mask) - mask.sum()
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf: continue
                score = (left_n * self._gini(y[mask]) + right_n * self._gini(y[~mask])) / len(y)
                if score < best[0]: best = (score, feature, float(threshold), mask)
        _, feature, threshold, mask = best
        if feature is None: return node
        node.feature, node.threshold = feature, threshold
        node.left, node.right = self._grow(X[mask], y[mask], depth + 1), self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X):
        if not hasattr(self, "root_"): raise RuntimeError("fit must be called before predict")
        return np.asarray([self.classes_[self._predict_one(np.asarray(row, dtype=float), self.root_)] for row in X])

    def _predict_one(self, row, node):
        while node.left is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return int(np.argmax(node.value))

    def get_params(self):
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split, "min_samples_leaf": self.min_samples_leaf}

    def get_visualization_data(self):
        def export(node):
            if node is None: return None
            return {"feature": None if node.feature is None else self.feature_names_[node.feature], "threshold": node.threshold, "samples": node.samples, "value": node.value, "left": export(node.left), "right": export(node.right)}
        return export(self.root_) if hasattr(self, "root_") else {}
