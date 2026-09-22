"""Scratch random forest built from this project's DecisionTree."""

import numpy as np

from backend.core.base_model import BaseModel
from backend.core.registry import register_model
from backend.models.scratch.decision_tree import DecisionTree


@register_model("random_forest")
class RandomForest(BaseModel):
    task_type = "classification"

    def __init__(self, n_estimators=100, max_depth=None, max_features=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.n_estimators = int(n_estimators)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.max_features = max_features
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)

    def fit(self, X, y=None):
        if y is None: raise ValueError("RandomForest is a classifier and requires y")
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        feature_count = X.shape[1]
        if self.max_features is None: selected_count = max(1, int(np.sqrt(feature_count)))
        elif isinstance(self.max_features, float): selected_count = max(1, min(feature_count, int(np.ceil(feature_count * self.max_features))))
        else: selected_count = max(1, min(feature_count, int(self.max_features)))
        self.trees_, self.feature_subsets_ = [], []
        for _ in range(self.n_estimators):
            rows = rng.integers(0, len(X), len(X))
            features = np.sort(rng.choice(feature_count, selected_count, replace=False))
            tree = DecisionTree(self.max_depth, self.min_samples_split, self.min_samples_leaf).fit(X[rows][:, features], y[rows])
            self.trees_.append(tree)
            self.feature_subsets_.append(features)
        return self

    def predict(self, X):
        if not hasattr(self, "trees_"): raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        votes = np.asarray([tree.predict(X[:, features]) for tree, features in zip(self.trees_, self.feature_subsets_)])
        return np.asarray([self.classes_[np.argmax([(column == label).sum() for label in self.classes_])] for column in votes.T])

    def get_params(self):
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth, "max_features": self.max_features, "min_samples_split": self.min_samples_split, "min_samples_leaf": self.min_samples_leaf, "random_state": self.random_state}

    def get_visualization_data(self):
        if not hasattr(self, "trees_"): return {}
        def depth(node): return 0 if node is None else 1 + max(depth(node.left), depth(node.right))
        return {"type": "random_forest", "n_estimators": len(self.trees_), "tree_depths": [depth(tree.root_) for tree in self.trees_]}
