"""决策树 CART（分类）

成员 D 负责。B 提供 BaseModel 后改成:
    @register_model("decision_tree")
    class DecisionTree(BaseModel):
"""

import numpy as np


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, samples=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples


class DecisionTree:

    task_type = "classification"

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.root = None
        self.n_classes_ = 0
        self.feature_names = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_classes_ = len(np.unique(y))
        self.root = self._grow(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.root is None:
            raise RuntimeError("模型未训练")
        return np.array([self._predict_one(x, self.root) for x in X])

    def get_params(self):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def get_visualization_data(self):
        if self.root is None:
            return {}
        return self._node_to_dict(self.root)

    def _gini(self, y):
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / counts.sum()
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y):
        """找加权不纯度最小的切分点"""
        best_score = float("inf")
        best_feature, best_threshold = None, None
        n_samples, n_features = X.shape

        for j in range(n_features):
            for t in np.unique(X[:, j]):
                left = X[:, j] <= t
                if left.sum() == 0 or (~left).sum() == 0:
                    continue
                score = (
                    left.sum() * self._gini(y[left])
                    + (~left).sum() * self._gini(y[~left])
                ) / n_samples
                if score < best_score:
                    best_score = score
                    best_feature, best_threshold = j, t

        return best_feature, best_threshold

    def _grow(self, X, y, depth):
        n_samples = X.shape[0]

        if (self.max_depth is not None and depth >= self.max_depth) \
                or len(np.unique(y)) == 1 \
                or n_samples < self.min_samples_split:
            return self._make_leaf(y)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return self._make_leaf(y)

        left_mask = X[:, feature] <= threshold
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return self._make_leaf(y)

        node = TreeNode(feature=feature, threshold=float(threshold),
                        samples=n_samples, value=self._class_counts(y))
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _make_leaf(self, y):
        return TreeNode(value=self._class_counts(y), samples=len(y))

    def _class_counts(self, y):
        return np.bincount(y, minlength=self.n_classes_).tolist()

    def _predict_one(self, x, node):
        while node.left is not None or node.right is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return int(np.argmax(node.value))

    def _node_to_dict(self, node):
        if node is None:
            return None
        name = node.feature
        if self.feature_names is not None and node.feature is not None:
            name = self.feature_names[node.feature]
        return {
            "feature": name,
            "threshold": node.threshold,
            "samples": node.samples,
            "value": node.value,
            "left": self._node_to_dict(node.left),
            "right": self._node_to_dict(node.right),
        }
