"""梯度提升（分类）

用回归树逐轮拟合残差，按 learning_rate 累加以纠正上一轮的错误。
成员 D 负责。
"""

import numpy as np


class _RegTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=0.0, samples=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples


class _RegressionTree:
    """拟合连续值（残差）的弱学习器"""

    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1,
                 max_candidates=32):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_candidates = max_candidates
        self.root = None

    def fit(self, X, r, hess=None):
        """hess 为二阶导，传入时叶子值用牛顿步长"""
        X = np.asarray(X, dtype=float)
        r = np.asarray(r, dtype=float)
        self.root = self._grow(X, r, hess, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._one(x, self.root) for x in X])

    def _leaf_value(self, r, hess):
        if hess is None:
            return float(np.mean(r))
        denom = float(np.sum(hess))
        return float(np.sum(r) / denom) if denom > 1e-12 else 0.0

    def _grow(self, X, r, hess, depth):
        n_samples = X.shape[0]

        if (self.max_depth is not None and depth >= self.max_depth) \
                or n_samples < self.min_samples_split:
            return _RegTreeNode(value=self._leaf_value(r, hess), samples=n_samples)

        feature, threshold = self._best_split(X, r)
        if feature is None:
            return _RegTreeNode(value=self._leaf_value(r, hess), samples=n_samples)

        left_mask = X[:, feature] <= threshold
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return _RegTreeNode(value=self._leaf_value(r, hess), samples=n_samples)

        node = _RegTreeNode(feature=feature, threshold=float(threshold),
                            value=self._leaf_value(r, hess), samples=n_samples)
        node.left = self._grow(X[left_mask], r[left_mask],
                               None if hess is None else hess[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], r[~left_mask],
                                None if hess is None else hess[~left_mask], depth + 1)
        return node

    def _best_split(self, X, r):
        best_score = float("inf")
        best_feature, best_threshold = None, None

        for j in range(X.shape[1]):
            col = np.unique(X[:, j])
            if col.size > self.max_candidates:
                col = np.quantile(X[:, j], np.linspace(0, 1, self.max_candidates))
            for t in col:
                left = X[:, j] <= t
                if left.sum() == 0 or (~left).sum() == 0:
                    continue
                score = np.var(r[left]) * left.sum() + np.var(r[~left]) * (~left).sum()
                if score < best_score:
                    best_score = score
                    best_feature, best_threshold = j, t

        return best_feature, best_threshold

    def _one(self, x, node):
        while node.left is not None or node.right is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value


class GradientBoosting:

    task_type = "classification"

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.classes_ = None
        self.trees_ = []
        self.loss_history_ = []

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples = X.shape[0]

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        Y = np.zeros((n_samples, n_classes))
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1.0

        F = np.zeros((n_samples, n_classes))
        self.trees_ = []
        self.loss_history_ = []

        for _ in range(self.n_estimators):
            P = self._softmax(F)
            self.loss_history_.append(self._log_loss(Y, P))

            R = Y - P
            trees = []
            for k in range(n_classes):
                tree = _RegressionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                ).fit(X, R[:, k], P[:, k] * (1.0 - P[:, k]))
                trees.append(tree)
                F[:, k] += self.learning_rate * tree.predict(X)

            self.trees_.append(trees)

        self.loss_history_.append(self._log_loss(Y, self._softmax(F)))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if not self.trees_:
            raise RuntimeError("模型未训练")

        F = np.zeros((X.shape[0], len(self.classes_)))
        for trees in self.trees_:
            for k, tree in enumerate(trees):
                F[:, k] += self.learning_rate * tree.predict(X)

        return self.classes_[np.argmax(F, axis=1)]

    def get_params(self):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def get_visualization_data(self):
        return {
            "loss_history": self.loss_history_,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
        }

    @staticmethod
    def _softmax(F):
        E = np.exp(F - F.max(axis=1, keepdims=True))
        return E / E.sum(axis=1, keepdims=True)

    @staticmethod
    def _log_loss(Y, P):
        return float(-np.mean(np.log(np.sum(Y * P, axis=1) + 1e-15)))
