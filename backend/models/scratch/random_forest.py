"""随机森林（分类）

基于同目录自实现的 DecisionTree，未使用 sklearn 的树。
成员 D 负责。
"""

import numpy as np

from models.scratch.decision_tree import DecisionTree


class RandomForest:

    task_type = "classification"

    def __init__(self, n_estimators=100, max_depth=None, max_features=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.trees = []
        self.feature_subsets = []
        self.n_classes_ = 0
        self.feature_names = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.n_classes_ = len(np.unique(y))

        rng = np.random.default_rng(self.random_state)
        max_features = self.max_features or max(1, int(np.sqrt(n_features)))

        self.trees = []
        self.feature_subsets = []

        for _ in range(self.n_estimators):
            idx = rng.integers(0, n_samples, size=n_samples)
            feats = rng.choice(n_features, size=max_features, replace=False)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X[idx][:, feats], y[idx])

            self.trees.append(tree)
            self.feature_subsets.append(feats)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if not self.trees:
            raise RuntimeError("模型未训练")

        all_preds = np.array([
            tree.predict(X[:, feats])
            for tree, feats in zip(self.trees, self.feature_subsets)
        ])

        out = np.empty(X.shape[0], dtype=int)
        for j in range(X.shape[0]):
            out[j] = np.argmax(np.bincount(all_preds[:, j], minlength=self.n_classes_))
        return out

    def get_params(self):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }

    def get_visualization_data(self):
        if not self.trees:
            return {}
        return {
            "n_estimators": len(self.trees),
            "tree_depths": [self._depth(t.root) for t in self.trees],
            "feature_importances": self._feature_importances(),
            "feature_names": self.feature_names,
        }

    def _depth(self, node):
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def _feature_importances(self):
        n_features = max(int(np.max(feats)) for feats in self.feature_subsets) + 1 \
            if self.feature_subsets else 0
        importances = np.zeros(n_features)

        for tree, feats in zip(self.trees, self.feature_subsets):
            self._walk(tree.root, feats, importances)

        if importances.sum() > 0:
            importances = importances / importances.sum()
        return importances.tolist()

    def _walk(self, node, feats, importances):
        if node is None or node.left is None or node.right is None:
            return
        importances[feats[node.feature]] += node.samples
        self._walk(node.left, feats, importances)
        self._walk(node.right, feats, importances)
