"""随机森林（分类）— Scratch 自实现

负责人：成员 D
严格基于同目录下自己实现的 DecisionTree，不调用 sklearn 的树。

核心三步：
1. Bootstrap：有放回抽样，给每棵树一份略有不同的训练数据
2. 随机特征：每棵树（或每次分叉）只看随机抽的一部分特征
3. 投票：所有树各自预测，少数服从多数
"""

import numpy as np

from models.scratch.decision_tree import DecisionTree, TreeNode


class RandomForest:
    # TODO（B 交付框架后）：@register_model("random_forest") + class RandomForest(BaseModel)

    task_type = "classification"

    def __init__(self, n_estimators=100, max_depth=None, max_features=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features      # None 时自动取 sqrt(特征数)
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
            # 1. Bootstrap：有放回抽样，抽同样多个
            idx = rng.integers(0, n_samples, size=n_samples)
            # 2. 随机特征子集
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
            raise RuntimeError("模型还没训练，请先调用 fit()")

        # 每棵树对全部样本给出预测，形状 (树数, 样本数)
        all_preds = np.array([
            tree.predict(X[:, feats])
            for tree, feats in zip(self.trees, self.feature_subsets)
        ])

        # 3. 多数投票：逐列统计哪个类别票最多
        out = np.empty(X.shape[0], dtype=int)
        for j in range(X.shape[0]):
            counts = np.bincount(all_preds[:, j], minlength=self.n_classes_)
            out[j] = int(np.argmax(counts))
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
        """返回树数量、每棵树的深度，以及特征重要性（可用于前端柱状图）"""
        if not self.trees:
            return {}
        return {
            "n_estimators": len(self.trees),
            "tree_depths": [self._depth(t.root) for t in self.trees],
            "feature_importances": self._feature_importances(),
            "feature_names": self.feature_names,
        }

    # ------------------------------------------------------------------
    def _depth(self, node):
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def _feature_importances(self):
        """用"每个特征被用作切分点时带来了多少纯度提升"累加得到重要性"""
        n_features = max(int(np.max(feats)) for feats in self.feature_subsets) + 1 \
            if self.feature_subsets else 0
        importances = np.zeros(n_features)

        for tree, feats in zip(self.trees, self.feature_subsets):
            self._walk(tree.root, feats, importances)

        total = importances.sum()
        if total > 0:
            importances = importances / total
        return importances.tolist()

    def _walk(self, node, feats, importances):
        if node is None or node.left is None or node.right is None:
            return
        real_feature = feats[node.feature]
        # 该节点上的样本数作为权重，越靠上层权重越大
        importances[real_feature] += node.samples
        self._walk(node.left, feats, importances)
        self._walk(node.right, feats, importances)
