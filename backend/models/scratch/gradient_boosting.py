"""梯度提升（Gradient Boosting，分类）— Scratch 自实现

负责人：成员 D

核心思路（接力补错）：
1. 先给每个类别一个初始分数（这里是 0）
2. 把分数转成概率（softmax），看看离真实答案差多少 → 这个差就是"残差"
3. 训练一棵回归树，专门去拟合这个残差（也就是补上次的错）
4. 把这棵树的预测乘一个小的 learning_rate 加到总分上
5. 重复 n_estimators 轮，每一轮记录一次 loss，交给前端画 Loss 曲线
"""

import numpy as np


# ----------------------------------------------------------------------
# 内部用的回归树：梯度提升的弱学习器要拟合连续值（残差），
# 所以不能用分类树（Gini + 投票），必须用"均方误差 + 均值"的回归树。
# ----------------------------------------------------------------------
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
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1,
                 max_candidates=32):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_candidates = max_candidates
        self.root = None

    def fit(self, X, r, hess=None):
        """hess = 每个样本的二阶导（这里是 p(1-p)），给了就按牛顿步长算叶子值"""
        X = np.asarray(X, dtype=float)
        r = np.asarray(r, dtype=float)
        self.root = self._grow(X, r, hess, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._one(x, self.root) for x in X])

    def _leaf_value(self, r, hess):
        # 普通情况：叶子取残差均值（相当于固定步长）
        # 给了 hess：牛顿步长 = 残差之和 / 二阶导之和，收敛更快更准
        if hess is None:
            return float(np.mean(r))
        denom = float(np.sum(hess))
        if denom < 1e-12:
            return 0.0
        return float(np.sum(r) / denom)

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
        left_hess = None if hess is None else hess[left_mask]
        right_hess = None if hess is None else hess[~left_mask]
        node.left = self._grow(X[left_mask], r[left_mask], left_hess, depth + 1)
        node.right = self._grow(X[~left_mask], r[~left_mask], right_hess, depth + 1)
        return node

    def _best_split(self, X, r):
        best_score = float("inf")
        best_feature, best_threshold = None, None
        n_samples, n_features = X.shape

        for j in range(n_features):
            col = np.unique(X[:, j])
            # 取值太多时按分位数抽样，避免训练过慢（sklearn 也这么做）
            if col.size > self.max_candidates:
                col = np.quantile(X[:, j], np.linspace(0, 1, self.max_candidates))
            for t in col:
                left = X[:, j] <= t
                if left.sum() == 0 or (~left).sum() == 0:
                    continue
                # 均方误差：叶子里的值取均值时，误差 = 方差 × 样本数
                score = (
                    np.var(r[left]) * left.sum()
                    + np.var(r[~left]) * (~left).sum()
                )
                if score < best_score:
                    best_score = score
                    best_feature, best_threshold = j, t

        return best_feature, best_threshold

    def _one(self, x, node):
        while node.left is not None or node.right is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value


# ----------------------------------------------------------------------
# 梯度提升主体
# ----------------------------------------------------------------------
class GradientBoosting:
    # TODO（B 交付框架后）：@register_model("gradient_boosting") + class GradientBoosting(BaseModel)

    task_type = "classification"

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.classes_ = None
        self.trees_ = []          # 每一轮有 K 棵树（每个类别一棵）
        self.init_scores_ = None
        self.loss_history_ = []

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples = X.shape[0]

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # 真实标签转成 one-hot：属于第 k 类就是 1，其余 0
        Y = np.zeros((n_samples, n_classes))
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1.0

        # 初始分数全部为 0（等价于一开始认为每个类概率相同）
        F = np.zeros((n_samples, n_classes))
        self.init_scores_ = F.copy()
        self.trees_ = []
        self.loss_history_ = []

        for _ in range(self.n_estimators):
            P = self._softmax(F)
            self.loss_history_.append(self._log_loss(Y, P))

            # 残差 = 真实 - 预测（这就是"还差多少"）
            R = Y - P

            trees = []
            for k in range(n_classes):
                # hess = p(1-p)，用于牛顿步长，让叶子值更准
                hess = P[:, k] * (1.0 - P[:, k])
                tree = _RegressionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                ).fit(X, R[:, k], hess)
                trees.append(tree)
                # 按 learning_rate 把这一轮学到的东西加上去
                F[:, k] += self.learning_rate * tree.predict(X)

            self.trees_.append(trees)

        # 记录最后一轮的 loss
        self.loss_history_.append(self._log_loss(Y, self._softmax(F)))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if not self.trees_:
            raise RuntimeError("模型还没训练，请先调用 fit()")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        F = np.zeros((n_samples, n_classes))

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
        """规范第 13 条：梯度提升给前端返回 loss_history 画 Loss 曲线"""
        return {
            "loss_history": self.loss_history_,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(F):
        F = F - F.max(axis=1, keepdims=True)   # 防止指数溢出
        E = np.exp(F)
        return E / E.sum(axis=1, keepdims=True)

    @staticmethod
    def _log_loss(Y, P):
        return float(-np.mean(np.log(np.sum(Y * P, axis=1) + 1e-15)))
