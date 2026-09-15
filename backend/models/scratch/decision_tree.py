"""决策树（CART 分类）— Scratch 自实现骨架

负责人：成员 D
遵循：MLBench Studio 统一接口规范（2026-09-15 版）

重要：
1. 等 B 提供 BaseModel / register_model 之后，只需改成：
       @register_model("decision_tree")
       class DecisionTree(BaseModel):
   算法主体不需要改动。
2. 你现在只需要填两个 TODO：_gini() 和 _best_split()
   这两个就是决策树算法真正的核心，其余的"管道"我都写好了。
"""

import numpy as np


class TreeNode:
    """树的节点。字段与前端可视化规范第 14 条一一对应。"""

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, samples=0):
        self.feature = feature      # 用第几个特征来问问题（内部节点）或 None（叶子）
        self.threshold = threshold  # 切分阈值
        self.left = left            # 左子树：满足 X[:, feature] <= threshold
        self.right = right          # 右子树
        self.value = value          # 该节点上每类的样本数，如 [34, 33, 33]
        self.samples = samples      # 该节点总样本数


class DecisionTree:
    # TODO（B 交付框架后）：改成 @register_model("decision_tree") + class DecisionTree(BaseModel)

    task_type = "classification"    # 规范第 3 条：必须标明任务类型

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        # 规范第 7 条：前端传来的参数就是这三个
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.root = None
        self.n_classes_ = 0
        self.feature_names = None

    # ------------------------------------------------------------------
    # 规范要求的四个方法：fit / predict / get_params / get_visualization_data
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """训练：从根节点开始递归长出一棵完整的树"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.n_classes_ = int(np.bincount(y).size) if y.dtype.kind in "iu" else len(np.unique(y))
        self.root = self._grow(X, y, depth=0)
        return self

    def predict(self, X):
        """预测：每个样本从根走到底，落到哪个叶子就取叶子的多数类"""
        X = np.asarray(X, dtype=float)
        if self.root is None:
            raise RuntimeError("模型还没训练，请先调用 fit()")
        return np.array([self._predict_one(x, self.root) for x in X])

    def get_params(self):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def get_visualization_data(self):
        """规范第 13、14 条：把树结构导成前端能直接画图的 JSON"""
        if self.root is None:
            return {}
        return self._node_to_dict(self.root)

    # ------------------------------------------------------------------
    # 下面是你需要动手填的两处核心
    # ------------------------------------------------------------------
    def _gini(self, y):
        """TODO 1：计算一堆样本的"不纯度"（越乱数值越大，全同类时为 0）

        思路（就三步）：
            counts = np.bincount(y)          # 每个类别各有多少个
            p = counts / counts.sum()        # 每个类别的占比
            return 1.0 - np.sum(p ** 2)      # 基尼 = 1 - 各占比的平方和

        直观理解：全都是同一类 → p=[1.0] → 1-1 = 0（最纯）
                  两类各一半   → p=[0.5,0.5] → 1-0.5 = 0.5（最乱）
        """
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / counts.sum()
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y):
        """TODO 2：找出"分完之后最干净"的那个特征和切分点

        返回：(feature_index, threshold)，找不到有效切分时返回 (None, None)

        思路：
            best_gain_score = 无穷大；best_feature, best_threshold = None, None
            for j in range(特征数):                      # 逐个特征试
                for t in np.unique(X[:, j]):            # 该列出现过的取值都当候选阈值
                    left  = X[:, j] <= t                # 左：小于等于
                    right = ~left                       # 右：其余
                    如果左或右为空就跳过
                    score = (左边个数 * self._gini(y[left]) + 右边个数 * self._gini(y[right])) / 总个数
                    if score < best_gain_score:         # 加权不纯度越小越好
                        记录 best_feature = j, best_threshold = t
            return best_feature, best_threshold
        """
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

    # ------------------------------------------------------------------
    # 下面的"管道代码"已经写好，你读懂即可，不用改
    # ------------------------------------------------------------------
    def _grow(self, X, y, depth):
        """递归建树：先判断是否该停下，不停就切一刀再对左右两半继续"""
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))

        # 停止条件：够深了 / 已经纯了 / 样本太少
        if (self.max_depth is not None and depth >= self.max_depth) \
                or n_labels == 1 \
                or n_samples < self.min_samples_split:
            return self._make_leaf(y)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return self._make_leaf(y)

        left_mask = X[:, feature] <= threshold
        # 切完有一边样本太少，那这一刀也不切了
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return self._make_leaf(y)

        node = TreeNode(feature=feature, threshold=float(threshold), samples=n_samples,
                        value=self._class_counts(y))
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _make_leaf(self, y):
        return TreeNode(value=self._class_counts(y), samples=len(y))

    def _class_counts(self, y):
        return np.bincount(y, minlength=self.n_classes_).tolist()

    def _predict_one(self, x, node):
        while node.left is not None or node.right is not None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return int(np.argmax(node.value))   # 叶子里哪个类最多就判哪个

    def _node_to_dict(self, node):
        """转成规范第 14 条的 JSON 结构；叶子节点 left/right 为 None"""
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
