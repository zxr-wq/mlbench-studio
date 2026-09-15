"""树模型单元测试（成员 D）

运行方式（在 E:\\mlbench-studio 下）：
    .\\.venv\\Scripts\\python.exe -m pytest backend\\tests\\test_tree_models.py -v
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from models.scratch.decision_tree import DecisionTree  # noqa: E402
from models.scratch.gradient_boosting import GradientBoosting  # noqa: E402
from models.scratch.random_forest import RandomForest  # noqa: E402

from sklearn.datasets import load_iris  # noqa: E402


def _iris():
    return load_iris(return_X_y=True)


def test_gini_pure_node_is_zero():
    """全同类的节点，不纯度必须是 0"""
    tree = DecisionTree()
    tree.n_classes_ = 2
    assert abs(tree._gini(np.array([1, 1, 1, 1])) - 0.0) < 1e-9


def test_gini_balanced_node():
    """两类各一半时，Gini = 1 - 0.5² - 0.5² = 0.5"""
    tree = DecisionTree()
    tree.n_classes_ = 2
    assert abs(tree._gini(np.array([0, 0, 1, 1])) - 0.5) < 1e-9


def test_decision_tree_fits_perfectly_on_simple_data():
    """一条竖线就能分开的数据，决策树应该 100% 分对"""
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTree(max_depth=1).fit(X, y)
    assert np.array_equal(tree.predict(X), y)


def test_decision_tree_reaches_reasonable_accuracy():
    """iris 上准确率应超过 0.9"""
    X, y = _iris()
    tree = DecisionTree(max_depth=3).fit(X, y)
    assert np.mean(tree.predict(X) == y) > 0.9


def test_tree_structure_follows_frontend_spec():
    """树结构必须包含规范第 14 条的六个字段，叶子节点 left/right 为 None"""
    X, y = _iris()
    tree = DecisionTree(max_depth=2)
    tree.feature_names = list(load_iris().feature_names)
    tree.fit(X, y)
    root = tree.get_visualization_data()

    for key in ("feature", "threshold", "samples", "value", "left", "right"):
        assert key in root

    # 找到一个叶子，确认 left / right 都是 None
    node, leaf = root, None
    while node is not None:
        if node["left"] is None and node["right"] is None:
            leaf = node
            break
        node = node["left"]
    assert leaf is not None
    assert leaf["left"] is None and leaf["right"] is None


def test_random_forest_uses_own_decision_tree():
    """随机森林内部装的必须是本项目自己实现的 DecisionTree"""
    X, y = _iris()
    forest = RandomForest(n_estimators=5, max_depth=3, random_state=0).fit(X, y)
    assert len(forest.trees) == 5
    for tree in forest.trees:
        assert isinstance(tree, DecisionTree)


def test_random_forest_majority_vote():
    """投票结果必须落在真实类别集合内，且准确率合理"""
    X, y = _iris()
    forest = RandomForest(n_estimators=30, max_depth=5, random_state=42).fit(X, y)
    pred = forest.predict(X)
    assert set(np.unique(pred)).issubset(set(np.unique(y)))
    assert np.mean(pred == y) > 0.9


def test_random_forest_feature_importances_sum_to_one():
    X, y = _iris()
    forest = RandomForest(n_estimators=20, max_depth=3, random_state=42).fit(X, y)
    imp = forest.get_visualization_data()["feature_importances"]
    assert len(imp) == X.shape[1]
    assert abs(sum(imp) - 1.0) < 1e-6


def test_gradient_boosting_loss_decreases():
    """梯度提升的 loss 必须随迭代下降（说明每轮确实在补错）"""
    X, y = _iris()
    gb = GradientBoosting(n_estimators=30, learning_rate=0.1, max_depth=3).fit(X, y)
    loss = gb.get_visualization_data()["loss_history"]
    assert len(loss) >= 30
    assert loss[-1] < loss[0]


def test_gradient_boosting_accuracy():
    X, y = _iris()
    gb = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3).fit(X, y)
    assert np.mean(gb.predict(X) == y) > 0.9


def test_all_models_expose_required_interface():
    """三个模型都必须有 task_type 和规范要求的四个方法"""
    for model in (
        DecisionTree(max_depth=3),
        RandomForest(n_estimators=5, max_depth=3),
        GradientBoosting(n_estimators=5, max_depth=3),
    ):
        assert model.task_type == "classification"
        assert callable(model.fit)
        assert callable(model.predict)
        assert callable(model.get_params)
        assert callable(model.get_visualization_data)
        assert isinstance(model.get_params(), dict)


def test_fit_accepts_y_as_keyword():
    """规范第 1 条：fit 的签名必须是 fit(X, y=None)"""
    X, y = _iris()
    model = DecisionTree(max_depth=2)
    model.fit(X, y=y)      # 关键字传参也要能用
    assert model.predict(X).shape[0] == X.shape[0]
