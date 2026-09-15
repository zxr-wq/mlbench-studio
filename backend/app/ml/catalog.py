from __future__ import annotations

from app.core.registry import AlgorithmSpec, algorithm_registry
from app.ml.models import (
    DecisionTreeClassifier,
    GaussianNBClassifier,
    KNNClassifier,
    LinearSVMClassifier,
    SoftmaxRegression,
)
from app.ml.sklearn_models import SklearnGaussianNBClassifier, SklearnKNNClassifier


def register_algorithms() -> None:
    if algorithm_registry.list():
        return

    algorithm_registry.register(
        AlgorithmSpec(
            id="logistic_regression",
            name="多分类逻辑回归",
            short_name="Softmax",
            family="线性模型",
            description="使用 Softmax 与交叉熵完成多分类，支持 L2 正则化。",
            principle="线性得分经过 Softmax 归一化为类别概率，以梯度下降最小化交叉熵。",
            formula="P(y=k|x)=exp(w_k^T x)/sum_j exp(w_j^T x)",
            strengths=("训练稳定", "可解释权重", "多分类原生支持"),
            defaults={"learning_rate": 0.08, "epochs": 300, "l2": 0.001},
            parameter_schema=(
                {"key": "learning_rate", "label": "学习率", "type": "number", "min": 0.001, "max": 0.5, "step": 0.005},
                {"key": "epochs", "label": "迭代轮次", "type": "number", "min": 30, "max": 1200, "step": 10},
                {"key": "l2", "label": "L2 系数", "type": "number", "min": 0, "max": 0.2, "step": 0.001},
            ),
            factory=SoftmaxRegression,
        )
    )
    algorithm_registry.register(
        AlgorithmSpec(
            id="knn",
            name="K 近邻",
            short_name="KNN",
            family="实例学习",
            description="根据欧氏距离寻找近邻，以距离加权投票预测类别。",
            principle="相近样本拥有相似标签，预测由距离最近的 K 个训练样本共同决定。",
            formula="y_hat=argmax_c sum_{i in N_k(x)} I(y_i=c)/(d_i+eps)",
            strengths=("无需参数训练", "非线性边界", "直观易解释"),
            defaults={"k": 5, "distance": "euclidean", "weights": "distance"},
            parameter_schema=(
                {"key": "k", "label": "近邻数 K", "type": "number", "min": 1, "max": 31, "step": 2},
                {"key": "distance", "label": "距离", "type": "select", "options": ["euclidean", "manhattan"]},
                {"key": "weights", "label": "投票方式", "type": "select", "options": ["uniform", "distance"]},
            ),
            factory=KNNClassifier,
            sklearn_factory=SklearnKNNClassifier,
        )
    )
    algorithm_registry.register(
        AlgorithmSpec(
            id="naive_bayes",
            name="高斯朴素贝叶斯",
            short_name="GNB",
            family="概率模型",
            description="估计各类别的先验概率和特征高斯分布，以后验概率完成分类。",
            principle="在条件独立假设下，联合似然可分解为各特征似然的乘积。",
            formula="P(c|x) proportional to P(c) product_j N(x_j; mu_cj, sigma_cj)",
            strengths=("训练极快", "小样本友好", "概率解释清晰"),
            defaults={"var_smoothing": 1e-9},
            parameter_schema=(
                {"key": "var_smoothing", "label": "方差平滑", "type": "number", "min": 1e-12, "max": 0.01, "step": 1e-9},
            ),
            factory=GaussianNBClassifier,
            sklearn_factory=SklearnGaussianNBClassifier,
        )
    )
    algorithm_registry.register(
        AlgorithmSpec(
            id="decision_tree",
            name="CART 决策树",
            short_name="CART",
            family="树模型",
            description="递归搜索降低基尼不纯度的特征切分，形成可解释的决策规则。",
            principle="每个节点选择加权基尼不纯度下降最大的阈值，将问题递归分解。",
            formula="Gini(D)=1-sum_k p_k^2",
            strengths=("无需特征缩放", "捕捉非线性", "规则可解释"),
            defaults={"max_depth": 6, "min_samples_split": 4, "min_samples_leaf": 2},
            parameter_schema=(
                {"key": "max_depth", "label": "最大深度", "type": "number", "min": 1, "max": 15, "step": 1},
                {"key": "min_samples_split", "label": "最小分裂样本", "type": "number", "min": 2, "max": 30, "step": 1},
                {"key": "min_samples_leaf", "label": "叶节点最小样本", "type": "number", "min": 1, "max": 20, "step": 1},
            ),
            factory=DecisionTreeClassifier,
        )
    )
    algorithm_registry.register(
        AlgorithmSpec(
            id="svm",
            name="线性支持向量机",
            short_name="Linear SVM",
            family="最大间隔",
            description="以一对多策略优化多分类合页损失，并加入 L2 正则。",
            principle="最大化分类间隔等价于最小化合页损失与权重正则项。",
            formula="min_w lambda/2 ||w||^2 + mean(max(0,1-y w^T x))",
            strengths=("高维有效", "泛化能力强", "凸目标"),
            defaults={"learning_rate": 0.025, "epochs": 260, "regularization": 0.01},
            parameter_schema=(
                {"key": "learning_rate", "label": "学习率", "type": "number", "min": 0.001, "max": 0.2, "step": 0.002},
                {"key": "epochs", "label": "迭代轮次", "type": "number", "min": 30, "max": 1200, "step": 10},
                {"key": "regularization", "label": "正则系数", "type": "number", "min": 0.0001, "max": 0.2, "step": 0.001},
            ),
            factory=LinearSVMClassifier,
        )
    )
