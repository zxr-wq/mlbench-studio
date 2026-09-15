"""SVM（自实现）：简化 SMO（Platt）+ one-vs-rest 多分类。

- 核函数：linear / rbf / poly，gamma 默认 'scale'（= 1 / (d * Var(X))，与 sklearn 一致）
- 核矩阵每个二分类问题预计算一次，误差向量 f 增量更新
- 算法设计与 scripts/prototype_validate.py 中的已验证原型一致
"""

import numpy as np

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


def _kernel_value_matrix(A, B, kernel, gamma, degree):
    if kernel == "linear":
        return A @ B.T
    if kernel == "poly":
        return (gamma * (A @ B.T) + 1.0) ** degree
    sq = (
        (A ** 2).sum(axis=1)[:, None]
        + (B ** 2).sum(axis=1)[None, :]
        - 2.0 * (A @ B.T)
    )
    return np.exp(-gamma * np.maximum(sq, 0.0))


def _scale_gamma(X):
    d = X.shape[1]
    variance = X.var(axis=0).mean()
    return 1.0 / (d * variance) if variance > 1e-12 else 1.0


def _smo_binary(K, y, C, tol=1e-3, eps=1e-5, max_iter=40000):
    """简化 SMO：f 缓存 + argmax|E1-E2| 二次选择启发式（含循环兜底）。"""
    n = len(y)
    alpha = np.zeros(n)
    b = 0.0
    f = -y.astype(float)  # f_i = decision_i - y_i（decision 初值 0）
    applied = 0
    attempts = 0
    streak = 0
    attempt_limit = max_iter * 2 + 5000
    while applied < max_iter and attempts < attempt_limit:
        attempts += 1
        r = y * f
        candidates = [
            i
            for i in range(n)
            if (r[i] < -tol and alpha[i] < C - eps) or (r[i] > tol and alpha[i] > eps)
        ]
        if not candidates:
            break
        i1 = candidates[int(np.argmax(np.abs(r[candidates])))]
        E1 = f[i1]
        if streak > 20:
            i2 = candidates[(candidates.index(i1) + 1 + attempts) % len(candidates)]
        else:
            i2 = int(np.argmax(np.abs(E1 - f)))
            if i2 == i1:
                i2 = candidates[(candidates.index(i1) + 1) % len(candidates)]
        y1, y2 = y[i1], y[i2]
        a1o, a2o = alpha[i1], alpha[i2]
        s = y1 * y2
        if s < 0:
            L, H = max(0.0, a2o - a1o), min(C, C + a2o - a1o)
        else:
            L, H = max(0.0, a1o + a2o - C), min(C, a1o + a2o)
        if L == H:
            streak += 1
            continue
        eta = 2 * K[i1, i2] - K[i1, i1] - K[i2, i2]
        if eta >= -1e-12:
            streak += 1
            continue
        a2n = min(max(a2o - y2 * (E1 - f[i2]) / eta, L), H)
        if abs(a2n - a2o) < eps * (a2n + a2o + eps):
            streak += 1
            continue
        a1n = a1o + s * (a2o - a2n)
        alpha[i1], alpha[i2] = a1n, a2n
        # 阈值更新（符号很重要：b 吸收 -E 而不是 +E）
        da1, da2 = a1n - a1o, a2n - a2o
        b1 = b - E1 - y1 * da1 * K[i1, i1] - y2 * da2 * K[i1, i2]
        b2 = b - f[i2] - y1 * da1 * K[i1, i2] - y2 * da2 * K[i2, i2]
        if eps < a1n < C - eps:
            b_new = b1
        elif eps < a2n < C - eps:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2
        f = f + y1 * da1 * K[i1, :] + y2 * da2 * K[i2, :] + (b_new - b)
        b = b_new
        applied += 1
        streak = 0
    return alpha, b


@register_model("svm")
class SVM(BaseModel):
    """支持向量机（scratch）。

    参数与统一 Config 的 model.params 对应::

        {"C": 1.0, "kernel": "linear"}
        {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
        {"C": 1.0, "kernel": "poly", "degree": 3}
    """

    task_type = "classification"

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, tol=1e-3, max_iter=40000):
        self.C = float(C)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.classes_ = None
        self.support_vectors_ = []  # 每个二分类器的支持向量（下标存原始样本）
        self.coefficients_ = []
        self.intercepts_ = []
        self._gamma = 1.0

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("SVM 是分类模型，fit 需要 y：model.fit(X, y)")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._gamma = _scale_gamma(X) if self.gamma == "scale" else float(self.gamma)

        cfg = (self.kernel, self._gamma, self.degree)
        # 核矩阵只依赖 X，one-vs-rest 的每个二分类器复用同一份
        K = _kernel_value_matrix(X, X, *cfg)
        self.support_vectors_ = []
        self.coefficients_ = []
        self.intercepts_ = []
        for cls in self.classes_:
            yb = np.where(y == cls, 1.0, -1.0)
            alpha, b = _smo_binary(K, yb, self.C, tol=self.tol, max_iter=self.max_iter)
            sv = alpha > 1e-6
            self.support_vectors_.append(X[sv])
            self.coefficients_.append(alpha[sv] * yb[sv])
            self.intercepts_.append(b)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        cfg = (self.kernel, self._gamma, self.degree)
        decisions = []
        for sv, coef, b in zip(self.support_vectors_, self.coefficients_, self.intercepts_):
            decisions.append(coef @ _kernel_value_matrix(sv, X, *cfg) + b)
        return np.array(decisions)

    def predict(self, X):
        if self.classes_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit")
        return self.classes_[np.argmax(self.decision_function(X), axis=0)]

    def get_params(self):
        return {
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "degree": self.degree,
            "tol": self.tol,
            "max_iter": self.max_iter,
        }

    def get_visualization_data(self):
        return {
            "support_vector_count": int(sum(len(sv) for sv in self.support_vectors_)),
            "classes": self.classes_.tolist() if self.classes_ is not None else [],
        }
