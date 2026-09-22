"""PCA（自实现）：协方差矩阵特征分解 + 最大方差方向投影。

降维模型（统一规范 第 2 节）：
- 统一接口 fit(X) / transform(X) / fit_transform(X)，不强制实现 predict
- n_components 为整数（维度）或 (0,1) 小数（保留方差比例）
"""

import numpy as np

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


@register_model("pca")
class PCA(BaseModel):
    """主成分分析（scratch）。

    参数::

        {"n_components": 2}        # 保留 2 个主成分
        {"n_components": 0.95}     # 保留 95% 方差
    """

    task_type = "dimensionality_reduction"

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = 0
        self._fit_X = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self._fit_X = X
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        covariance = (Xc.T @ Xc) / (n - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0, None)
        eigenvectors = eigenvectors[:, order]

        total = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues / total if total > 0 else np.zeros(d)

        if self.n_components is None:
            k = d
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            cumulative = np.cumsum(self.explained_variance_ratio_)
            k = int(np.searchsorted(cumulative, self.n_components) + 1)
        else:
            k = int(self.n_components)
        self.n_components_ = max(1, min(k, d))

        self.components_ = eigenvectors[:, : self.n_components_].T
        self.explained_variance_ = eigenvalues[: self.n_components_]
        return self

    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def get_params(self):
        return {"n_components": self.n_components}

    def get_visualization_data(self):
        if self.components_ is None:
            return {}
        # projection：训练样本在主成分上的投影，前端可取前两维画散点
        return {
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
            "projection": self.transform(self._fit_X).tolist(),
        }
