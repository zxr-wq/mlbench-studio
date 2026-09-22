"""PCA 的 sklearn 参考实现（统一规范 第 15 节）。"""

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


@register_model("pca", implementation="sklearn")
class PCA(BaseModel):
    task_type = "dimensionality_reduction"

    def __init__(self, n_components=None):
        self.n_components = n_components
        self._model = None
        self._fit_X = None
        self.explained_variance_ratio_ = None

    def fit(self, X, y=None):
        self._fit_X = np.asarray(X, dtype=float)
        self._model = SklearnPCA(n_components=self.n_components)
        self._model.fit(self._fit_X)
        self.explained_variance_ratio_ = self._model.explained_variance_ratio_
        return self

    def transform(self, X):
        if self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit")
        return self._model.transform(np.asarray(X, dtype=float))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def get_params(self):
        return {"n_components": self.n_components}

    def get_visualization_data(self):
        if self._model is None:
            return {}
        return {
            "explained_variance_ratio": self._model.explained_variance_ratio_.tolist(),
            "projection": self.transform(self._fit_X).tolist(),
        }
