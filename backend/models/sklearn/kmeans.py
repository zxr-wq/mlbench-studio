"""K-Means 的 sklearn 参考实现（统一规范 第 15 节）。"""

import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


@register_model("kmeans", implementation="sklearn")
class KMeans(BaseModel):
    task_type = "clustering"

    def __init__(self, k=3, random_state=42, n_init=10, max_iter=300, tol=1e-4):
        self.k = int(k)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._model = None
        self.labels_ = None
        self.centroids_ = None
        self.inertia_ = 0.0

    def fit(self, X, y=None):
        # 聚类训练不使用 y（统一规范 第 8 节），y 仅被忽略。
        self._model = SklearnKMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self._model.fit(np.asarray(X, dtype=float))
        self.labels_ = self._model.labels_
        self.centroids_ = self._model.cluster_centers_
        self.inertia_ = float(self._model.inertia_)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit")
        return self._model.predict(np.asarray(X, dtype=float))

    def fit_predict(self, X):
        self.fit(X)
        return self._model.labels_

    def get_params(self):
        return {
            "k": self.k,
            "random_state": self.random_state,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def get_visualization_data(self):
        return {}
