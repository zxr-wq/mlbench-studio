"""K-Means（自实现）：k-means++ 初始化 + 多次重启取最小簇内平方误差。

聚类模型（统一规范 第 2/8 节）：
- 统一接口 fit(X) / predict(X) / fit_predict(X)，不接收、不使用 y
- get_visualization_data() 返回 {"centroid_history": [...]} 供前端画图
"""

import numpy as np

from backend.core.base_model import BaseModel
from backend.core.registry import register_model


def _kmeans_plus_plus(X, k, rng):
    centroids = [X[rng.randint(len(X))]]
    for _ in range(1, k):
        d2 = np.min(
            ((X[:, None, :] - np.asarray(centroids)[None, :, :]) ** 2).sum(-1), axis=1
        )
        total = d2.sum()
        if total <= 0:
            centroids.append(X[rng.randint(len(X))])
            continue
        probs = np.cumsum(d2 / total)
        centroids.append(X[int(np.searchsorted(probs, rng.rand()))])
    return np.asarray(centroids, dtype=float)


@register_model("kmeans")
class KMeans(BaseModel):
    """K-Means 聚类（scratch）。

    参数::

        {"k": 3, "random_state": 42, "n_init": 8, "max_iter": 300, "tol": 1e-6}
    """

    task_type = "clustering"

    def __init__(self, k=3, random_state=42, n_init=8, max_iter=300, tol=1e-6):
        self.k = int(k)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = 0.0
        self.n_iter_ = 0
        self._centroid_history = []

    def fit(self, X, y=None):
        # 注意：聚类训练不使用 y（统一规范 第 8 节），y 仅被忽略。
        X = np.asarray(X, dtype=float)
        if self.k < 1 or self.k > len(X):
            raise ValueError(f"k 必须在 1..{len(X)} 之间，当前为 {self.k}")
        rng = np.random.RandomState(self.random_state)

        best = None
        for _ in range(self.n_init):
            fit = self._single_run(X, rng)
            if best is None or fit[2] < best[2]:
                best = fit

        self.labels_, self.centroids_, self.inertia_, self.n_iter_, history = best
        self._centroid_history = history
        return self

    def _single_run(self, X, rng):
        centroids = _kmeans_plus_plus(X, self.k, rng)
        history = [centroids.copy()]
        labels = None
        n_iter = 0
        for it in range(self.max_iter):
            n_iter = it + 1
            d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            new_labels = d2.argmin(axis=1)
            new_centroids = centroids.copy()
            for c in range(self.k):
                if (new_labels == c).any():
                    new_centroids[c] = X[new_labels == c].mean(axis=0)
            shift = np.abs(new_centroids - centroids).max()
            centroids = new_centroids
            history.append(centroids.copy())
            converged = labels is not None and (new_labels == labels).all() and shift < self.tol
            labels = new_labels
            if converged:
                break
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        inertia = float(d2[np.arange(len(X)), labels].sum())
        return labels.copy(), centroids, inertia, n_iter, history

    def predict(self, X):
        if self.centroids_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit")
        X = np.asarray(X, dtype=float)
        d2 = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(-1)
        return d2.argmin(axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def get_params(self):
        return {
            "k": self.k,
            "random_state": self.random_state,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def get_visualization_data(self):
        # 每次迭代所有质心的轨迹：[[iter0 的 k 个质心], [iter1 的 k 个质心], ...]
        return {"centroid_history": [[c.tolist() for c in step] for step in self._centroid_history]}
