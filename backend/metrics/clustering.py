"""聚类指标（统一规范 第 11 节）：silhouette_score / inertia。

聚类不能拿 Accuracy 评价，统一用轮廓系数 + 簇内平方误差。
"""

import numpy as np
from sklearn.metrics import silhouette_score as _sk_silhouette


def silhouette_score(X, labels, sample_size=1200, random_state=42):
    """轮廓系数 ∈ [-1, 1]，越高说明簇内紧、簇间远。大样本抽样计算保证速度。"""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return 0.0
    if sample_size and len(X) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X, labels = X[idx], labels[idx]
    return float(_sk_silhouette(X, labels))


def inertia(X, labels, centroids):
    """簇内平方误差：每个样本到其所属簇中心的距离平方之和。"""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    centroids = np.asarray(centroids, dtype=float)
    if not len(centroids):
        return 0.0
    return float(((X - centroids[labels]) ** 2).sum())
