"""统一数据格式（统一规范 第 8 节）。

Dataset Loader 统一返回::

    {
        "X": np.ndarray,
        "y": np.ndarray,
        "feature_names": [...],
        "target_names": [...],
        "task_type": "classification",
    }

对于 K-Means / PCA 这类无标签任务：y 仍然保留用于结果分析和画图，
但训练时不能把 y 交给模型（kmeans.fit(X_train) 而不是 kmeans.fit(X_train, y_train)）。
"""

from functools import lru_cache

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine

LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
    "diabetes": load_diabetes,
}


@lru_cache(maxsize=None)
def load_dataset(name):
    """加载数据集，返回统一格式 dict。结果缓存，重复实验不重复读盘。"""
    if name not in LOADERS:
        raise KeyError(f"未知数据集 {name!r}，可选: {sorted(LOADERS)}")
    raw = LOADERS[name]()
    task_type = "regression" if name == "diabetes" else "classification"
    target_names = ["disease_progression"] if task_type == "regression" else [str(n) for n in raw.target_names]
    return {
        "X": np.asarray(raw.data, dtype=float),
        "y": np.asarray(raw.target, dtype=int),
        "feature_names": [str(n) for n in raw.feature_names],
        "target_names": target_names,
        "task_type": task_type,
    }
