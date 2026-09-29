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
from types import SimpleNamespace

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine, make_circles, make_friedman1, make_moons

LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
    "diabetes": load_diabetes,
}


def _synthetic(name, X, y, feature_names):
    return SimpleNamespace(data=X, target=y, feature_names=feature_names, target_names=["class 0", "class 1"])


def _moons():
    X, y = make_moons(n_samples=500, noise=.22, random_state=42)
    return _synthetic("moons", X, y, ["x₁", "x₂"])


def _circles():
    X, y = make_circles(n_samples=500, noise=.12, factor=.48, random_state=42)
    return _synthetic("circles", X, y, ["x₁", "x₂"])


def _friedman1():
    X, y = make_friedman1(n_samples=600, n_features=10, noise=1.0, random_state=42)
    return SimpleNamespace(data=X, target=y, feature_names=[f"feature_{index + 1}" for index in range(10)], target_names=["target"])


LOADERS.update({"moons": _moons, "circles": _circles, "friedman1": _friedman1})


@lru_cache(maxsize=None)
def load_dataset(name):
    """加载数据集，返回统一格式 dict。结果缓存，重复实验不重复读盘。"""
    if name not in LOADERS:
        raise KeyError(f"未知数据集 {name!r}，可选: {sorted(LOADERS)}")
    raw = LOADERS[name]()
    task_type = "regression" if name in {"diabetes", "friedman1"} else "classification"
    target_names = ["disease_progression"] if task_type == "regression" else [str(n) for n in raw.target_names]
    return {
        "X": np.asarray(raw.data, dtype=float),
        "y": np.asarray(raw.target, dtype=float if task_type == "regression" else int),
        "feature_names": [str(n) for n in raw.feature_names],
        "target_names": target_names,
        "task_type": task_type,
    }
