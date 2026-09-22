"""降维指标（统一规范 第 11 节）：explained_variance_ratio / cumulative_explained_variance。

PCA 不应该强行计算 Accuracy。
"""

import numpy as np


def explained_variance_ratio(eigenvalues):
    """按特征值从大到小给出各主成分解释方差占比。"""
    values = np.asarray(eigenvalues, dtype=float)
    values = np.clip(values, 0, None)
    total = values.sum()
    if total <= 0:
        return [0.0] * len(values)
    return (values / total).tolist()


def cumulative_explained_variance(ratios):
    """累计解释方差，第 i 项 = 前 i+1 个主成分的解释方差之和。"""
    return np.cumsum(np.asarray(ratios, dtype=float)).tolist()
