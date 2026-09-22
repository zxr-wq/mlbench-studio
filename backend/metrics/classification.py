"""分类指标（统一规范 第 11 节）：accuracy / precision / recall / f1（macro 平均）。

numpy 直接实现，与 sklearn.metrics 对齐（见 tests/test_correctness.py）。
"""

import numpy as np


def accuracy(y_true, y_pred):
    y_true, y_pred = _as_labels(y_true, y_pred)
    return float((y_true == y_pred).mean())


def precision(y_true, y_pred, average="macro"):
    return _per_class_average(y_true, y_pred, _precision_of, average)


def recall(y_true, y_pred, average="macro"):
    return _per_class_average(y_true, y_pred, _recall_of, average)


def f1(y_true, y_pred, average="macro"):
    return _per_class_average(y_true, y_pred, _f1_of, average)


def confusion_matrix(y_true, y_pred):
    y_true, y_pred = _as_labels(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[index[t], index[p]] += 1
    return matrix.tolist()


def _precision_of(tp, fp, _fn):
    return tp / (tp + fp) if tp + fp else 0.0


def _recall_of(tp, _fp, fn):
    return tp / (tp + fn) if tp + fn else 0.0


def _f1_of(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def _per_class_average(y_true, y_pred, metric_of, average):
    y_true, y_pred = _as_labels(y_true, y_pred)
    values = [
        metric_of(
            np.sum((y_true == c) & (y_pred == c)),
            np.sum((y_true != c) & (y_pred == c)),
            np.sum((y_true == c) & (y_pred != c)),
        )
        for c in np.unique(np.concatenate([y_true, y_pred]))
    ]
    if average == "macro":
        return float(np.mean(values)) if values else 0.0
    raise ValueError(f"不支持的平均方式 {average!r}")


def _as_labels(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true 与 y_pred 长度不一致: {len(y_true)} vs {len(y_pred)}")
    return y_true, y_pred
