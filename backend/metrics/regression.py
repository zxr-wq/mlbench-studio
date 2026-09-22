"""回归指标（统一规范 第 11 节）：mse / rmse / mae / r2。"""

import numpy as np


def mse(y_true, y_pred):
    y_true, y_pred = _as_float(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    y_true, y_pred = _as_float(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    y_true, y_pred = _as_float(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _as_float(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true 与 y_pred 长度不一致: {len(y_true)} vs {len(y_pred)}")
    return y_true, y_pred
