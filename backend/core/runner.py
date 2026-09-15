"""统一实验 Runner（统一规范 第 7/9/11/12 节）。

输入统一 Experiment Config::

    {
        "dataset": "iris",
        "split": {"test_size": 0.2, "random_state": 42},
        "preprocessing": ["standard_scaler"],
        "model": {"name": "knn", "implementation": "scratch", "params": {"k": 5}},
        "metrics": ["accuracy", "precision", "recall", "f1"],
    }

输出统一 Result::

    {
        "model": ..., "task_type": ..., "implementation": ..., "dataset": ...,
        "params": {...},
        "metrics": {...},          # 指标从 metrics 动态读取，不写死 accuracy/f1
        "training_time": ...,
        "inference_time": ...,
        "visualization": {...},
    }

关键规则：
- 分类正式实验 test_size=0.2、random_state=42、stratify=y（第 9 节）
- 聚类 / 降维训练时不能把 y 交给模型：model.fit(X_train)（第 8 节）
- 不同任务用各自的指标集合，PCA 不计算 Accuracy（第 10/11 节）
"""

import time

import numpy as np
from sklearn.model_selection import train_test_split

from backend.core.factory import ModelFactory
from backend.datasets.loader import load_dataset
from backend.metrics import classification as clf_metrics
from backend.metrics import clustering, dimensionality, regression

SUPPORTED_PREPROCESSING = ("standard_scaler", "minmax_scaler")

METRIC_KEYS = {
    "classification": ("accuracy", "precision", "recall", "f1"),
    "regression": ("mse", "rmse", "mae", "r2"),
    "clustering": ("silhouette_score", "inertia"),
    "dimensionality_reduction": ("explained_variance_ratio", "cumulative_explained_variance"),
}


def _preprocess(X_train, X_test, preprocessing):
    for step in preprocessing:
        if step not in SUPPORTED_PREPROCESSING:
            raise ValueError(f"不支持的预处理 {step!r}，可选: {SUPPORTED_PREPROCESSING}")
        if step == "standard_scaler":
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0)
            std[std == 0] = 1.0
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std
        else:  # minmax_scaler
            lo = X_train.min(axis=0)
            span = X_train.max(axis=0) - lo
            span[span == 0] = 1.0
            X_train = (X_train - lo) / span
            X_test = (X_test - lo) / span
    return X_train, X_test


def _validate_metrics(task_type, requested):
    supported = METRIC_KEYS[task_type]
    if not requested:
        return list(supported)
    unknown = [m for m in requested if m not in supported]
    if unknown:
        raise ValueError(
            f"任务 {task_type} 不支持指标 {unknown}，可用: {list(supported)}（统一规范 第 11 节）"
        )
    return list(requested)


def _compute_metrics(task_type, model, X_train, X_test, y_train, y_test, requested):
    requested = _validate_metrics(task_type, requested)
    metrics = {}

    if task_type == "classification":
        y_pred = model.predict(X_test)
        computers = {
            "accuracy": clf_metrics.accuracy,
            "precision": clf_metrics.precision,
            "recall": clf_metrics.recall,
            "f1": clf_metrics.f1,
        }
        for key in requested:
            metrics[key] = computers[key](y_test, y_pred)

    elif task_type == "regression":
        y_pred = model.predict(X_test)
        computers = {
            "mse": regression.mse,
            "rmse": regression.rmse,
            "mae": regression.mae,
            "r2": regression.r2,
        }
        for key in requested:
            metrics[key] = computers[key](y_test, y_pred)

    elif task_type == "clustering":
        labels = model.labels_
        for key in requested:
            if key == "silhouette_score":
                metrics[key] = clustering.silhouette_score(X_train, labels)
            elif key == "inertia":
                metrics[key] = clustering.inertia(X_train, labels, model.centroids_)

    else:  # dimensionality_reduction
        ratios = np.asarray(model.explained_variance_ratio_, dtype=float)
        for key in requested:
            if key == "explained_variance_ratio":
                metrics[key] = ratios.tolist()
            elif key == "cumulative_explained_variance":
                metrics[key] = dimensionality.cumulative_explained_variance(ratios)

    return metrics


def prepare_data(config, model_task_type=None):
    """按统一规范加载数据集并完成划分 + 预处理（第 8/9 节）。

    独立导出供 benchmark 复用，保证正式实验与对比实验用同一套划分规则。
    stratify=y 只在「分类数据集 + 分类模型」时启用；聚类 / 降维不按 y 分层。
    """
    split = config.get("split", {}) or {}
    test_size = float(split.get("test_size", 0.2))
    random_state = int(split.get("random_state", 42))
    preprocessing = list(config.get("preprocessing", []) or [])

    bundle = load_dataset(config["dataset"])
    X, y = bundle["X"], bundle["y"]

    # 第 9 节：分类统一 80/20 + stratify + seed 42
    stratify = y if bundle["task_type"] == "classification" and model_task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify,
    )
    X_train, X_test = _preprocess(X_train, X_test, preprocessing)
    return bundle, X_train, X_test, y_train, y_test


def run_experiment(config):
    """执行一次统一格式的实验，返回统一 Result。"""
    model_spec = config["model"]
    name = model_spec["name"]
    implementation = model_spec.get("implementation", "scratch")
    params = model_spec.get("params", {}) or {}
    requested_metrics = list(config.get("metrics", []) or [])

    # 指标与训练方式由模型的 task_type 决定（第 3/11 节）
    model = ModelFactory.create(name, params, implementation)
    task_type = model.task_type
    bundle, X_train, X_test, y_train, y_test = prepare_data(config, model_task_type=task_type)

    started = time.perf_counter()
    if task_type in ("classification", "regression"):
        model.fit(X_train, y_train)
    else:
        # 第 8 节：聚类 / 降维训练不接收 y
        model.fit(X_train)
    training_time = time.perf_counter() - started

    started = time.perf_counter()
    if task_type in ("classification", "regression", "clustering"):
        model.predict(X_test)
    elif task_type == "dimensionality_reduction":
        model.transform(X_test)
    inference_time = time.perf_counter() - started

    return {
        "model": name,
        "task_type": task_type,
        "implementation": implementation,
        "dataset": config["dataset"],
        "params": model.get_params(),
        "metrics": _compute_metrics(
            task_type, model, X_train, X_test, y_train, y_test, requested_metrics
        ),
        "training_time": round(training_time, 6),
        "inference_time": round(inference_time, 6),
        "visualization": model.get_visualization_data(),
    }
