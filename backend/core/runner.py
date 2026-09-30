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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from backend.core.factory import ModelFactory
from backend.datasets.loader import load_dataset
from backend.metrics import classification as clf_metrics
from backend.metrics import clustering, dimensionality, regression

SUPPORTED_PREPROCESSING = ("standard_scaler", "minmax_scaler")

METRIC_KEYS = {
    "classification": ("accuracy", "precision", "recall", "f1", "roc_auc"),
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
        # ROC-AUC is opt-in: it only makes sense for probability-producing binary classifiers.
        return [metric for metric in supported if metric != "roc_auc"]
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
            if key == "roc_auc":
                if len(np.unique(y_test)) != 2 or not hasattr(model, "predict_proba"):
                    raise ValueError("roc_auc 只适用于可输出概率的二分类模型")
                probabilities = model.predict_proba(X_test)
                metrics[key] = float(roc_auc_score(y_test, probabilities[:, 1]))
            else:
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


def _project_2d(X, reference=None):
    """A small dependency-free PCA projection for result visualizations."""
    X = np.asarray(X, dtype=float)
    center = X.mean(axis=0) if reference is None else np.asarray(reference, dtype=float).mean(axis=0)
    _, _, vectors = np.linalg.svd(X - center, full_matrices=False)
    basis = vectors[: min(2, X.shape[1])]
    projection = (X - center) @ basis.T
    if projection.shape[1] == 1:
        projection = np.column_stack([projection[:, 0], np.zeros(len(projection))])
    return projection


def _histogram(values, bins=12):
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    return {
        "counts": counts.astype(int).tolist(),
        "edges": [round(float(value), 5) for value in edges.tolist()],
    }


def _tree_feature_usage(node, scores):
    if node is None:
        return
    feature = getattr(node, "feature", None)
    if feature is not None:
        scores[int(feature)] += 1
    _tree_feature_usage(getattr(node, "left", None), scores)
    _tree_feature_usage(getattr(node, "right", None), scores)


def _feature_contributions(model, feature_names):
    """Return a truthful, model-dependent feature summary when one exists."""
    scores = np.zeros(len(feature_names), dtype=float)
    label = None
    coefficients = getattr(model, "coef_", None)
    if coefficients is None and hasattr(model, "model_"):
        coefficients = getattr(model.model_, "coef_", None)
    if coefficients is None:
        coefficients = getattr(model, "weights_", None)
    if coefficients is not None:
        values = np.asarray(coefficients, dtype=float)
        scores = np.abs(values).mean(axis=0) if values.ndim > 1 else np.abs(values)
        label = "absolute coefficient"
    elif hasattr(model, "root_"):
        _tree_feature_usage(model.root_, scores)
        label = "tree split count"
    elif hasattr(model, "feature_subsets_"):
        for subset in model.feature_subsets_:
            scores[np.asarray(subset, dtype=int)] += 1
        label = "forest feature usage"
    elif hasattr(model, "trees_") and model.__class__.__name__ == "GradientBoosting":
        for stage in model.trees_:
            for tree in stage:
                _tree_feature_usage(tree.root_, scores)
        label = "boosting split count"
    elif hasattr(model, "estimator_") and hasattr(model.estimator_, "feature_importances_"):
        scores = np.asarray(model.estimator_.feature_importances_, dtype=float)
        label = "feature importance"
    if label is None or not np.any(scores):
        return None
    order = np.argsort(scores)[::-1][: min(10, len(scores))]
    return {
        "label": label,
        "items": [
            {"name": str(feature_names[index]), "value": round(float(scores[index]), 6)}
            for index in order
        ],
    }


def _standard_visualization(task_type, model, bundle, X_train, X_test, y_train, y_test):
    """Build chart-ready data from the actual run, shared by every frontend."""
    raw = model.get_visualization_data() or {}
    payload = {
        **(raw if isinstance(raw, dict) else {}),
        "feature_names": list(bundle["feature_names"]),
        "target_names": list(bundle["target_names"]),
        "model_data": raw,
    }
    if task_type == "classification":
        prediction = model.predict(X_test)
        labels = np.unique(np.concatenate([np.asarray(y_test), np.asarray(prediction)]))
        matrix = [[int(((y_test == actual) & (prediction == predicted)).sum()) for predicted in labels] for actual in labels]
        coordinates = _project_2d(X_test, X_train)
        payload["classification"] = {
            "labels": [str(bundle["target_names"][int(label)]) if int(label) < len(bundle["target_names"]) else str(label) for label in labels],
            "confusion_matrix": matrix,
            "actual_distribution": [int((y_test == label).sum()) for label in labels],
            "predicted_distribution": [int((prediction == label).sum()) for label in labels],
            "points": [
                {"x": round(float(x), 5), "y": round(float(y), 5), "actual": int(actual), "predicted": int(predicted)}
                for (x, y), actual, predicted in zip(coordinates, y_test, prediction)
            ],
        }
        if len(labels) == 2 and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
            positive = labels[1]
            model_classes = np.asarray(getattr(model, "classes_", labels))
            column = int(np.where(model_classes == positive)[0][0]) if np.any(model_classes == positive) else 1
            fpr, tpr, _ = roc_curve(y_test, probabilities[:, column], pos_label=positive)
            payload["classification"]["roc_curve"] = {
                "points": [{"fpr": round(float(x), 5), "tpr": round(float(y), 5)} for x, y in zip(fpr, tpr)],
                "auc": round(float(roc_auc_score(y_test, probabilities[:, column])), 5),
            }
        contribution = _feature_contributions(model, bundle["feature_names"])
        if contribution:
            payload["feature_contributions"] = contribution
    elif task_type == "regression":
        prediction = model.predict(X_test)
        residuals = np.asarray(y_test, dtype=float) - np.asarray(prediction, dtype=float)
        payload["regression"] = {
            "pairs": [
                {"actual": round(float(actual), 4), "predicted": round(float(predicted), 4)}
                for actual, predicted in zip(y_test[:180], prediction[:180])
            ],
            "residual_histogram": _histogram(residuals),
        }
        contribution = _feature_contributions(model, bundle["feature_names"])
        if contribution:
            payload["feature_contributions"] = contribution
    elif task_type == "clustering":
        projection = _project_2d(X_train)
        labels = np.asarray(model.labels_, dtype=int)
        payload["clustering"] = {
            "points": [
                {"x": round(float(x), 5), "y": round(float(y), 5), "cluster": int(label)}
                for (x, y), label in zip(projection, labels)
            ],
            "sizes": [int((labels == index).sum()) for index in range(int(labels.max()) + 1)],
        }
    else:
        projection = model.transform(X_test)
        payload["pca"] = {
            "points": [
                {"x": round(float(row[0]), 5), "y": round(float(row[1] if len(row) > 1 else 0), 5), "label": int(label)}
                for row, label in zip(projection, y_test)
            ],
            "explained_variance_ratio": np.asarray(model.explained_variance_ratio_, dtype=float).tolist(),
        }
    loss_history = raw.get("loss_history", []) if isinstance(raw, dict) else []
    if loss_history:
        payload["loss_curve"] = [
            round(float(item.get("loss", 0) if isinstance(item, dict) else item), 7)
            for item in loss_history
        ]
    return payload


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
    if task_type in ("classification", "regression") and bundle["task_type"] != task_type:
        raise ValueError(
            f"模型 {name!r} 是 {task_type} 任务，不能在 {bundle['task_type']} 数据集 {config['dataset']!r} 上评价"
        )

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
        "visualization": _standard_visualization(
            task_type, model, bundle, X_train, X_test, y_train, y_test
        ),
    }
