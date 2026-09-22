"""统一实验对比 Benchmark（统一规范 第 9/10/15 节）。

规则：
- 正式划分统一：test_size=0.2、random_state=42、分类 stratify=y
- 不同任务分别 Benchmark：分类 / 回归 / 聚类 / 降维 分开列表，绝不混在一张 Accuracy 表里
- Scratch / sklearn 对照：统一比较 Accuracy Difference、Prediction Agreement、
  Training Time（分类）；聚类 / 降维按各自任务指标比较

用法::

    python -m backend.benchmark.run_benchmark            # 全部数据集
    python -m backend.benchmark.run_benchmark iris       # 指定数据集
"""

import json
import time
from pathlib import Path

from backend.core.factory import ModelFactory
from backend.core.registry import registered_names
from backend.core.runner import prepare_data, run_experiment
from backend.datasets.loader import load_dataset
from backend.metrics import classification as clf_metrics

import backend.models  # noqa: F401  触发 @register_model 注册

DATASETS = ("iris", "wine", "breast_cancer", "digits")

# 各任务模型默认参数（统一算法名，model.params 全部收拢在这里）
DEFAULT_PARAMS = {
    "svm": {"C": 1.0, "kernel": "rbf"},
    "kmeans": None,  # k 依赖数据集类别数，运行时填
    "pca": {"n_components": 0.95},
}

DEFAULT_METRICS = {
    "classification": ["accuracy", "precision", "recall", "f1"],
    "clustering": ["silhouette_score", "inertia"],
    "dimensionality_reduction": ["explained_variance_ratio", "cumulative_explained_variance"],
}


def _task_default_params(name, bundle):
    params = DEFAULT_PARAMS.get(name, {}) or {}
    if name == "kmeans":
        params = {**params, "k": len(bundle["target_names"])}
    return params


def benchmark_task(task_type, dataset, implementation="scratch"):
    """对某一任务类型下所有已注册模型跑正式基准，返回统一 Result 列表。"""
    results = []
    for name in registered_names(implementation):
        model_class = ModelFactory.create(name, {}, implementation)
        if model_class.task_type != task_type:
            continue
        bundle = load_dataset(dataset)
        config = {
            "dataset": dataset,
            "split": {"test_size": 0.2, "random_state": 42},
            "preprocessing": ["standard_scaler"],
            "model": {
                "name": name,
                "implementation": implementation,
                "params": _task_default_params(name, bundle),
            },
            "metrics": DEFAULT_METRICS[task_type],
        }
        results.append(run_experiment(config))
    return results


def compare_scratch_sklearn(name, dataset):
    """同一配置下 scratch / sklearn 对照（统一规范 第 15 节）。"""
    if name not in registered_names("scratch") or name not in registered_names("sklearn"):
        return None
    params = _task_default_params(name, load_dataset(dataset))
    task_type = ModelFactory.create(name, params, "scratch").task_type
    bundle, X_train, X_test, y_train, y_test = prepare_data(
        {"dataset": dataset, "preprocessing": ["standard_scaler"]},
        model_task_type=task_type,
    )

    scratch = ModelFactory.create(name, params, "scratch")
    reference = ModelFactory.create(name, params, "sklearn")

    started = time.perf_counter()
    if task_type in ("classification", "regression"):
        scratch.fit(X_train, y_train)
    else:
        scratch.fit(X_train)
    scratch_time = time.perf_counter() - started

    started = time.perf_counter()
    if task_type in ("classification", "regression"):
        reference.fit(X_train, y_train)
    else:
        reference.fit(X_train)
    sklearn_time = time.perf_counter() - started

    row = {"model": name, "dataset": dataset, "task_type": task_type}

    if task_type == "classification":
        y_scratch = scratch.predict(X_test)
        y_ref = reference.predict(X_test)
        acc_s, acc_r = clf_metrics.accuracy(y_test, y_scratch), clf_metrics.accuracy(y_test, y_ref)
        row.update(
            scratch_accuracy=acc_s,
            sklearn_accuracy=acc_r,
            accuracy_difference=round(acc_s - acc_r, 4),
            prediction_agreement=round(float((y_scratch == y_ref).mean()), 4),
            scratch_training_time=round(scratch_time, 4),
            sklearn_training_time=round(sklearn_time, 4),
        )
    elif task_type == "regression":
        from backend.metrics import regression as reg_metrics

        y_scratch, y_ref = scratch.predict(X_test), reference.predict(X_test)
        row.update(
            scratch_r2=reg_metrics.r2(y_test, y_scratch),
            sklearn_r2=reg_metrics.r2(y_test, y_ref),
            mse_difference=round(
                reg_metrics.mse(y_test, y_scratch) - reg_metrics.mse(y_test, y_ref), 4
            ),
            scratch_training_time=round(scratch_time, 4),
            sklearn_training_time=round(sklearn_time, 4),
        )
    elif task_type == "clustering":
        from backend.metrics import clustering

        row.update(
            scratch_silhouette=clustering.silhouette_score(X_train, scratch.labels_),
            sklearn_silhouette=clustering.silhouette_score(X_train, reference.labels_),
            inertia_difference=round(
                clustering.inertia(X_train, scratch.labels_, scratch.centroids_)
                - clustering.inertia(X_train, reference.labels_, reference.centroids_),
                4,
            ),
            scratch_training_time=round(scratch_time, 4),
            sklearn_training_time=round(sklearn_time, 4),
        )
    else:  # dimensionality_reduction
        import numpy as np

        proj_s, proj_r = scratch.transform(X_train), reference.transform(X_train)
        n_components = min(proj_s.shape[1], proj_r.shape[1])
        proj_s, proj_r = proj_s[:, :n_components], proj_r[:, :n_components]
        # 特征向量符号不唯一：按各分量与 sklearn 投影的相关性翻转后比较
        corr = (proj_s * proj_r).sum(axis=0)
        aligned = proj_s * np.where(corr < 0, -1.0, 1.0)
        agreement = float(
            np.mean(np.abs(aligned - proj_r) < 1e-6 * (1 + np.abs(proj_r)))
        )
        row.update(
            scratch_explained=scratch.explained_variance_ratio_.tolist(),
            sklearn_explained=reference.explained_variance_ratio_.tolist(),
            projection_agreement=round(agreement, 4),
            scratch_training_time=round(scratch_time, 4),
            sklearn_training_time=round(sklearn_time, 4),
        )
    return row


def run_full_benchmark(datasets=DATASETS):
    """全部任务 × 全部数据集：统一结果 + scratch/sklearn 对照。"""
    payload = {"results": [], "comparisons": []}
    for dataset in datasets:
        for task_type in ("classification", "clustering", "dimensionality_reduction"):
            payload["results"].extend(benchmark_task(task_type, dataset))
        for name in registered_names("scratch"):
            comparison = compare_scratch_sklearn(name, dataset)
            if comparison:
                payload["comparisons"].append(comparison)
    return payload


def _print_report(payload):
    print("=" * 72)
    print("统一实验对比（test_size=0.2 · random_state=42 · stratify=y）")
    print("=" * 72)
    for result in payload["results"]:
        metrics = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}=[{len(v)} 项]"
            for k, v in result["metrics"].items()
        )
        print(
            f"[{result['task_type']:<24}] {result['model']:<8} × {result['dataset']:<14} "
            f"{metrics}  train={result['training_time']:.3f}s"
        )
    print("-" * 72)
    print("Scratch / sklearn 对照")
    for row in payload["comparisons"]:
        if row["task_type"] == "classification":
            print(
                f"{row['model']:<8} × {row['dataset']:<14} "
                f"scratch={row['scratch_accuracy']:.4f} sklearn={row['sklearn_accuracy']:.4f} "
                f"diff={row['accuracy_difference']:+.4f} agreement={row['prediction_agreement']:.3f}"
            )
        elif row["task_type"] == "clustering":
            print(
                f"{row['model']:<8} × {row['dataset']:<14} "
                f"silhouette scratch={row['scratch_silhouette']:.4f} "
                f"sklearn={row['sklearn_silhouette']:.4f} "
                f"Δinertia={row['inertia_difference']:+.4f}"
            )
        else:
            print(
                f"{row['model']:<8} × {row['dataset']:<14} "
                f"projection_agreement={row['projection_agreement']:.3f}"
            )


if __name__ == "__main__":
    import sys

    selected = sys.argv[1:] or list(DATASETS)
    payload = run_full_benchmark(selected)
    _print_report(payload)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "benchmark_results.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out_file}")
