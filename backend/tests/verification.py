"""Scratch ↔ sklearn 统一对照验证模板（成员 D 负责，全组通用）

用法（其他人直接抄）：

    from verification import compare_scratch_sklearn, print_comparison

    res = compare_scratch_sklearn(
        scratch_model=MyKNN(k=5),
        sklearn_model=KNeighborsClassifier(n_neighbors=5),
        X=X, y=y,
        model_name="knn",
        dataset_name="iris",
    )
    print_comparison(res)

规范第 9 条：正式实验统一 test_size=0.2、random_state=42、分类 stratify=y
规范第 15 条：比较 Scratch 指标 / sklearn 指标 / 差值 / 预测一致率 / 训练时间
"""

import time

import numpy as np

from sklearn.metrics import accuracy_score, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


def compare_scratch_sklearn(scratch_model, sklearn_model, X, y,
                            model_name="model", dataset_name="dataset",
                            task_type="classification",
                            test_size=0.2, random_state=42):
    """跑一次对照实验，返回统一格式的字典（可直接塞进 Result 的 metrics 里）"""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # 统一划分：80/20 + 固定种子，分类任务分层抽样
    stratify = y if task_type in ("classification", "regression") and _is_discrete(y) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # ---- Scratch 自己的实现 ----
    t0 = time.perf_counter()
    scratch_model.fit(X_train, y_train)
    t1 = time.perf_counter()
    pred_scratch = scratch_model.predict(X_test)
    t2 = time.perf_counter()

    # ---- sklearn 参考实现 ----
    t3 = time.perf_counter()
    sklearn_model.fit(X_train, y_train)
    t4 = time.perf_counter()
    pred_sklearn = sklearn_model.predict(X_test)

    # ---- 统一按任务类型算指标 ----
    if task_type == "regression":
        main_metric = "r2"
        score_scratch = r2_score(y_test, pred_scratch)
        score_sklearn = r2_score(y_test, pred_sklearn)
    else:
        main_metric = "accuracy"
        score_scratch = accuracy_score(y_test, pred_scratch)
        score_sklearn = accuracy_score(y_test, pred_sklearn)

    agreement = float(np.mean(np.asarray(pred_scratch) == np.asarray(pred_sklearn)))

    return {
        "model": model_name,
        "dataset": dataset_name,
        "task_type": task_type,
        "implementation": "scratch_vs_sklearn",
        "main_metric": main_metric,
        "metrics": {
            "scratch": float(score_scratch),
            "sklearn": float(score_sklearn),
            "difference": float(abs(score_scratch - score_sklearn)),
            "prediction_agreement": agreement,
        },
        "training_time": {"scratch": t1 - t0, "sklearn": t4 - t3},
        "inference_time": {"scratch": t2 - t1},
    }


def print_comparison(res):
    m = res["metrics"]
    print(f"\n[{res['model']}]  数据集: {res['dataset']}  指标: {res['main_metric']}")
    print("-" * 52)
    print(f"{'':<24}{'Scratch':>13}{'sklearn':>13}")
    print(f"{res['main_metric']:<24}{m['scratch']:>13.4f}{m['sklearn']:>13.4f}")
    print(f"{'training time (s)':<24}"
          f"{res['training_time']['scratch']:>13.5f}{res['training_time']['sklearn']:>13.5f}")
    print("-" * 52)
    print(f"差值 difference      : {m['difference']:.4f}")
    print(f"预测一致率 agreement : {m['prediction_agreement']:.4f}")

    ok = m["difference"] < 0.05 and m["prediction_agreement"] > 0.9
    print("结论：", "通过 ✅ 与 sklearn 基本一致" if ok else "差距偏大 ⚠️ 需要检查实现")
    return ok


def _is_discrete(y):
    return np.asarray(y).dtype.kind in "iu" or len(np.unique(y)) <= 20
