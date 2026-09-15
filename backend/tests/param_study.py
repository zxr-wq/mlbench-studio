"""参数敏感性实验：树的数量对准确率 / 训练时间的影响（报告用）

运行方式（在 E:\\mlbench-studio 下）：
    .\\.venv\\Scripts\\python.exe backend\\tests\\param_study.py
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from models.scratch.gradient_boosting import GradientBoosting  # noqa: E402
from models.scratch.random_forest import RandomForest  # noqa: E402

from sklearn.datasets import load_wine  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


def main():
    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    grid = [10, 30, 50, 100]

    print("=" * 60)
    print("随机森林（wine，max_depth=5）：树数量 → 准确率 / 训练时间")
    print("=" * 60)
    print(f"{'n_estimators':<14}{'accuracy':>12}{'train(s)':>12}{'predict(s)':>12}")
    for n in grid:
        model = RandomForest(n_estimators=n, max_depth=5, random_state=42)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t1 = time.perf_counter()
        pred = model.predict(X_test)
        t2 = time.perf_counter()
        print(f"{n:<14}{accuracy_score(y_test, pred):>12.4f}"
              f"{t1 - t0:>12.4f}{t2 - t1:>12.4f}")

    print()
    print("=" * 60)
    print("梯度提升（wine，lr=0.1，max_depth=3）：迭代轮数 → 准确率 / 训练时间")
    print("=" * 60)
    print(f"{'n_estimators':<14}{'accuracy':>12}{'train(s)':>12}{'final_loss':>12}")
    for n in grid:
        model = GradientBoosting(n_estimators=n, learning_rate=0.1, max_depth=3)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t1 = time.perf_counter()
        pred = model.predict(X_test)
        loss = model.get_visualization_data()["loss_history"][-1]
        print(f"{n:<14}{accuracy_score(y_test, pred):>12.4f}"
              f"{t1 - t0:>12.4f}{loss:>12.4f}")

    print()
    print("=" * 60)
    print("结论要点（可直接写进报告）：")
    print("  · 树数量增加，准确率通常先升后趋于平稳")
    print("  · 训练时间随树数量近似线性增长")
    print("  · 梯度提升的 loss 随迭代单调下降，说明每轮确实在补错")
    print("=" * 60)


if __name__ == "__main__":
    main()
