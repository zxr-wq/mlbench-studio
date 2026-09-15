"""决策树 Scratch vs sklearn 对照实验

运行方式（在 E:\\mlbench-studio 目录下）：
    .\\.venv\\Scripts\\python.exe backend\\tests\\compare_decision_tree.py

规范第 9 条：正式实验统一 test_size=0.2、random_state=42、分类用 stratify
规范第 15 条：比较 Scratch Accuracy / Sklearn Accuracy / 差值 / 预测一致率 / 训练时间
"""

import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.scratch.decision_tree import DecisionTree  # noqa: E402

from sklearn.datasets import load_iris  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402


def main():
    data = load_iris()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    # 统一划分：80% 训练 / 20% 测试 / seed=42 / 分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- 你的实现 ----
    mine = DecisionTree(max_depth=3)
    mine.feature_names = feature_names
    t0 = time.perf_counter()
    mine.fit(X_train, y_train)
    t1 = time.perf_counter()
    y_pred_mine = mine.predict(X_test)
    t2 = time.perf_counter()

    # ---- sklearn 参考实现 ----
    ref = DecisionTreeClassifier(max_depth=3, random_state=42)
    t3 = time.perf_counter()
    ref.fit(X_train, y_train)
    t4 = time.perf_counter()
    y_pred_ref = ref.predict(X_test)

    acc_mine = accuracy_score(y_test, y_pred_mine)
    acc_ref = accuracy_score(y_test, y_pred_ref)
    agreement = float(np.mean(y_pred_mine == y_pred_ref))

    print("=" * 46)
    print(f"{'':<22}{'Scratch':>12}{'sklearn':>12}")
    print("-" * 46)
    print(f"{'Accuracy':<22}{acc_mine:>12.4f}{acc_ref:>12.4f}")
    print(f"{'Training time (s)':<22}{t1 - t0:>12.5f}{t4 - t3:>12.5f}")
    print("-" * 46)
    print(f"Accuracy 差值         : {abs(acc_mine - acc_ref):.4f}")
    print(f"预测一致率(agreement) : {agreement:.4f}")
    print(f"推理时间 (s)          : {t2 - t1:.5f}")
    print("=" * 46)

    ok = abs(acc_mine - acc_ref) < 0.05 and agreement > 0.9
    print("结论：", "通过，你的实现和 sklearn 基本一致 ✅" if ok else "差距偏大，检查 _gini / _best_split")

    print("\n树结构（给前端 A 画树用，节选）:")
    vis = mine.get_visualization_data()
    print(f"  根节点: feature={vis['feature']}, threshold={vis['threshold']}, "
          f"samples={vis['samples']}, value={vis['value']}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print("还没写完：", e)
        print("去 backend/models/scratch/decision_tree.py 把 TODO 1 和 TODO 2 填上再跑。")
