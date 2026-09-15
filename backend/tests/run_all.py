"""三个算法的 Scratch / sklearn 批量对照

运行：.\\.venv\\Scripts\\python.exe backend\\tests\\run_all.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.scratch.decision_tree import DecisionTree  # noqa: E402
from models.scratch.gradient_boosting import GradientBoosting  # noqa: E402
from models.scratch.random_forest import RandomForest  # noqa: E402

from sklearn.datasets import load_iris, load_wine  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from verification import compare_scratch_sklearn, print_comparison  # noqa: E402


def main():
    datasets = [("iris", load_iris()), ("wine", load_wine())]
    passed, total = 0, 0

    for name, data in datasets:
        X, y = data.data, data.target
        print("\n" + "=" * 52)
        print(f"数据集：{name}  {X.shape}")
        print("=" * 52)

        results = []

        results.append(compare_scratch_sklearn(
            DecisionTree(max_depth=3),
            DecisionTreeClassifier(max_depth=3, random_state=42),
            X, y, model_name="decision_tree", dataset_name=name,
        ))

        results.append(compare_scratch_sklearn(
            RandomForest(n_estimators=50, max_depth=5, random_state=42),
            RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
            X, y, model_name="random_forest", dataset_name=name,
        ))

        results.append(compare_scratch_sklearn(
            GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3),
            GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                       max_depth=3, random_state=42),
            X, y, model_name="gradient_boosting", dataset_name=name,
        ))

        for res in results:
            total += 1
            if print_comparison(res):
                passed += 1

    print("\n" + "=" * 52)
    print("可视化数据（给前端 A 用）")
    print("=" * 52)
    X, y = load_iris(return_X_y=True)

    dt = DecisionTree(max_depth=3).fit(X, y)
    dt.feature_names = list(load_iris().feature_names)
    tree_json = dt.get_visualization_data()
    print(f"决策树根节点: feature={tree_json['feature']}, "
          f"threshold={tree_json['threshold']}, samples={tree_json['samples']}")

    rf = RandomForest(n_estimators=30, max_depth=5, random_state=42).fit(X, y)
    rf_data = rf.get_visualization_data()
    print(f"随机森林: 树数={rf_data['n_estimators']}, "
          f"特征重要性={[round(v, 3) for v in rf_data['feature_importances']]}")

    gb = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3).fit(X, y)
    loss = gb.get_visualization_data()["loss_history"]
    print(f"梯度提升: loss 从 {loss[0]:.4f} 降到 {loss[-1]:.4f}（共 {len(loss)} 个点）")

    print("\n" + "=" * 52)
    print(f"总计 {passed}/{total} 项通过")
    print("=" * 52)


if __name__ == "__main__":
    main()
