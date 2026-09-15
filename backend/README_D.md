# 成员 D 交付说明：树模型与集成学习

负责人：D　｜　分支：`tree-ensemble-models`

## 1. 已完成的文件

| 文件 | 内容 |
|---|---|
| `models/scratch/decision_tree.py` | 决策树 CART 分类（Gini + 递归划分 + 树结构导出） |
| `models/scratch/random_forest.py` | 随机森林（Bootstrap + 随机特征 + 投票 + 特征重要性） |
| `models/scratch/gradient_boosting.py` | 梯度提升（softmax + 残差 + 牛顿步长叶子 + loss 曲线） |
| `tests/verification.py` | **全组通用的 Scratch ↔ sklearn 对照验证模板** |
| `tests/check_env.py` | 环境自检 |
| `tests/compare_decision_tree.py` | 决策树单个对照实验 |
| `tests/run_all.py` | 三个算法 × 两个数据集的批量对照 |
| `tests/param_study.py` | 参数敏感性实验（树数量 → 准确率 / 时间） |
| `tests/test_tree_models.py` | 12 个单元测试（pytest，全部通过） |

## 2. 怎么跑

在 `E:\mlbench-studio` 目录下：

```powershell
.\.venv\Scripts\python.exe backend\tests\run_all.py       # 三个算法全量对照
.\.venv\Scripts\python.exe backend\tests\param_study.py   # 参数敏感性实验
```

单元测试：`.\.venv\Scripts\python.exe -m pytest backend\tests\test_tree_models.py -v`
（12 passed，覆盖 Gini 正确性、树结构规范、随机森林必须复用自写树、GB loss 单调下降、接口签名等）

## 3. 当前实验结果（iris + wine，80/20、seed=42、分层抽样）

| 模型 | 数据集 | Scratch | sklearn | 差值 | 预测一致率 |
|---|---|---|---|---|---|
| decision_tree | iris | 0.9667 | 0.9667 | 0.0000 | 1.0000 |
| decision_tree | wine | 0.9444 | 0.9444 | 0.0000 | 0.9444 |
| random_forest | iris | 0.9000 | 0.9000 | 0.0000 | 1.0000 |
| random_forest | wine | 0.9722 | 1.0000 | 0.0278 | 0.9722 |
| gradient_boosting | iris | 0.9333 | 0.9667 | 0.0333 | 0.9667 |
| gradient_boosting | wine | 0.9722 | 0.9444 | 0.0278 | 0.9722 |

**6/6 项通过**（判定标准：指标差 < 0.05 且预测一致率 > 0.9）

## 4. 参数敏感性（wine）

随机森林：10 棵 → 0.9444（0.09s），30 棵 → 0.9722（0.26s），100 棵 → 0.9722（0.86s）
梯度提升：10 轮 → 0.9444（loss 0.117），50 轮 → 0.9722（loss 0.0001），100 轮 → 0.9722（10.8s）

结论：树数量超过约 30 后准确率趋于平稳，训练时间近似线性增长。

## 5. 给全组的验证模板怎么用

其他人照抄这段即可：

```python
from verification import compare_scratch_sklearn, print_comparison

res = compare_scratch_sklearn(
    scratch_model=MyKNN(k=5),
    sklearn_model=KNeighborsClassifier(n_neighbors=5),
    X=X, y=y,
    model_name="knn",
    dataset_name="iris",
)
print_comparison(res)
```

返回结构已按规范第 12 条对齐（`metrics` / `training_time` / `inference_time`）。

## 6. 给前端 A 的可视化数据

| 模型 | `get_visualization_data()` 返回 |
|---|---|
| 决策树 | `{feature, threshold, samples, value, left, right}` 递归结构（规范第 14 条） |
| 随机森林 | `{n_estimators, tree_depths, feature_importances, feature_names}` |
| 梯度提升 | `{loss_history, n_estimators, learning_rate}` |

## 7. 后续

- 等 B 的 `BaseModel` / `register_model` 到位后，三个类各加一行装饰器 + 改继承即可
- 报告章节（D 负责）：三种算法原理 + 上面的对照表 + 参数敏感性分析
