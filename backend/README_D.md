# D 的开工指南（树模型与集成学习）

## 1. 环境（已配好，在 E 盘）

- 项目目录：`E:\mlbench-studio`
- Python 虚拟环境：`E:\mlbench-studio\.venv`（numpy 2.5.3 + scikit-learn 1.9.1）
- 当前分支：`tree-ensemble-models`

以后所有命令都在 `E:\mlbench-studio` 目录下执行，并且**用 .venv 里的 python**：

```powershell
cd E:\mlbench-studio
.\.venv\Scripts\python.exe backend\tests\check_env.py              # 环境自检
.\.venv\Scripts\python.exe backend\tests\compare_decision_tree.py  # 跟 sklearn 对照
```

VSCode 里按 `Ctrl+Shift+P` → `Python: Select Interpreter` → 选 `E:\mlbench-studio\.venv\Scripts\python.exe`，
之后直接点右上角运行按钮即可。

## 2. 现在要做的唯一一件事

打开 `backend/models/scratch/decision_tree.py`，把两个 TODO 填上：

- **TODO 1 `_gini()`**：算不纯度，3 行（公式写在注释里）
- **TODO 2 `_best_split()`**：遍历所有特征和阈值，找加权不纯度最小的那一刀

填完跑：

```powershell
.\.venv\Scripts\python.exe backend\tests\compare_decision_tree.py
```

看到「通过，你的实现和 sklearn 基本一致 ✅」就说明决策树写对了，
iris 上的合理水平是 **0.95 左右**（sklearn 也是这个数）。

## 3. 顺序

1. 决策树跑通（当前步骤）
2. 随机森林（复用自己写的决策树，禁止调用 sklearn 的树）
3. 梯度提升（记录 loss_history）
4. 统一 Scratch / sklearn 验证模板（你负责给全组用）

## 4. 规范速查（2026-09-15 版）

- 名字：`decision_tree` / `random_forest` / `gradient_boosting`
- `task_type = "classification"`
- 四个方法：`fit(X, y=None)`、`predict(X)`、`get_params()`、`get_visualization_data()`
- 参数：决策树 `{max_depth, min_samples_split}`；随机森林 `{n_estimators, max_depth}`
- 树结构导出格式（规范第 14 条）：`{feature, threshold, samples, value, left, right}`，叶子 `left=right=None`
- 正式实验：80/20 划分、`random_state=42`、分类 `stratify=y`

## 5. 推送前

```powershell
git add .
git commit -m "feat: 完成决策树 scratch 实现"
git push -u origin tree-ensemble-models
```

如果 push 报 403，说明你还没被加为仓库协作者——让组长（仓库所有者 zxr-wq）在
GitHub 仓库 Settings → Collaborators 里邀请你，或者你 fork 一份到自己账号再提 PR。
