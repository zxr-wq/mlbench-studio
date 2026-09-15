# 成员 A 前端与可视化交付说明

## 页面功能

实验工作台可选择数据集、划分方式、评价指标、算法、Scratch/sklearn 实现、预处理、随机种子和模型超参数。参数表单由 Registry 返回的 `parameter_schema` 动态生成，支持数字、布尔和选项参数。

实验提交后，前端优先通过 WebSocket 接收进度、阶段和中间结果；连接失败时自动降级为 REST 轮询。通信实现已从页面状态中拆分为独立 Service，便于 B 替换服务端实现。

结果页根据 `task_type` 和返回的 `metrics` 动态显示：

- 分类：Accuracy、Precision、Recall、F1 和混淆矩阵。
- 回归：MSE、RMSE、MAE 和 R²。
- 聚类：Silhouette Score 和 Inertia。
- 降维：Explained Variance Ratio 和 Cumulative Explained Variance。

所有结果都可显示训练时间、推理时间、参数、实现版本和可视化数据。

## 图表组件

- 通用分类混淆矩阵。
- 通用二维样本投影和误判标记。
- 训练损失曲线。
- KNN 查询样本与 K 个最近邻。
- Naive Bayes 类别先验、特征均值和方差。
- Decision Tree 递归树结构。
- SVM 决策边界与支持向量。
- K-Means 质心迭代路径。
- PCA 二维投影与解释方差。
- Gradient Boosting 迭代损失。

统一入口为 `VisualizationPanel.vue`，其他成员只需按对接文档返回数据。

## Benchmark

Benchmark 页面展示 A 负责的 KNN 和 Naive Bayes 在 Iris、Wine、Breast Cancer 上的 Scratch/sklearn Accuracy、预测一致率、训练时间和推理时间，并包含 K 值敏感性曲线。页面数据来自可复现脚本，不是 Mock 随机数。C 后续可将静态基线替换为 Benchmark API，页面表格与图表无需重写。

## 验收情况

- Python 单元与 API 测试全部通过。
- Vite 生产构建通过。
- 已使用真实 REST 实验生成 4 项 KNN/Naive Bayes Scratch/sklearn 结果。
- 已对实验配置页、真实结果页和 Benchmark 页进行 1440 px 桌面宽度截图验收。
- 页面支持 `?view=benchmark` 和 `?experiment=<id>` 深链接，方便演示和回放。
