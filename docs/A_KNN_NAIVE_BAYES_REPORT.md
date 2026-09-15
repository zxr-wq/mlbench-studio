# KNN 与朴素贝叶实验说明

## 实验目标

本实验验证 NumPy Scratch 实现的 KNN 和 Gaussian Naive Bayes 是否正确，并与 sklearn 参考实现在完全相同的数据划分和预处理下比较。实验同时检查参数敏感性、运行时间以及前端可视化数据。

## 算法实现

KNN 支持 `k`、`euclidean/manhattan` 距离以及 `uniform/distance` 投票。`fit` 保存训练集，`predict` 计算测试样本到训练样本的距离，找到前 K 个近邻并投票。距离加权时，近邻权重与距离成反比。

Gaussian Naive Bayes 按类别计算先验概率、各特征的均值与方差，预测时在对数空间累加各特征的高斯对数似然。方差平滑用于避免常量特征导致除零或数值溢出。

## 实验协议

- 数据集：Iris、Wine、Breast Cancer Wisconsin。
- 划分：80% 训练集、20% 测试集，`random_state=42`，分类标签分层抽样。
- 预处理：StandardScaler，均值和标准差只在训练集上拟合。
- 指标：Accuracy、Macro Precision、Macro Recall、Macro F1。
- 时间：训练和推理分开计时，连续运行 5 次后取平均值。

## Scratch 与 sklearn 对照结果

| 数据集 | 算法 | Scratch Accuracy | sklearn Accuracy | Accuracy Difference | Prediction Agreement |
|---|---|---:|---:|---:|---:|
| Iris | KNN | 0.9667 | 0.9667 | 0.0000 | 1.0000 |
| Iris | Naive Bayes | 0.9667 | 0.9667 | 0.0000 | 1.0000 |
| Wine | KNN | 0.9722 | 0.9722 | 0.0000 | 1.0000 |
| Wine | Naive Bayes | 0.9722 | 0.9722 | 0.0000 | 1.0000 |
| Breast Cancer | KNN | 0.9561 | 0.9561 | 0.0000 | 1.0000 |
| Breast Cancer | Naive Bayes | 0.9298 | 0.9298 | 0.0000 | 1.0000 |

在这三个固定划分上，两个 Scratch 模型与 sklearn 的预测完全一致。这证明核心距离、投票、高斯似然与后验比较逻辑是正确的。数据集很小，毫秒级时间会明显受 Python 调度和测量误差影响，不应从单次数据推导通用速度结论。

| 数据集 | 算法 | Scratch 训练 ms | sklearn 训练 ms | Scratch 推理 ms | sklearn 推理 ms |
|---|---|---:|---:|---:|---:|
| Iris | KNN | 0.019 | 0.378 | 0.459 | 0.394 |
| Iris | Naive Bayes | 0.088 | 0.497 | 0.031 | 0.111 |
| Wine | KNN | 0.012 | 0.440 | 0.711 | 0.448 |
| Wine | Naive Bayes | 0.108 | 0.552 | 0.042 | 0.138 |
| Breast Cancer | KNN | 0.029 | 0.399 | 7.324 | 11.776 |
| Breast Cancer | Naive Bayes | 0.179 | 0.682 | 0.056 | 0.163 |

## 参数实验

KNN 的 `k` 取 1、3、5、7、9、11、15。Iris 和 Wine 在本次划分上从 `k=7` 开始达到 1.0000 Accuracy；Breast Cancer 在 `k=3` 时最高，Accuracy 为 0.9825，`k=1` 时为 0.9386。这说明 K 太小时更容易受局部噪声影响，但最优 K 仍取决于数据集。

Naive Bayes 的 `var_smoothing` 取 `1e-12`、`1e-9`、`1e-6`和 `1e-3`时，三个数据集的 Accuracy 在本次划分上没有变化。这表明该参数在当前数据范围内主要起数值稳定作用。

## 可视化设计

KNN 在预测后保存测试样本的 K 个近邻，返回二维投影、类别、排名和距离。前端用 Q 表示查询点，以虚线连接近邻，便于观察投票过程。Naive Bayes 返回各类先验概率、特征均值和方差，前端以条形图和统计量表格展示。

## 复现方法

```powershell
$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe backend\benchmark_a.py
```

脚本会输出完整指标、平均时间、预测一致率和参数实验数据。
