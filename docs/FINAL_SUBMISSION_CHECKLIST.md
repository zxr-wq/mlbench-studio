# MLBench Studio 最终提交检查清单

## 已验证的技术内容

- 10 个 Scratch 算法和 10 个 sklearn 对照均已注册。
- 分类、回归、聚类、降维四类任务均可通过 FastAPI 实际运行。
- 统一 BaseModel、Registry、Factory、Runner、Config 和 Result 已接通。
- 固定 80/20、五折交叉验证以及 1、3、5 组随机种子实验可用。
- 8 个内置数据集代表性链路通过，多次实验报告均值和标准差。
- 前端生产构建通过；100% 和 125% 缩放下主要页面无横向溢出。
- 完整实验 JSON 和指标 CSV 可下载，大型图表数据使用 IndexedDB 保存。

## 推荐验收命令

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m backend.benchmark.validate_datasets
.\.venv\Scripts\python.exe -m backend.benchmark.run_benchmark iris
Set-Location frontend
pnpm run build
```

## 录屏建议链路

1. Iris + KNN：快速分类链路，Accuracy 约 0.93。
2. Breast Cancer + Logistic Regression：二分类、ROC-AUC 和特征贡献。
3. Diabetes + Linear Regression：R²、RMSE、真实值预测图和残差图。
4. Iris + K-Means：Silhouette、簇规模与质心迭代轨迹。
5. Wine + PCA：解释方差谱与二维投影。
6. Iris + Decision Tree：混淆矩阵、特征贡献与树结构。

具体数值会因随机种子和评价策略变化；多随机种子实验应重点解释平均值和标准差，而不是承诺每次都得到完全相同的非固定结果。

## 课程提交前必须人工补齐

- 在报告中填写组号、组名、所有成员姓名和学号。
- 写清每位成员的详细分工与个人收获感悟。
- 使用最终界面重新截取配置、运行、结果与下载页面。
- 将正式 Benchmark 的表格与差异分析写入报告。
- 在报告中放入公开仓库链接和本地运行方法。
- 更新 Word 目录与页码，检查图片、表格和分页后导出唯一 PDF。
- 邮件仅附一个最终 PDF，不附源码、压缩包或数据集。
