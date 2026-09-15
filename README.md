# MLBench Studio

机器学习理论与实践课程项目的第一版交互框架。

在线预览：<https://zxr-wq.github.io/mlbench-studio/>

当前版本用于确认页面结构和产品方向。界面采用“实验室工作台”风格，并将不同任务拆分为独立路由：

- `/`：项目概览、近期实验和当前开发进度
- `/experiments/new`：三步式实验配置与运行流程
- `/runs`：支持搜索和模型筛选的实验记录表
- `/runs/:id`：指标、混淆矩阵、逐类 Precision/Recall/F1 和配置详情
- `/benchmark`：算法 × 数据集统一对比矩阵、PCA 方差分析、CSV 导出
- `/algorithms`：支持分类筛选与详情抽屉的算法注册表
- `/datasets`：数据集目录及实验入口
- 真实实验在浏览器本地 CPU 上运行，完成后自动进入新实验详情
- 与真实后端解耦的 `BenchmarkClient` 接口

## 机器学习核心（src/ml）

前端浏览器内 ML 核心（TypeScript），经 `scripts/prototype_validate.py` Python 原型验证后移植：

- `data.ts`：加载 `public/data/*.json` 数据集（由 `scripts/export_datasets.py` 从 scikit-learn 导出：iris / wine / breast_cancer / digits）
- `preprocess.ts`：mulberry32 可复现随机种子、分层划分、分层五折交叉验证、Standard/Min-Max Scaler
- `metrics.ts`：Accuracy、Macro Precision/Recall/F1、混淆矩阵、逐类指标、Silhouette 系数
- `svm.ts`：简化 SMO（Platt）训练的多分类 SVM，支持 linear / rbf / poly 核，gamma='scale'
- `kmeans.ts`：k-means++ 初始化 + 多次重启取最小簇内平方误差，多数投票映射为分类器
- `pca.ts`：协方差矩阵 Jacobi 特征分解实现 PCA（与 numpy.linalg.eigh 对齐到机器精度）
- `pipeline.ts`：统一实验流水线 划分 → 预处理/降维 → 训练 → 预测 → 评估
- `registry.ts`：Benchmark 对比矩阵的模型规格（SVM 三种核、K-Means、PCA+SVM）

## Python 后端（backend/，四人统一接口规范）

按《MLBench Studio 四人开发统一规范（最新版）》实现，目录与规范第 16 节一致：

```
backend/
├── core/          # base_model.py 统一基础类 · registry.py 注册 · factory.py 工厂 · runner.py 统一 Runner
├── datasets/      # load_dataset → {X, y, feature_names, target_names, task_type}
├── metrics/       # classification / regression / clustering / dimensionality 四套指标
├── models/
│   ├── scratch/   # svm · kmeans · pca（@register_model 注册，自主实现）
│   └── sklearn/   # 同名 sklearn 对照实现（implementation="sklearn"）
├── benchmark/     # run_benchmark.py 按任务分别 Benchmark + scratch/sklearn 对照
└── tests/         # 接口合规测试 + 与 sklearn 正确性对比
```

规范要点与落实：

- BaseModel 统一 `fit(X, y=None)` / `get_params()` / `get_visualization_data()`，所有模型声明 `task_type`
- 分类用 `predict()`；K-Means 用 `predict()` / `fit_predict()`；PCA 用 `transform()` / `fit_transform()`，不实现 predict
- 统一算法名（svm / kmeans / pca…），通过 `ModelFactory.create(name, params, implementation)` 创建，杜绝 if/elif
- 统一 Config：`{dataset, split:{test_size, random_state}, preprocessing:[], model:{name, implementation, params}, metrics:[]}`
- 统一 Result：`{model, task_type, implementation, dataset, params, metrics{}, training_time, inference_time, visualization}`，指标从 metrics 动态读取
- 正式划分统一 test_size=0.2、random_state=42、分类 stratify=y；K-Means / PCA 训练不接收 y
- 可视化数据统一从 `get_visualization_data()` 输出（K-Means 质心轨迹、PCA 方差解释率 + 投影、SVM 支持向量数）

运行：

```bash
python -m backend.benchmark.run_benchmark            # 全部数据集（iris/wine/breast_cancer/digits）
python -m backend.benchmark.run_benchmark iris       # 指定数据集
python -m unittest backend.tests.test_interface backend.tests.test_correctness
```

## 本地运行

```bash
pnpm install
pnpm dev
```

浏览器访问 `http://127.0.0.1:5173`。

推送到 `main` 分支后，GitHub Actions 会自动构建并发布 GitHub Pages。

## 当前边界

实验在浏览器端完成：`src/services/benchmark.ts` 中的 `LocalBenchmarkClient` 直接调用 `src/ml` 的流水线，数据集 JSON 随页面静态加载。下一阶段可实现 `WebSocketBenchmarkClient`，保持页面和业务类型不变，将训练切换到 Python 后端（如 scikit-learn 对照、更大规模数据）。

## 后续建议

1. 根据反馈调整工作台布局、色彩和信息优先级。
2. 确定十大算法和数据集名单。
3. 固化配置 Schema 与前后端消息协议。
4. 建立 Python Pipeline、注册表及首个算法闭环。
5. 加入实验历史、可视化和报告导出。
