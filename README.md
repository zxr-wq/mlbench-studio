# MLBench Studio

机器学习理论与实践课程项目。当前 `integration/v2` 分支已将前端、统一机器学习核心和 FastAPI/WebSocket 通信层分离。

在线预览：<https://zxr-wq.github.io/mlbench-studio/>

当前版本用于确认页面结构和产品方向。界面采用“实验室工作台”风格，并将不同任务拆分为独立路由：

- `/`：项目概览、近期实验和当前开发进度
- `/experiments/new`：三步式实验配置与运行流程
- `/runs`：支持搜索和模型筛选的实验记录表
- `/runs/:id`：指标、混淆矩阵和配置详情
- `/algorithms`：支持分类筛选与详情抽屉的算法注册表
- `/datasets`：数据集目录及实验入口
- 网页通过 FastAPI + WebSocket 发起并展示真实实验
- 统一 Config / Result 协议、Registry / Factory / Runner
- 已接入 10 个 Scratch 算法：Linear Regression、Logistic Regression、KNN、Gaussian NB、SVM、K-Means、PCA、Decision Tree、Random Forest、Gradient Boosting

## 本地运行

先启动后端：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r backend/requirements.txt
.venv/Scripts/python -m uvicorn backend.server.app:app --port 8765 --reload
```

另开终端启动前端：

```bash
cd frontend
pnpm install
pnpm dev
```

浏览器访问 `http://127.0.0.1:5173`。

## 验证与 Benchmark

```bash
.venv/Scripts/python -m pytest backend/tests -q
.venv/Scripts/python -m backend.benchmark.run_benchmark
```

Benchmark 以任务类型分组：七个分类模型在 Iris、Wine、Breast Cancer、Digits 上对比；Linear Regression 在 Diabetes 上单独以 MSE/RMSE/MAE/R2 对比；K-Means 与 PCA 使用各自的聚类/降维指标。Scratch 与 sklearn 使用相同划分、随机种子和模型参数。

实现与接口说明见 [B 算法与集成说明](docs/B_ALGORITHMS_AND_INTEGRATION.md)。

推送到 `main` 分支后，GitHub Actions 会自动构建并发布 GitHub Pages。

## 当前边界

GitHub Pages 只能托管静态前端；要让公开网页运行真实训练，需要将 `backend/server/app.py` 部署到 Python 服务，并在前端构建时设置 `VITE_API_BASE`。本地开发时 Vite 会自动代理 `/api` 到 `http://127.0.0.1:8765`。

## 后续建议

1. 用 42、52、62 等多个随机种子运行正式 Benchmark，并报告均值与标准差。
2. 将 Python 后端部署到公开服务，再为 GitHub Pages 设置 `VITE_API_BASE`。
3. 导出正式结果表、可视化截图与课程报告。
