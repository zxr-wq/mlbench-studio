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
- 已接入 8 个 Scratch 算法：KNN、Gaussian NB、SVM、K-Means、PCA、Decision Tree、Random Forest、Gradient Boosting

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

推送到 `main` 分支后，GitHub Actions 会自动构建并发布 GitHub Pages。

## 当前边界

GitHub Pages 只能托管静态前端；要让公开网页运行真实训练，需要将 `backend/server/app.py` 部署到 Python 服务，并在前端构建时设置 `VITE_API_BASE`。本地开发时 Vite 会自动代理 `/api` 到 `http://127.0.0.1:8765`。

## 后续建议

1. 根据反馈调整工作台布局、色彩和信息优先级。
2. 确定十大算法和数据集名单。
3. 固化配置 Schema 与前后端消息协议。
4. 建立 Python Pipeline、注册表及首个算法闭环。
5. 加入实验历史、可视化和报告导出。
