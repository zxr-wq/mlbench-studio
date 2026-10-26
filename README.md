# MLBench Studio

机器学习理论与实践课程项目。前端、统一机器学习核心和 FastAPI/WebSocket 通信层相互分离，并通过统一 Config / Result 协议协作。

在线预览：<https://zxr-wq.github.io/mlbench-studio/>

界面采用米白色“实验室工作台”风格，不同任务拆分为独立路由：

- `/`：项目概览、近期实验和当前开发进度
- `/experiments/new`：三步式实验配置与运行流程
- `/runs`：支持搜索和模型筛选的实验记录表
- `/runs/:id`：指标、混淆矩阵和配置详情
- `/benchmark`：运行真实批量实验、按评价条件分组对比、导出结果集
- `/algorithms`：支持分类筛选与详情抽屉的算法注册表
- `/datasets`：数据集目录及实验入口
- 网页通过 FastAPI + WebSocket 发起并展示真实实验
- 统一 Config / Result 协议、Registry / Factory / Runner
- 支持固定 80/20、五折交叉验证以及 1/3/5 组随机种子稳定性实验
- 结果保存均值、标准差、逐折/逐种子明细和总运行耗时
- 进度状态来自后端实际完成的训练次数；前端历史摘要保存在 localStorage，完整图表保存在 IndexedDB，后端结果另保存在本机 SQLite
- 实验中刷新后可恢复连接；后端重启不会丢失已完成结果，尚未完成的实验会提示重新运行
- KNN 真实近邻投票、二维 SVM 决策区域与支持向量、K-Means 逐轮播放、树结构、集成规模与性能变化，以及任务相关诊断图
- 每张图明确记录代表性运行的种子与折数，多次指标使用均值和标准差，二者不会混淆
- 单次与批量实验可下载完整 JSON、指标 CSV；运行记录页可导入 JSON 恢复历史
- 图表支持放大、SVG 图片下载、坐标刻度、图例与悬停读数；Digits 混淆矩阵支持滚动与按行归一化
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

浏览器访问 `http://127.0.0.1:5173/mlbench-studio/`。

首次完成上述依赖安装后，电脑重启后无需重新部署或安装：双击项目根目录的 `启动本地实验台.bat`，再访问 `http://127.0.0.1:5173/mlbench-studio/` 即可恢复前后端真实实验环境。

## 验证与 Benchmark

```bash
.venv/Scripts/python -m pytest backend/tests -q
.venv/Scripts/python -m backend.benchmark.run_benchmark
.venv/Scripts/python -m backend.benchmark.validate_datasets
```

Benchmark 以任务类型分组：七个分类模型在 Iris、Wine、Breast Cancer、Digits 上对比；Linear Regression 在 Diabetes 上单独以 MSE/RMSE/MAE/R2 对比；K-Means 与 PCA 使用各自的聚类/降维指标。Scratch 与 sklearn 使用相同划分、随机种子和模型参数。`validate_datasets` 会为 8 个内置数据集各跑一条三随机种子的代表性链路，可用于录屏前快速验收。

Scratch SVM 用教学版核方法实现，在 Breast Cancer、Digits 或多折重复实验上明显慢于 sklearn；录屏快速演示建议优先使用 Iris、Moons，或选择 sklearn 对照实现。

实现与接口说明见 [B 算法与集成说明](docs/B_ALGORITHMS_AND_INTEGRATION.md)。
提交前检查见 [最终提交检查清单](docs/FINAL_SUBMISSION_CHECKLIST.md)。
可视化使用方法、验收范围和已知边界见 [网站交付说明](docs/WEBSITE_DELIVERY.md)。

推送到 `main` 分支后，GitHub Actions 会自动构建并发布 GitHub Pages。

## 当前边界

GitHub Pages 只能托管静态前端；要让公开网页运行真实训练，需要将 `backend/server/app.py` 部署到 Python 服务，并在前端构建时设置 `VITE_API_BASE`。本地开发时 Vite 会自动代理 `/api` 到 `http://127.0.0.1:8765`。

## 后续建议

1. 根据正式实验结果扩展统计分析；均值 ± 标准差不等于置信区间。
2. 将 Python 后端部署到公开服务，再为 GitHub Pages 设置 `VITE_API_BASE`。
3. 导出正式结果表、可视化截图与课程报告。
