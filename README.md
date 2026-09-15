# 智模工坊 · ML Studio

一个面向“机器学习理论与实践”课程大作业的全栈可视化实验系统。平台将经典分类算法的原理、自主实现、实验评估和前端可视化放在同一个可复现流程中，重点服务于**公平比较、信息表达、可解释性与复现性**。

## 已实现能力

- 4 个经典数据集：Iris、Wine、Breast Cancer、Digits；运行时由 Scikit-learn 加载，仓库不保存数据集本体。
- 5 个 NumPy 自主实现模型：Softmax 多分类逻辑回归、KNN、高斯朴素贝叶斯、CART 决策树、线性 SVM；KNN 和朴素贝叶已接入 sklearn 对照。
- 3 种验证策略：随机划分、分层抽样、3-10 折分层交叉验证。
- 可调超参数、训练集拟合的标准化、固定随机种子与完整实验配置记录。
- Accuracy、Macro Precision、Macro Recall、Macro F1、逐折准确率、训练耗时和推理耗时。
- 模型横向比较、Scratch/sklearn 一致性、混淆矩阵、PCA 二维投影、KNN 近邻、朴素贝叶统计量、误判标记、损失曲线和特征重要性数据。
- FastAPI REST API + WebSocket 训练进度；Vue 3 响应式单页界面。
- 算法注册表、数据集服务和实验编排服务，支持低耦合扩展。

## 快速运行（Windows PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Set-Location frontend
npm install
npm run build
Set-Location ..
.\.venv\Scripts\python.exe run.py
```

浏览器打开 <http://127.0.0.1:8765>。接口文档位于 <http://127.0.0.1:8765/docs>。如需更换端口，可先设置 `ML_STUDIO_PORT` 环境变量。

### 前后端分离开发

后端：

```powershell
.\.venv\Scripts\python.exe run.py
```

前端（另一个终端）：

```powershell
Set-Location frontend
npm run dev
```

Vite 默认将 `/api` 和 WebSocket 请求代理到 `127.0.0.1:8765`；可用 `ML_STUDIO_API` 修改目标地址。

### GitHub Pages 演示

`main` 分支会自动构建并发布纯前端演示版。演示版无需 Python 服务，使用成员 A 的真实 Benchmark 基线数据，可浏览算法、配置、运行结果、图表与可视化；本地全栈模式仍通过真实 API 执行新实验。

## 测试

```powershell
$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

测试覆盖算法基本性能、数据集/算法目录接口和异步实验闭环。

## 项目结构

```text
backend/app/
  api/                 # 请求模型、REST 与 WebSocket 路由
  core/registry.py     # 算法元数据和工厂注册表
  ml/                  # 自主实现模型与预处理
  services/            # 数据集加载、切分、训练、评估与状态
frontend/src/
  components/          # 配置、结果、图表、图鉴、记录与架构视图
  App.vue              # 应用状态和实时连接
docs/                  # 架构与算法说明
```

详细设计见 [架构文档](docs/ARCHITECTURE.md) 与 [算法说明](docs/ALGORITHM_NOTES.md)。

## 新增算法

1. 在 `backend/app/ml/` 中实现 `fit` 和 `predict`。
2. 在 `backend/app/ml/catalog.py` 注册元数据、默认参数与工厂。
3. 前端会从 `/api/algorithms` 自动读取并生成算法卡片和参数表单。

这条扩展路径不需要修改实验服务或前端核心状态逻辑。
