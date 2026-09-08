# MLBench Studio

机器学习理论与实践课程项目的第一版交互框架。

在线预览：<https://zxr-wq.github.io/mlbench-studio/>

当前版本用于确认页面结构和产品方向。界面采用“实验室工作台”风格，并将不同任务拆分为独立路由：

- `/`：项目概览、近期实验和当前开发进度
- `/experiments/new`：三步式实验配置与运行流程
- `/runs`：支持搜索和模型筛选的实验记录表
- `/runs/:id`：指标、混淆矩阵和配置详情
- `/algorithms`：支持分类筛选与详情抽屉的算法注册表
- `/datasets`：数据集目录及实验入口
- 可交互的模拟实验运行，完成后自动进入新实验详情
- 与真实后端解耦的 `BenchmarkClient` 接口

## 本地运行

```bash
pnpm install
pnpm dev
```

浏览器访问 `http://127.0.0.1:5173`。

推送到 `main` 分支后，GitHub Actions 会自动构建并发布 GitHub Pages。

## 当前边界

当前实验数据由 `src/services/benchmark.ts` 中的模拟客户端生成。下一阶段可实现 `WebSocketBenchmarkClient`，保持页面和业务类型不变，将模拟数据替换为 Python 后端返回的真实实验数据。

## 后续建议

1. 根据反馈调整工作台布局、色彩和信息优先级。
2. 确定十大算法和数据集名单。
3. 固化配置 Schema 与前后端消息协议。
4. 建立 Python Pipeline、注册表及首个算法闭环。
5. 加入实验历史、可视化和报告导出。
