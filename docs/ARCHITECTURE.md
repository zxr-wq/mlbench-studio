# 架构设计

## 设计目标

系统围绕三件事设计：保证不同算法接受相同的数据处理和验证流程；让实验配置与结果可复现；让新增算法、数据集和可视化组件彼此独立。

## 分层

```text
Vue 3 SPA
  ├─ REST：目录、创建实验、读取结果和历史
  └─ WebSocket：状态、进度和阶段性结果
                   ↓
FastAPI API Layer
  ├─ Pydantic 参数边界
  ├─ HTTP / WebSocket 路由
  └─ 静态资源托管
      ↓
Application Services
  ├─ DatasetService：按标识加载公开数据集及描述
  ├─ ExperimentService：状态机、后台计算、结果保存
  └─ ConnectionManager：按实验编号广播消息
      ↓
ML Core
  ├─ AlgorithmRegistry：元数据 + 参数模式 + 模型工厂
  ├─ Split / Scale / PCA / Metrics
  └─ 5 个统一 Classifier 接口的自主实现
```

## 实验时序

1. 前端从目录接口读取数据集和算法元数据，动态生成配置项。
2. `POST /api/experiments` 对配置进行校验，生成实验编号并立即返回。
3. 前端连接 `/api/ws/experiments/{id}`；若连接不可用，自动降级为 REST 轮询。
4. 服务层用相同的索引划分训练所有模型。标准化参数只在训练集拟合，避免数据泄漏。
5. 每个模型在工作线程中训练，事件循环继续服务 WebSocket 和 HTTP。
6. 模型完成后立刻广播中间结果；全部完成后计算最优模型并封存配置、指标和时间。

## 扩展性

### 算法扩展

算法只依赖 `Classifier.fit/predict` 协议。注册项同时保存原理、公式、优势、默认参数和参数模式，因此前后端共享同一个事实来源。新增算法不改实验循环。

### 数据集扩展

`DatasetService` 将外部对象归一成 `DatasetBundle`。数据集本体不进入仓库；可继续接入 OpenML、对象存储或用户上传服务。

### 指标与图表扩展

训练结果是稳定的 JSON 结构。新指标可在服务层增加，新图表只需消费相应字段，不影响模型实现。

## 正确性与公平性

- 同次实验所有模型使用完全相同的切分索引。
- 标准化器只读取训练折统计量。
- 多分类 Precision、Recall、F1 使用宏平均，避免大类别掩盖小类别表现。
- 交叉验证汇总所有折的 out-of-fold 预测，再生成总体混淆矩阵。
- 随机过程全部由实验随机种子控制。
- 对模型、数据集标识与超参数范围在 API 边界校验。

## 当前边界与生产化建议

当前记录存储于进程内，适合课程演示。生产部署可将记录换为 PostgreSQL/Redis，并把计算任务下沉到 Celery、RQ 或独立推理队列；注册表与 API 合约无需改变。
