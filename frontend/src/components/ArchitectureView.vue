<template>
  <section class="content-view architecture-view">
    <div class="view-intro">
      <div><span class="section-number">SYSTEM DESIGN</span><h2>小内核，清晰边界，可持续扩展</h2></div>
      <p>平台以注册表隔离算法实现，以服务层编排实验，以双通道 API 将计算状态映射到界面。</p>
    </div>

    <div class="architecture-map paper-panel">
      <div class="arch-client arch-box">
        <span>01 / CLIENT</span><h3>Vue 3 单页应用</h3>
        <p>配置表单 · 对比视图 · PCA 投影 · 混淆矩阵</p>
      </div>
      <div class="arch-flow">
        <i>REST</i><span>配置 / 结果</span><b>⇄</b><i>WS</i><span>实时进度</span>
      </div>
      <div class="arch-api arch-box">
        <span>02 / APPLICATION</span><h3>FastAPI 服务层</h3>
        <p>参数校验 · 任务编排 · 状态存储 · 连接管理</p>
      </div>
      <div class="arch-down">↓</div>
      <div class="arch-core arch-box">
        <span>03 / ML CORE</span><h3>统一实验管线</h3>
        <div class="pipeline-steps"><i>Load</i><b>→</b><i>Split</i><b>→</b><i>Scale</i><b>→</b><i>Fit</i><b>→</b><i>Evaluate</i></div>
      </div>
      <div class="arch-branches">
        <div><span>算法注册表</span><b>5</b><small>Scratch models</small></div>
        <div><span>数据注册表</span><b>4</b><small>Classic datasets</small></div>
        <div><span>评估器</span><b>4+</b><small>Metrics & plots</small></div>
      </div>
    </div>

    <div class="design-principles">
      <article class="paper-panel"><span>01</span><h3>开放封闭</h3><p>新增算法只需实现 fit / predict 并注册元信息，无需修改实验调度器。</p></article>
      <article class="paper-panel"><span>02</span><h3>公平比较</h3><p>同一次实验中的模型共享完全相同的划分、标准化策略与指标定义。</p></article>
      <article class="paper-panel"><span>03</span><h3>非阻塞计算</h3><p>训练进入工作线程，API 保持响应；WebSocket 逐阶段推送进度与中间结果。</p></article>
      <article class="paper-panel"><span>04</span><h3>可复现实验</h3><p>配置、超参数、随机种子、折叠得分和结果被一并保存，方便复盘分析。</p></article>
    </div>
  </section>
</template>
