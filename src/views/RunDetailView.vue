<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import { useLabStore } from '../stores/lab'

const route = useRoute()
const { runs } = useLabStore()
const run = computed(() => runs.value.find((item) => item.id === route.params.id) ?? runs.value[0])
</script>

<template>
  <div class="page detail-page">
    <RouterLink to="/runs" class="back-link">← 返回运行记录</RouterLink>
    <header class="run-detail-header">
      <div><span class="status-label"><i></i>运行成功</span><h1>{{ run.id }} · {{ run.model }}</h1><p>{{ run.dataset }} / Stratified 80-20 / seed 42</p></div>
      <RouterLink to="/experiments/new" class="button ghost">复制为新实验</RouterLink>
    </header>

    <section class="detail-metrics">
      <div><span>Accuracy</span><strong>{{ run.accuracy.toFixed(3) }}</strong><small>测试集</small></div>
      <div><span>Macro F1</span><strong>{{ run.f1.toFixed(3) }}</strong><small>测试集</small></div>
      <div><span>训练耗时</span><strong>{{ run.duration.toFixed(2) }}s</strong><small>本地 CPU</small></div>
      <div><span>可复现</span><strong class="yes-value"><AppIcon name="check" :size="24" /> Yes</strong><small>配置与随机种子已保存</small></div>
    </section>

    <div class="detail-grid">
      <section class="content-section">
        <div class="section-title-row"><div><span class="overline">EVALUATION</span><h2>类别表现</h2></div></div>
        <div class="confusion-wrap">
          <div class="confusion-grid">
            <span class="heat h4">69</span><span class="heat h1">2</span>
            <span class="heat h1">3</span><span class="heat h5">40</span>
          </div>
          <div class="confusion-copy"><span>预测结果矩阵</span><p>模型在恶性类别上出现 3 个漏判样本，下一步适合查看错误样本的半径与纹理特征。</p><button class="plain-link">查看错误样本 <AppIcon name="arrow" :size="14" /></button></div>
        </div>
      </section>
      <aside class="content-section config-code">
        <div class="section-title-row"><div><span class="overline">RUN CONFIG</span><h2>实验配置</h2></div></div>
        <pre><code>dataset: breast_cancer
splitter: stratified_80_20
preprocessing: standard_scaler
model:
  name: {{ run.model.toLowerCase() }}
  kernel: rbf
  C: 1.0
metrics: [accuracy, macro_f1]
seed: 42</code></pre>
      </aside>
    </div>
  </div>
</template>
