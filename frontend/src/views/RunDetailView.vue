<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import AlgorithmGuide from '../components/AlgorithmGuide.vue'
import ResultVisualizations from '../components/ResultVisualizations.vue'
import { useLabStore } from '../stores/lab'

const route = useRoute()
const { runs } = useLabStore()
const run = computed(() => runs.value.find((item) => item.id === route.params.id) ?? runs.value[0])
const metricEntries = computed(() => Object.entries(run.value.metrics ?? { accuracy: run.value.accuracy, f1: run.value.f1 })
  .filter(([, value]) => typeof value === 'number').slice(0, 4))
const metricLabel = (metric: string) => ({ accuracy: 'Accuracy', f1: 'Macro F1', precision: 'Precision', recall: 'Recall', mse: 'MSE', rmse: 'RMSE', mae: 'MAE', r2: 'R²', silhouette_score: 'Silhouette', inertia: 'Inertia', cumulative_explained_variance: '累计解释方差' }[metric] ?? metric)
const configText = computed(() => JSON.stringify({ dataset: run.value.dataset, model: run.value.model, implementation: run.value.implementation, params: run.value.params, metrics: run.value.metrics ? Object.keys(run.value.metrics) : [] }, null, 2))
function download(kind: 'json' | 'csv') {
  const data = kind === 'json'
    ? JSON.stringify({ exported_at: new Date().toISOString(), experiment: run.value }, null, 2)
    : ['metric,value', ...Object.entries(run.value.metrics ?? {}).filter(([, value]) => typeof value === 'number').map(([key, value]) => `${key},${value}`)].join('\n')
  const blob = new Blob([data], { type: kind === 'json' ? 'application/json' : 'text/csv;charset=utf-8' })
  const anchor = document.createElement('a')
  anchor.href = URL.createObjectURL(blob)
  anchor.download = `${run.value.id}-${run.value.model}-result.${kind}`
  anchor.click()
  URL.revokeObjectURL(anchor.href)
}
</script>

<template>
  <div class="page detail-page refreshed-detail">
    <RouterLink to="/runs" class="back-link">← 返回历史记录</RouterLink>
    <header class="run-detail-header result-hero">
      <div><span class="status-label"><i></i>实验运行完成</span><h1>{{ run.id }} · {{ run.model }}</h1><p>{{ run.dataset }} · {{ run.implementation ?? 'scratch' }} · 固定随机种子，可复现</p></div>
      <div class="result-actions"><button class="button ghost" @click="download('json')">下载完整 JSON</button><button class="button ghost" @click="download('csv')">下载指标 CSV</button><RouterLink to="/experiments/new" class="button green-button">创建下一次实验</RouterLink></div>
    </header>

    <section class="detail-metrics result-metric-strip">
      <div v-for="[metric, value] in metricEntries" :key="metric"><span>{{ metricLabel(metric) }}</span><strong>{{ Number(value).toFixed(3) }}</strong><small>{{ run.taskType ?? 'classification' }}</small></div>
      <div><span>训练耗时</span><strong>{{ run.duration.toFixed(2) }}s</strong><small>本地 CPU</small></div>
    </section>

    <ResultVisualizations :visualization="run.visualization" :task-type="run.taskType" />
    <AlgorithmGuide :model="run.model" :task-type="run.taskType" :metrics="run.metrics" />

    <section class="result-config-card"><div><span class="section-kicker">REPRODUCIBILITY</span><h2>本次实验配置</h2><p>把同样的配置再次运行，可以复现同一数据划分与结果。</p></div><pre><code>{{ configText }}</code></pre></section>
  </div>
</template>
