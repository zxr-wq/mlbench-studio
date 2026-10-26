<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import AlgorithmGuide from '../components/AlgorithmGuide.vue'
import ResultVisualizations from '../components/ResultVisualizations.vue'
import { algorithms, datasets } from '../data/catalog'
import { useLabStore } from '../stores/lab'

const route = useRoute()
const { runs } = useLabStore()
const run = computed(() => runs.value.find((item) => item.id === route.params.id))
const metricEntries = computed(() => run.value ? Object.entries(run.value.metrics ?? { accuracy: run.value.accuracy, f1: run.value.f1 })
  .map(([metric, value]) => [metric, Array.isArray(value) ? value[value.length - 1] : value] as [string, number])
  .filter(([, value]) => typeof value === 'number' && Number.isFinite(value)).slice(0, 4) : [])
const metricLabel = (metric: string) => ({ accuracy: 'Accuracy', f1: 'Macro F1', precision: 'Precision', recall: 'Recall', mse: 'MSE', rmse: 'RMSE', mae: 'MAE', r2: 'R²', silhouette_score: 'Silhouette', inertia: 'Inertia', cumulative_explained_variance: '累计解释方差' }[metric] ?? metric)
const metricDeviation = (metric: string) => {
  const value = run.value?.metricStd?.[metric]
  return Number(Array.isArray(value) ? value[value.length - 1] : value ?? 0)
}
const evaluationLabel = computed(() => {
  if (!run.value) return ''
  const evaluation = run.value.evaluation
  if (!evaluation) return '固定随机种子，可复现'
  const method = evaluation.strategy === 'kfold' ? `${evaluation.n_splits} 折交叉验证` : '固定 80 / 20 划分'
  return `${method} · ${evaluation.repetitions} 组随机种子 · ${evaluation.total_runs} 次训练`
})
const configText = computed(() => run.value ? JSON.stringify(run.value.config ?? { dataset: run.value.dataset, model: run.value.model, implementation: run.value.implementation, params: run.value.params, metrics: run.value.metrics ? Object.keys(run.value.metrics) : [], evaluation: run.value.evaluation }, null, 2) : '')
const modelName = (id: string) => algorithms.find((item) => item.id === id)?.name ?? id
const datasetName = (id: string) => datasets.find((item) => item.id === id)?.name ?? id
function download(kind: 'json' | 'csv') {
  if (!run.value) return
  const csvRows = Object.entries(run.value.metrics ?? {}).flatMap(([key, value]) => {
    const deviation = run.value?.metricStd?.[key]
    if (Array.isArray(value)) {
      return value.map((item, index) => [
        `${key}[${index + 1}]`,
        item,
        Array.isArray(deviation) ? deviation[index] ?? 0 : deviation ?? 0,
      ].join(','))
    }
    return [`${key},${value},${Array.isArray(deviation) ? deviation[0] ?? 0 : deviation ?? 0}`]
  })
  const data = kind === 'json'
    ? JSON.stringify({ exported_at: new Date().toISOString(), experiment: run.value }, null, 2)
    : ['metric,mean,std', ...csvRows].join('\n')
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
    <template v-if="run">
      <header class="run-detail-header result-hero">
      <div><span class="status-label"><i></i>实验运行完成</span><h1>{{ run.id }} · {{ modelName(run.model) }}</h1><p>{{ datasetName(run.dataset) }} · {{ run.implementation ?? 'scratch' }} · {{ evaluationLabel }}</p></div>
      <div class="result-actions"><button class="button ghost" @click="download('json')">下载完整 JSON</button><button class="button ghost" @click="download('csv')">下载指标 CSV</button><RouterLink to="/experiments/new" class="button green-button">创建下一次实验</RouterLink></div>
    </header>

    <section class="detail-metrics result-metric-strip">
      <div v-for="[metric, value] in metricEntries" :key="metric"><span>{{ metricLabel(metric) }}</span><strong>{{ Number(value).toFixed(3) }}</strong><small class="metric-uncertainty">{{ run.evaluation && run.evaluation.total_runs > 1 ? `± ${metricDeviation(metric).toFixed(3)} · ` : '' }}{{ run.taskType ?? 'classification' }}</small></div>
      <div><span>总模型计算耗时</span><strong>{{ run.duration.toFixed(2) }}s</strong><small>训练 {{ (run.trainingTime ?? run.duration).toFixed(2) }}s · 预测 {{ (run.inferenceTime ?? 0).toFixed(2) }}s</small></div>
    </section>

    <ResultVisualizations :visualization="run.visualization" :task-type="run.taskType" />
    <AlgorithmGuide :model="run.model" :task-type="run.taskType" :metrics="run.metrics" />

    <section class="result-config-card"><div><span class="section-kicker">REPRODUCIBILITY</span><h2>本次实验配置</h2><p>结果中的主数值是多次训练的平均值；“±”后面是标准差，越小说明模型越稳定。</p></div><pre><code>{{ configText }}</code></pre></section>
    </template>
    <section v-else class="missing-run-state">
      <span class="section-kicker">EXPERIMENT NOT FOUND</span>
      <h1>没有找到这条实验记录</h1>
      <p>它可能来自其他浏览器、已被清理，或链接编号不正确。请返回运行记录重新选择实验。</p>
      <RouterLink to="/runs" class="button primary">返回运行记录</RouterLink>
    </section>
  </div>
</template>
