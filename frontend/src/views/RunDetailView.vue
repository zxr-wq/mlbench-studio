<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import AlgorithmGuide from '../components/AlgorithmGuide.vue'
import ResultVisualizations from '../components/ResultVisualizations.vue'
import { algorithms, datasets } from '../data/catalog'
import { useLabStore } from '../stores/lab'
import { downloadFile, metricsCsv, metricName } from '../services/exports'

const route = useRoute()
const { runs, isHydrating, storageNotice } = useLabStore()
const run = computed(() => runs.value.find((item) => item.id === route.params.id))
const metricEntries = computed(() => run.value ? Object.entries(run.value.metrics ?? { accuracy: run.value.accuracy, f1: run.value.f1 })
  .map(([metric, value]) => [metric, Array.isArray(value) ? value[value.length - 1] : value] as [string, number])
  .filter(([key, value]) => key !== 'explained_variance_ratio' && typeof value === 'number' && Number.isFinite(value)) : [])
const metricLabel = metricName
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
  const data = kind === 'json'
    ? JSON.stringify({ schema_version: 2, exported_at: new Date().toISOString(), experiment: run.value }, null, 2)
    : metricsCsv([run.value])
  downloadFile(`${run.value.id}-${run.value.model}-result.${kind}`,data,kind==='json' ? 'application/json' : 'text/csv;charset=utf-8')
}
</script>

<template>
  <div class="page detail-page refreshed-detail">
    <RouterLink to="/runs" class="back-link">← 返回历史记录</RouterLink>
    <template v-if="run">
      <header class="run-detail-header result-hero">
      <div><span class="status-label"><i></i>实验运行完成</span><h1><span class="run-title-id">{{ run.id }}</span><span class="run-title-separator"> · </span><span>{{ modelName(run.model) }}</span></h1><p>{{ datasetName(run.dataset) }} · {{ run.implementation ?? 'scratch' }} · {{ evaluationLabel }}</p></div>
      <div class="result-actions"><button class="button ghost" @click="download('json')">下载完整 JSON</button><button class="button ghost" @click="download('csv')">下载指标 CSV</button><RouterLink v-if="run.inputConfig" :to="{path:'/experiments/new',query:{rerun:run.id}}" class="button ghost">按此配置重跑</RouterLink><RouterLink to="/experiments/new" class="button green-button">新建实验</RouterLink></div>
    </header>

    <section class="detail-metrics result-metric-strip">
      <div v-for="[metric, value] in metricEntries" :key="metric"><span>{{ metricLabel(metric) }}</span><strong>{{ Number(value).toFixed(3) }}</strong><small class="metric-uncertainty">{{ run.evaluation && run.evaluation.total_runs > 1 ? `± ${metricDeviation(metric).toFixed(3)} · ` : '' }}{{ run.taskType ?? 'classification' }}</small></div>
      <div><span>总模型计算耗时</span><strong>{{ run.duration.toFixed(2) }}s</strong><small>训练 {{ (run.trainingTime ?? run.duration).toFixed(2) }}s · 预测 {{ (run.inferenceTime ?? 0).toFixed(2) }}s</small><small v-if="run.wallDuration!=null">含排队与图表共 {{ run.wallDuration.toFixed(2) }}s</small></div>
    </section>

    <p v-if="storageNotice" role="status" class="evidence-provenance">{{ storageNotice }}</p>
    <p v-if="!run.visualization" class="evidence-provenance">此记录目前只有指标摘要。完整图表尚在加载或未成功保存；可按原配置重新运行。</p>

    <ResultVisualizations :visualization="run.visualization" :task-type="run.taskType" />
    <AlgorithmGuide :model="run.model" :task-type="run.taskType" :metrics="run.metrics" :visualization="run.visualization" />
    <details v-if="run.evaluation?.per_run?.length" class="evaluation-details"><summary>查看 {{ run.evaluation.total_runs }} 次运行的逐种子 / 逐折明细</summary><div class="benchmark-matrix"><table class="benchmark-table"><thead><tr><th>种子</th><th>折</th><th v-for="[key] in metricEntries" :key="key">{{ metricName(key) }}</th></tr></thead><tbody><tr v-for="(row,i) in run.evaluation.per_run" :key="i"><td>{{ row.seed }}</td><td>{{ row.fold }}</td><td v-for="[key] in metricEntries" :key="key">{{ Number(Array.isArray(row.metrics[key]) ? (row.metrics[key] as number[]).at(-1) : row.metrics[key]).toFixed(4) }}</td></tr></tbody></table></div></details>

    <section class="result-config-card"><div><span class="section-kicker">REPRODUCIBILITY</span><h2>本次实验配置</h2><p>结果中的主数值是多次训练的平均值；“±”后面是标准差，越小说明模型越稳定。</p></div><pre><code>{{ configText }}</code></pre></section>
    </template>
    <section v-else-if="isHydrating" class="missing-run-state" role="status"><h1>正在读取实验记录…</h1></section>
    <section v-else class="missing-run-state">
      <span class="section-kicker">EXPERIMENT NOT FOUND</span>
      <h1>没有找到这条实验记录</h1>
      <p>它可能来自其他浏览器、已被清理，或链接编号不正确。请返回运行记录重新选择实验。</p>
      <RouterLink to="/runs" class="button primary">返回运行记录</RouterLink>
    </section>
  </div>
</template>
