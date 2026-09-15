<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import { useLabStore } from '../stores/lab'

const route = useRoute()
const { runs } = useLabStore()
const run = computed(() => runs.value.find((item) => item.id === route.params.id) ?? runs.value[0])
const detail = computed(() => run.value?.detail)
const splitLabel = computed(() =>
  detail.value?.config.split === 'kfold_5' ? 'Stratified 5-Fold CV' : 'Stratified 80-20',
)

const maxConfusion = computed(() => {
  const matrix = detail.value?.confusionMatrix ?? []
  return Math.max(1, ...matrix.flat())
})

function heatStyle(value: number) {
  const intensity = Math.sqrt(value / maxConfusion.value)
  return {
    background: `rgba(232, 93, 63, ${0.08 + intensity * 0.72})`,
    color: intensity > 0.55 ? '#fff' : '#5a5c58',
  }
}

const configYaml = computed(() => {
  const config = detail.value?.config
  if (!config) return null
  return `dataset: ${config.dataset}
splitter: ${config.split}
preprocessing: ${config.preprocessing}
model:
  name: ${config.model}
  kernel: ${config.kernel}
  C: ${config.c.toFixed(1)}
metrics: [${config.metrics.join(', ')}]
seed: ${config.seed}`
})
</script>

<template>
  <div class="page detail-page">
    <RouterLink to="/runs" class="back-link">← 返回运行记录</RouterLink>
    <header class="run-detail-header">
      <div>
        <span class="status-label"><i></i>{{ run.status === 'completed' ? '运行成功' : run.status }}</span>
        <h1>{{ run.id }} · {{ run.model }}</h1>
        <p>{{ run.dataset }} / {{ splitLabel }} / seed {{ detail?.config.seed ?? 42 }}</p>
      </div>
      <RouterLink to="/experiments/new" class="button ghost">复制为新实验</RouterLink>
    </header>

    <section class="detail-metrics">
      <div><span>Accuracy</span><strong>{{ run.accuracy.toFixed(3) }}</strong><small>{{ splitLabel }}</small></div>
      <div><span>Macro F1</span><strong>{{ run.f1.toFixed(3) }}</strong><small>全类别平均</small></div>
      <div><span>训练耗时</span><strong>{{ run.duration.toFixed(2) }}s</strong><small>浏览器本地 CPU</small></div>
      <div>
        <span v-if="detail?.silhouette !== undefined">Silhouette</span>
        <span v-else-if="detail?.pcaComponents !== undefined">PCA 维度</span>
        <span v-else>可复现</span>
        <strong v-if="detail?.silhouette !== undefined" class="yes-value">{{ detail.silhouette.toFixed(3) }}</strong>
        <strong v-else-if="detail?.pcaComponents !== undefined" class="yes-value">{{ detail.pcaComponents }}</strong>
        <strong v-else class="yes-value"><AppIcon name="check" :size="24" /> Yes</strong>
        <small v-if="detail?.silhouette !== undefined">聚类结构质量（越高越好）</small>
        <small v-else-if="detail?.pcaComponents !== undefined">PCA 降维后特征维度</small>
        <small v-else>配置与随机种子已保存</small>
      </div>
    </section>

    <div class="detail-grid">
      <section class="content-section">
        <div class="section-title-row"><div><span class="overline">EVALUATION</span><h2>类别表现</h2></div></div>

        <template v-if="detail">
          <div class="confusion-wrap">
            <div>
              <div class="confusion-axis"><span>对角线 = 预测正确</span><span>行 = 真实 · 列 = 预测</span></div>
              <div class="confusion-grid dynamic" :style="{ gridTemplateColumns: `repeat(${detail.confusionMatrix.length}, minmax(0, 1fr))` }">
                <template v-for="(row, r) in detail.confusionMatrix" :key="`row-${r}`">
                  <span v-for="(value, c) in row" :key="`cell-${r}-${c}`" class="heat dynamic-heat" :style="heatStyle(value)">{{ value }}</span>
                </template>
              </div>
              <div class="confusion-legend">
                <span v-for="label in detail.classLabels" :key="label">{{ label }}</span>
              </div>
            </div>
            <div class="confusion-copy">
              <span>预测结果矩阵</span>
              <p>{{ run.model }} 在测试集上的混淆矩阵，越深的格子表示样本越多。</p>
            </div>
          </div>

          <div class="perclass-sheet">
            <div class="perclass-row perclass-head">
              <span>类别</span><span>Precision</span><span>Recall</span><span>F1</span><span>样本数</span>
            </div>
            <div v-for="metric in detail.perClass" :key="metric.label" class="perclass-row">
              <span><b>{{ metric.label }}</b></span>
              <span>{{ metric.precision.toFixed(3) }}</span>
              <span>{{ metric.recall.toFixed(3) }}</span>
              <span><b>{{ metric.f1.toFixed(3) }}</b></span>
              <span>{{ metric.support }}</span>
            </div>
          </div>
        </template>

        <div v-else class="empty-state">该记录来自历史模拟数据，缺少详细评估，重新运行一次实验即可查看混淆矩阵。</div>
      </section>

      <aside class="content-section config-code">
        <div class="section-title-row"><div><span class="overline">RUN CONFIG</span><h2>实验配置</h2></div></div>
        <pre><code>{{ configYaml ?? `dataset: ${run.dataset}
splitter: stratified_80_20
preprocessing: standard_scaler
model:
  name: ${run.model.toLowerCase()}
  kernel: rbf
  C: 1.0
metrics: [accuracy, macro_f1]
seed: 42` }}</code></pre>
      </aside>
    </div>
  </div>
</template>
