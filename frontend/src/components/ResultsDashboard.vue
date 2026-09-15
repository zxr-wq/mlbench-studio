<script setup>
import { computed, ref, watch } from 'vue'
import ConfusionMatrix from './ConfusionMatrix.vue'
import ScatterPlot from './ScatterPlot.vue'
import TrainingCurve from './TrainingCurve.vue'
import VisualizationPanel from './VisualizationPanel.vue'

const props = defineProps({
  experiment: { type: Object, default: null },
  algorithms: { type: Array, default: () => [] },
})

const activeResultId = ref('')
const metricCatalog = {
  accuracy: '准确率', precision: '宏平均精确率', recall: '宏平均召回率', f1: '宏平均 F1',
  mse: 'MSE', rmse: 'RMSE', mae: 'MAE', r2: 'R²', silhouette_score: 'Silhouette', inertia: 'Inertia',
  cumulative_explained_variance: '累计解释方差',
}
const percentageMetrics = new Set(['accuracy', 'precision', 'recall', 'f1', 'cumulative_explained_variance'])
const lowerBetterMetrics = new Set(['mse', 'rmse', 'mae', 'inertia'])

const results = computed(() => props.experiment?.results || [])
const activeResult = computed(() => results.value.find((item) => item.id === activeResultId.value) || results.value[0] || null)
const taskType = computed(() => activeResult.value?.task_type || 'classification')
const primaryMetric = computed(() => ({ classification: 'accuracy', regression: 'r2', clustering: 'silhouette_score', dimensionality_reduction: 'cumulative_explained_variance' }[taskType.value] || Object.keys(activeResult.value?.metrics || {})[0]))
const bestResult = computed(() => {
  if (!results.value.length) return null
  const metric = primaryMetric.value
  return [...results.value].sort((a, b) => Number(b.metrics?.[metric] ?? -Infinity) - Number(a.metrics?.[metric] ?? -Infinity))[0]
})
const configLabel = computed(() => {
  const config = props.experiment?.config
  if (!config) return '尚未运行'
  if (config.split_method === 'kfold') return `${config.folds} 折交叉验证`
  return `${Math.round(config.test_size * 100)}% 测试集`
})
const metricNames = computed(() => {
  const available = [...new Set(results.value.flatMap((item) => Object.keys(item.metrics || {})))]
  const configured = props.experiment?.config?.metrics || []
  const selected = configured.filter((metric) => available.includes(metric))
  const finalMetrics = selected.length ? selected : available
  return Object.fromEntries(finalMetrics.map((metric) => [metric, metricCatalog[metric] || metric]).filter(([, label]) => label))
})
const visualizationTitle = computed(() => ({
  knn_neighbors: 'K 个最近邻',
  naive_bayes_stats: '类别先验与高斯统计量',
  decision_tree: '决策树结构',
  tree: '决策树结构',
  svm_boundary: '决策边界与支持向量',
  decision_boundary: '决策边界',
  kmeans_clusters: '聚类与质心迁移',
  centroid_history: '质心迭代路径',
  pca_projection: '主成分投影',
  pca: '主成分投影',
  loss_history: '迭代损失',
  gradient_boosting_loss: 'Gradient Boosting 损失',
}[activeResult.value?.visualization?.type] || '算法运行过程'))

watch(results, (items) => {
  if (items.length && !items.some((item) => item.id === activeResultId.value)) activeResultId.value = items[0].id
}, { immediate: true })

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`
}

function formatMetric(metric, value) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  if (percentageMetrics.has(metric)) return percent(value)
  return Number(value).toFixed(4)
}

function metricWidth(metric, value) {
  const values = results.value.map((item) => Number(item.metrics?.[metric])).filter(Number.isFinite)
  if (!values.length) return '0%'
  const min = Math.min(...values)
  const max = Math.max(...values)
  if (max === min) return '100%'
  const normalized = (Number(value) - min) / (max - min)
  return `${(lowerBetterMetrics.has(metric) ? 1 - normalized : normalized) * 70 + 30}%`
}

function formatTime(ms) {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}
</script>

<template>
<section class="results-panel">
  <div v-if="!experiment" class="empty-dashboard paper-panel">
    <div class="empty-visual" aria-hidden="true">
      <span class="orbit one"></span><span class="orbit two"></span><span class="orbit three"></span>
      <b>ML</b>
    </div>
    <span class="section-number">RESULTS / 02</span>
    <h2>让数据替你回答</h2>
    <p>配置数据集与算法后开始实验。训练指标、模型差异与误判样本会在这里实时展开。</p>
    <div class="empty-steps">
      <span><b>01</b> 选择数据</span><i></i><span><b>02</b> 并行比较</span><i></i><span><b>03</b> 观察结论</span>
    </div>
  </div>

  <template v-else>
    <div class="run-status paper-panel" :class="experiment.status">
      <div class="run-status-copy">
        <span class="live-badge"><i></i>{{ experiment.status === 'completed' ? '实验完成' : experiment.status === 'failed' ? '运行异常' : 'LIVE EXPERIMENT' }}</span>
        <h2>{{ experiment.stage }}</h2>
        <p v-if="experiment.status !== 'failed'">{{ results.length }} / {{ experiment.run_plan?.length || experiment.config?.model_ids?.length || 0 }} 项运行已有结果</p>
        <p v-else>{{ experiment.error }}</p>
      </div>
      <div class="progress-orb" :style="{ '--progress': `${experiment.progress || 0}%` }">
        <span>{{ experiment.progress || 0 }}<small>%</small></span>
      </div>
      <div class="progress-track"><i :style="{ width: `${experiment.progress || 0}%` }"></i></div>
    </div>

    <div v-if="results.length" class="metric-strip">
      <article class="metric-card featured">
        <span>最佳{{ metricCatalog[primaryMetric] || primaryMetric }}</span>
        <strong>{{ formatMetric(primaryMetric, bestResult.metrics[primaryMetric]) }}</strong>
        <small>{{ bestResult.short_name || bestResult.model }} 当前领先</small>
      </article>
      <article class="metric-card">
        <span>已评估模型</span>
        <strong>{{ results.length }}<em>/{{ experiment.run_plan?.length || experiment.config.model_ids.length }}</em></strong>
        <small>Scratch {{ results.filter((item) => item.implementation === 'scratch').length }} · Sklearn {{ results.filter((item) => item.implementation === 'sklearn').length }}</small>
      </article>
      <article class="metric-card">
        <span>验证策略</span>
        <strong class="text-metric">{{ configLabel }}</strong>
        <small>随机种子 {{ experiment.config.seed }}</small>
      </article>
      <article class="metric-card">
        <span>数据规模</span>
        <strong class="text-metric">{{ experiment.dataset?.samples || '—' }} × {{ experiment.dataset?.features || '—' }}</strong>
        <small>{{ experiment.dataset?.name || '计算完成后显示' }}</small>
      </article>
    </div>

    <div v-if="results.length" class="comparison-card paper-panel">
      <div class="card-heading">
        <div><span class="section-number">COMPARE / 03</span><h3>横向性能比较</h3></div>
        <span class="legend-note">按任务类型动态展示指标</span>
      </div>
      <div class="comparison-head" :style="{ gridTemplateColumns: `1.35fr repeat(${Object.keys(metricNames).length}, 1fr) .65fr` }"><span>模型</span><span v-for="label in metricNames" :key="label">{{ label.replace('宏平均', '') }}</span><span>耗时</span></div>
      <button
        v-for="result in results"
        :key="result.id"
        class="comparison-row"
        :class="{ active: activeResult?.id === result.id }"
        :style="{ gridTemplateColumns: `1.35fr repeat(${Object.keys(metricNames).length}, 1fr) .65fr` }"
        @click="activeResultId = result.id"
      >
        <span class="model-name"><i></i><b>{{ result.short_name }}</b><small>{{ result.implementation === 'sklearn' ? 'Sklearn 对照' : 'Scratch 自实现' }}</small></span>
        <span v-for="(_, metric) in metricNames" :key="metric" class="bar-cell">
          <i :style="{ width: metricWidth(metric, result.metrics[metric]) }"></i><em>{{ formatMetric(metric, result.metrics[metric]) }}</em>
        </span>
        <span class="time-cell">{{ formatTime(result.training_ms) }}<small> + {{ formatTime(result.inference_ms || 0) }}</small></span>
      </button>
    </div>

    <div v-if="activeResult" class="detail-card paper-panel">
      <div class="card-heading result-heading">
        <div>
          <span class="section-number">INSPECT / 04</span>
          <h3>{{ activeResult.name }} · 诊断视图</h3>
        </div>
        <div class="result-tabs">
          <button v-for="result in results" :key="result.id" :class="{ active: activeResult.id === result.id }" @click="activeResultId = result.id">{{ result.short_name }} · {{ result.implementation === 'sklearn' ? 'SK' : 'S' }}</button>
        </div>
      </div>

      <div class="diagnostic-grid">
        <article v-if="activeResult.projection?.length" class="viz-card wide">
          <div class="viz-heading"><div><span>样本空间</span><h4>PCA 二维投影</h4></div><small>黑圈代表误判</small></div>
          <ScatterPlot :points="activeResult.projection" :class-names="activeResult.class_names" />
        </article>
        <article v-if="activeResult.confusion_matrix?.length" class="viz-card matrix-card">
          <div class="viz-heading"><div><span>分类诊断</span><h4>混淆矩阵</h4></div><small>{{ percent(activeResult.metrics.accuracy) }} accuracy</small></div>
          <ConfusionMatrix :matrix="activeResult.confusion_matrix" :class-names="activeResult.class_names" />
        </article>

        <article v-if="activeResult.training_curve?.length" class="viz-card">
          <div class="viz-heading"><div><span>优化过程</span><h4>训练损失</h4></div><small>末轮 {{ activeResult.training_curve.at(-1)?.loss.toFixed(4) }}</small></div>
          <TrainingCurve :points="activeResult.training_curve" />
        </article>
        <article v-else class="viz-card insight-card">
          <div class="viz-heading"><div><span>模型特性</span><h4>非迭代式学习</h4></div></div>
          <p>该算法直接由训练样本或统计量形成决策规则，因此没有逐轮下降的损失曲线。</p>
          <div class="insight-number"><strong>{{ formatTime((activeResult.training_ms || 0) + (activeResult.inference_ms || 0)) }}</strong><small>训练与推理总耗时</small></div>
        </article>

        <article v-if="Object.keys(activeResult.visualization || {}).length" class="viz-card algorithm-viz">
          <div class="viz-heading"><div><span>统一可视化接口</span><h4>{{ visualizationTitle }}</h4></div><small>get_visualization_data</small></div>
          <VisualizationPanel :visualization="activeResult.visualization" />
        </article>

        <article v-if="activeResult.comparison" class="viz-card comparison-summary">
          <div class="viz-heading"><div><span>正确性验证</span><h4>Scratch 与 Sklearn</h4></div></div>
          <div><span><small>准确率差</small><strong>{{ percent(activeResult.comparison.accuracy_difference) }}</strong></span><span><small>预测一致率</small><strong>{{ percent(activeResult.comparison.prediction_agreement) }}</strong></span></div>
        </article>

        <article v-if="activeResult.feature_importance?.length" class="viz-card">
          <div class="viz-heading"><div><span>解释线索</span><h4>特征贡献 Top 8</h4></div><small v-if="!activeResult.feature_importance.length">当前模型不提供</small></div>
          <div class="importance-list">
            <div v-for="item in activeResult.feature_importance" :key="item.feature">
              <span :title="item.feature">{{ item.feature }}</span>
              <i><b :style="{ width: `${item.value / activeResult.feature_importance[0].value * 100}%` }"></b></i>
              <em>{{ item.value.toFixed(3) }}</em>
            </div>
          </div>
        </article>

        <article class="viz-card config-summary">
          <div class="viz-heading"><div><span>可复现性</span><h4>运行参数</h4></div><small>{{ activeResult.implementation }}</small></div>
          <div class="parameter-copy"><b v-for="(value, key) in (activeResult.params || activeResult.parameters || {})" :key="key">{{ key }} = {{ value }}</b></div>
          <p>训练 {{ formatTime(activeResult.training_ms || 0) }} · 推理 {{ formatTime(activeResult.inference_ms || 0) }}</p>
        </article>
      </div>
    </div>

    <div v-else-if="experiment.status !== 'failed'" class="waiting-card paper-panel">
      <span class="loader"></span><h3>模型正在学习</h3><p>首个模型完成后，将立即显示对比结果。</p>
    </div>
  </template>
</section>
</template>
