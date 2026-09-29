<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ visualization?: Record<string, any>, taskType?: string }>()
const visual = computed(() => props.visualization ?? {})
const palette = ['#268b65', '#3478bd', '#d87a46', '#8d5ca8', '#bd5470', '#a48631', '#538a8b', '#6b748d', '#bd6b2f', '#497251']

const classification = computed(() => visual.value.classification)
const regression = computed(() => visual.value.regression)
const clustering = computed(() => visual.value.clustering)
const pca = computed(() => visual.value.pca)
const contributions = computed(() => visual.value.feature_contributions)
const loss = computed<number[]>(() => visual.value.loss_curve ?? [])
const roc = computed(() => classification.value?.roc_curve)

function maxOf(values: number[]) { return Math.max(...values, 1) }
function heat(value: number) {
  const matrix = classification.value?.confusion_matrix?.flat?.() ?? [1]
  return `rgba(38, 139, 101, ${.12 + .78 * value / maxOf(matrix)})`
}
function bounds(points: any[], key: 'x' | 'y') {
  const values = points.map((point) => Number(point[key]))
  const min = Math.min(...values, 0), max = Math.max(...values, 1)
  return [min, max === min ? min + 1 : max]
}
function plot(value: number, min: number, max: number, start: number, span: number, flip = false) {
  const ratio = (value - min) / (max - min)
  return +(start + (flip ? 1 - ratio : ratio) * span).toFixed(2)
}
const classBounds = computed(() => classification.value ? {
  x: bounds(classification.value.points, 'x'), y: bounds(classification.value.points, 'y'),
} : null)
const clusterBounds = computed(() => clustering.value ? {
  x: bounds(clustering.value.points, 'x'), y: bounds(clustering.value.points, 'y'),
} : null)
const pcaBounds = computed(() => pca.value ? {
  x: bounds(pca.value.points, 'x'), y: bounds(pca.value.points, 'y'),
} : null)
const regressionBounds = computed(() => {
  if (!regression.value?.pairs?.length) return null
  const all = regression.value.pairs.flatMap((item: any) => [Number(item.actual), Number(item.predicted)])
  const min = Math.min(...all), max = Math.max(...all)
  return [min, max === min ? min + 1 : max]
})
const lossPoints = computed(() => {
  if (loss.value.length < 2) return ''
  const min = Math.min(...loss.value), max = Math.max(...loss.value)
  return loss.value.map((value, index) => `${plot(index, 0, loss.value.length - 1, 12, 236)},${plot(value, min, max, 8, 98, true)}`).join(' ')
})
const rocPoints = computed(() => roc.value?.points?.map((point: any) => `${28 + 252 * point.fpr},${126 - 114 * point.tpr}`).join(' ') ?? '')
</script>

<template>
  <section class="visual-suite" v-if="Object.keys(visual).length">
    <div class="visual-suite-head">
      <div><span class="section-kicker">RESULT EXPLORER</span><h2>从数据到结论</h2></div>
      <p>所有图表均来自本次真实训练与测试数据。</p>
    </div>

    <div v-if="classification" class="visual-grid">
      <article class="chart-card heatmap-card">
        <div class="chart-title"><b>混淆矩阵</b><small>行是真实类别，列是预测类别</small></div>
        <div class="heatmap-labels" :style="{ gridTemplateColumns: `90px repeat(${classification.labels.length}, minmax(38px, 1fr))` }"><span></span><span v-for="label in classification.labels" :key="label">{{ label }}</span></div>
        <div v-for="(row, rowIndex) in classification.confusion_matrix" :key="rowIndex" class="heatmap-row" :style="{ gridTemplateColumns: `90px repeat(${classification.labels.length}, minmax(38px, 1fr))` }">
          <small>{{ classification.labels[rowIndex] }}</small>
          <span v-for="(value, index) in row" :key="index" class="heat-cell" :style="{ background: heat(value), color: value / maxOf(classification.confusion_matrix.flat()) > .5 ? '#fff' : '#234438' }">{{ value }}</span>
        </div>
      </article>
      <article class="chart-card">
        <div class="chart-title"><b>类别分布</b><small>测试集真实值 vs 模型预测</small></div>
        <div v-for="(label, index) in classification.labels" :key="label" class="paired-bar">
          <span>{{ label }}</span><div><i :style="{ width: `${100 * classification.actual_distribution[index] / maxOf([...classification.actual_distribution, ...classification.predicted_distribution])}%` }"></i><em :style="{ width: `${100 * classification.predicted_distribution[index] / maxOf([...classification.actual_distribution, ...classification.predicted_distribution])}%` }"></em></div><small>{{ classification.actual_distribution[index] }} / {{ classification.predicted_distribution[index] }}</small>
        </div>
        <p class="chart-legend"><i></i>真实样本 <em></em>预测样本</p>
      </article>
      <article class="chart-card wide-chart">
        <div class="chart-title"><b>测试集二维投影</b><small>颜色代表真实类别，空心点代表误判</small></div>
        <svg class="scatter-chart" viewBox="0 0 300 155">
          <path d="M24 8V130H288" class="axis" />
          <circle v-for="(point, index) in classification.points" :key="index" :cx="plot(point.x, classBounds!.x[0], classBounds!.x[1], 28, 252)" :cy="plot(point.y, classBounds!.y[0], classBounds!.y[1], 12, 114, true)" r="3.6" :fill="point.actual === point.predicted ? palette[point.actual % palette.length] : '#fff'" :stroke="palette[point.actual % palette.length]" :stroke-width="point.actual === point.predicted ? 0 : 2" />
        </svg>
      </article>
      <article v-if="roc" class="chart-card">
        <div class="chart-title"><b>ROC 曲线</b><small>AUC = {{ roc.auc }}，越靠左上越好</small></div>
        <svg class="scatter-chart" viewBox="0 0 300 155"><path d="M24 8V130H288" class="axis" /><path d="M28 126L280 12" class="reference-line" /><polyline :points="rocPoints" class="roc-line" /></svg>
      </article>
    </div>

    <div v-if="regression" class="visual-grid">
      <article class="chart-card wide-chart">
        <div class="chart-title"><b>真实值与预测值</b><small>点越接近虚线，回归拟合越准确</small></div>
        <svg class="scatter-chart" viewBox="0 0 300 155">
          <path d="M24 8V130H288" class="axis" /><path d="M28 126L280 12" class="reference-line" />
          <circle v-for="(pair, index) in regression.pairs" :key="index" :cx="plot(pair.actual, regressionBounds![0], regressionBounds![1], 28, 252)" :cy="plot(pair.predicted, regressionBounds![0], regressionBounds![1], 12, 114, true)" r="2.8" fill="#268b65" fill-opacity=".68" />
        </svg>
      </article>
      <article class="chart-card">
        <div class="chart-title"><b>残差分布</b><small>理想情况应围绕 0 分散</small></div>
        <div class="histogram"><i v-for="(count, index) in regression.residual_histogram.counts" :key="index" :style="{ height: `${100 * count / maxOf(regression.residual_histogram.counts)}%` }"></i></div>
      </article>
    </div>

    <div v-if="clustering" class="visual-grid">
      <article class="chart-card wide-chart">
        <div class="chart-title"><b>聚类二维投影</b><small>仅用于观察；训练过程未使用真实标签</small></div>
        <svg class="scatter-chart" viewBox="0 0 300 155"><path d="M24 8V130H288" class="axis" />
          <circle v-for="(point, index) in clustering.points" :key="index" :cx="plot(point.x, clusterBounds!.x[0], clusterBounds!.x[1], 28, 252)" :cy="plot(point.y, clusterBounds!.y[0], clusterBounds!.y[1], 12, 114, true)" r="3" :fill="palette[point.cluster % palette.length]" fill-opacity=".7" />
        </svg>
      </article>
      <article class="chart-card"><div class="chart-title"><b>簇规模</b><small>每个簇包含的训练样本数</small></div><div class="vertical-bars"><div v-for="(size, index) in clustering.sizes" :key="index"><i :style="{ height: `${100 * Number(size) / maxOf(clustering.sizes.map(Number))}%`, background: palette[Number(index) % palette.length] }"></i><span>C{{ Number(index) + 1 }}</span><small>{{ size }}</small></div></div></article>
    </div>

    <div v-if="pca" class="visual-grid">
      <article class="chart-card"><div class="chart-title"><b>解释方差谱</b><small>每个主成分保留的信息比例</small></div><div class="vertical-bars"><div v-for="(value, index) in pca.explained_variance_ratio.slice(0, 10)" :key="index"><i :style="{ height: `${Number(value) * 100}%` }"></i><span>PC{{ Number(index) + 1 }}</span><small>{{ (Number(value) * 100).toFixed(0) }}%</small></div></div></article>
      <article class="chart-card wide-chart"><div class="chart-title"><b>PCA 投影</b><small>样本压缩到前两个主成分后的分布</small></div><svg class="scatter-chart" viewBox="0 0 300 155"><path d="M24 8V130H288" class="axis" /><circle v-for="(point, index) in pca.points" :key="index" :cx="plot(point.x, pcaBounds!.x[0], pcaBounds!.x[1], 28, 252)" :cy="plot(point.y, pcaBounds!.y[0], pcaBounds!.y[1], 12, 114, true)" r="3.2" :fill="palette[point.label % palette.length]" /></svg></article>
    </div>

    <div class="visual-grid secondary-visuals" v-if="contributions || loss.length">
      <article v-if="contributions" class="chart-card wide-chart"><div class="chart-title"><b>特征贡献</b><small>{{ contributions.label }} · 仅显示模型真实可获得的特征信息</small></div><div class="feature-bars"><div v-for="item in contributions.items" :key="item.name"><span>{{ item.name }}</span><i><em :style="{ width: `${100 * item.value / maxOf(contributions.items.map((entry: any) => entry.value))}%` }"></em></i><small>{{ item.value.toFixed(3) }}</small></div></div></article>
      <article v-if="loss.length" class="chart-card"><div class="chart-title"><b>训练损失曲线</b><small>损失下降并趋稳，说明训练逐步收敛</small></div><svg class="loss-chart" viewBox="0 0 260 116"><path d="M12 8V106H248" class="axis" /><polyline :points="lossPoints" class="loss-line" /></svg></article>
    </div>
  </section>
</template>
