<script setup>
import { computed } from 'vue'
import { aBenchmarkRows, knnParameterSeries } from '../data/aBenchmark'

const props = defineProps({ history: { type: Array, default: () => [] } })
const completedRuns = computed(() => props.history.filter((item) => item.status === 'completed').length)
const maxInference = Math.max(...aBenchmarkRows.flatMap((row) => [row.scratchPredict, row.sklearnPredict]))

function percent(value) {
  return `${(value * 100).toFixed(2)}%`
}

function line(values) {
  const min = Math.min(...values.map((item) => item.accuracy))
  const max = Math.max(...values.map((item) => item.accuracy))
  const span = Math.max(max - min, .01)
  return values.map((item, index) => {
    const x = 24 + index * (352 / Math.max(1, values.length - 1))
    const y = 18 + (1 - (item.accuracy - min) / span) * 92
    return `${index ? 'L' : 'M'} ${x} ${y}`
  }).join(' ')
}
</script>

<template>
  <section class="content-view benchmark-view">
    <div class="view-intro">
      <div><span class="section-number">BENCHMARK</span><h2>同一把尺子，验证每个实现</h2></div>
      <p>固定 80/20 分层划分、seed 42，并仅在训练集拟合 StandardScaler。当前展示 A 负责算法的正式基线。</p>
    </div>

    <div class="benchmark-summary">
      <article class="paper-panel"><span>已完成实验</span><strong>{{ completedRuns }}</strong><small>当前进程历史</small></article>
      <article class="paper-panel"><span>正式数据集</span><strong>3</strong><small>Iris · Wine · Breast Cancer</small></article>
      <article class="paper-panel"><span>预测一致率</span><strong>100%</strong><small>Scratch vs sklearn</small></article>
      <article class="paper-panel"><span>最大准确率差</span><strong>0.0000</strong><small>固定划分结果</small></article>
    </div>

    <div class="benchmark-table paper-panel">
      <div class="benchmark-title"><div><span class="section-number">REFERENCE CHECK</span><h3>Scratch 与 sklearn 正确性对照</h3></div><small>训练与推理分开计时 · ms</small></div>
      <div class="benchmark-head"><span>数据集 / 算法</span><span>Scratch</span><span>sklearn</span><span>一致率</span><span>Scratch 推理</span><span>sklearn 推理</span></div>
      <div v-for="row in aBenchmarkRows" :key="`${row.dataset}-${row.model}`" class="benchmark-row">
        <span><b>{{ row.model }}</b><small>{{ row.dataset }}</small></span>
        <span>{{ percent(row.scratch) }}</span>
        <span>{{ percent(row.sklearn) }}</span>
        <span class="agreement-value">{{ percent(row.agreement) }}</span>
        <span class="runtime-bar"><i :style="{ width: `${row.scratchPredict / maxInference * 100}%` }"></i><em>{{ row.scratchPredict.toFixed(3) }}</em></span>
        <span class="runtime-bar sklearn"><i :style="{ width: `${row.sklearnPredict / maxInference * 100}%` }"></i><em>{{ row.sklearnPredict.toFixed(3) }}</em></span>
      </div>
    </div>

    <div class="parameter-benchmark paper-panel">
      <div class="benchmark-title"><div><span class="section-number">PARAMETER STUDY</span><h3>K 值敏感性</h3></div><small>Accuracy · 距离加权</small></div>
      <div class="parameter-chart-grid">
        <article v-for="series in knnParameterSeries" :key="series.dataset">
          <h4>{{ series.dataset }}</h4>
          <svg viewBox="0 0 400 130" role="img" :aria-label="`${series.dataset} KNN K 值准确率曲线`">
            <line v-for="n in 4" :key="n" x1="24" x2="376" :y1="n * 27" :y2="n * 27" stroke="#ded8ce" />
            <path :d="line(series.values)" fill="none" stroke="#7157d9" stroke-width="3" stroke-linejoin="round" />
            <circle v-for="(point, index) in series.values" :key="point.k" :cx="24 + index * (352 / 6)" :cy="18 + (1 - (point.accuracy - Math.min(...series.values.map((item) => item.accuracy))) / Math.max(Math.max(...series.values.map((item) => item.accuracy)) - Math.min(...series.values.map((item) => item.accuracy)), .01)) * 92" r="4" fill="#7157d9"><title>K={{ point.k }} · {{ percent(point.accuracy) }}</title></circle>
          </svg>
          <div class="k-labels"><span v-for="point in series.values" :key="point.k">{{ point.k }}</span></div>
        </article>
      </div>
    </div>
  </section>
</template>
