<script setup lang="ts">
import { computed, reactive, ref } from 'vue'
import AppIcon from '../components/AppIcon.vue'
import { datasets } from '../data/catalog'
import { loadDataset } from '../ml/data'
import { fitPca } from '../ml/pca'
import { runPipeline } from '../ml/pipeline'
import { StandardScaler } from '../ml/preprocess'
import { benchmarkModels, specToConfig } from '../ml/registry'
import type { ExperimentConfig } from '../types/benchmark'

interface CellState {
  status: 'idle' | 'running' | 'done' | 'failed'
  accuracy?: number
  f1?: number
  duration?: number
}

type MatrixState = Record<string, Record<string, CellState>>

const matrix = reactive<MatrixState>({})
benchmarkModels.forEach((spec) => {
  matrix[spec.id] = {}
  datasets.forEach((dataset) => {
    matrix[spec.id][dataset.id] = { status: 'idle' }
  })
})

const totalCombinations = benchmarkModels.length * datasets.length
const doneCount = computed(() =>
  benchmarkModels.reduce(
    (sum, spec) => sum + datasets.filter((d) => matrix[spec.id][d.id].status === 'done').length,
    0,
  ),
)

const isRunningAll = ref(false)
const currentTask = ref('')
const runProgress = ref(0)

const bestCell = computed(() => {
  let best: { label: string; dataset: string; accuracy: number } | null = null
  for (const spec of benchmarkModels) {
    for (const dataset of datasets) {
      const cell = matrix[spec.id][dataset.id]
      if (cell.status === 'done' && cell.accuracy !== undefined) {
        if (!best || cell.accuracy > best.accuracy) {
          best = { label: spec.label, dataset: dataset.name, accuracy: cell.accuracy }
        }
      }
    }
  }
  return best
})

const averageDuration = computed(() => {
  const durations: number[] = []
  for (const spec of benchmarkModels) {
    for (const dataset of datasets) {
      const cell = matrix[spec.id][dataset.id]
      if (cell.status === 'done' && cell.duration !== undefined) durations.push(cell.duration)
    }
  }
  if (!durations.length) return null
  return durations.reduce((sum, value) => sum + value, 0) / durations.length
})

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

async function runPipelineFromDisk(config: ExperimentConfig, datasetId: string) {
  const bundle = await loadDataset(datasetId)
  return runPipeline(bundle, config)
}

async function runAll() {
  if (isRunningAll.value) return
  isRunningAll.value = true
  runProgress.value = 0
  let finished = 0
  try {
    for (const spec of benchmarkModels) {
      for (const dataset of datasets) {
        currentTask.value = `${spec.label} × ${dataset.name}`
        const cell = matrix[spec.id][dataset.id]
        cell.status = 'running'
        await sleep(24)
        try {
          const output = await runPipelineFromDisk(specToConfig(spec, dataset.id), dataset.id)
          cell.accuracy = output.accuracy
          cell.f1 = output.macroF1
          cell.duration = output.duration
          cell.status = 'done'
        } catch {
          cell.status = 'failed'
        }
        finished += 1
        runProgress.value = Math.round((finished / totalCombinations) * 100)
      }
    }
  } finally {
    isRunningAll.value = false
    currentTask.value = ''
  }
}

function exportCsv() {
  const lines: string[] = ['model,' + datasets.map((d) => `${d.id}_accuracy,${d.id}_f1,${d.id}_duration_s`).join(',')]
  for (const spec of benchmarkModels) {
    const cells = datasets.map((dataset) => {
      const cell = matrix[spec.id][dataset.id]
      return cell.status === 'done'
        ? `${cell.accuracy?.toFixed(4)},${cell.f1?.toFixed(4)},${cell.duration?.toFixed(2)}`
        : ',,'
    })
    lines.push(`${spec.id},${cells.join(',')}`)
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'mlbench-benchmark.csv'
  link.click()
  URL.revokeObjectURL(url)
}

/* ---------- PCA 专项分析 ---------- */

const pcaDatasetId = ref('digits')
const pcaState = ref<'idle' | 'running' | 'done'>('idle')
const pcaMessage = ref('')
const explainedRatios = ref<number[]>([])
const pcaCurve = ref<{ components: number; accuracy: number; duration: number }[]>([])

const pcaChartBars = computed(() => explainedRatios.value.slice(0, 24))
const pcaBarScale = computed(() => Math.max(...pcaChartBars.value, 0.0001))
const pcaCumulative = computed(() => {
  let acc = 0
  return pcaChartBars.value.map((value) => {
    acc += value
    return acc
  })
})
const pcaTotalKept = computed(() => explainedRatios.value.reduce((sum, value) => sum + value, 0))

function componentLadder(dim: number): number[] {
  const ladder = [1, 2, 4, 8, 16, 32, 64, 128].filter((value) => value < dim)
  ladder.push(dim)
  return ladder
}

async function runPcaAnalysis() {
  if (pcaState.value === 'running') return
  pcaState.value = 'running'
  pcaMessage.value = '正在加载数据集'
  try {
    const bundle = await loadDataset(pcaDatasetId.value)
    const dim = bundle.data[0]?.length ?? 0

    pcaMessage.value = '正在计算方差解释率'
    const scaler = new StandardScaler()
    const standardized = scaler.fitTransform(bundle.data)
    const model = fitPca(standardized, { maxComponents: dim })
    explainedRatios.value = model.explainedVarianceRatio

    pcaCurve.value = []
    const ladder = componentLadder(dim)
    for (const components of ladder) {
      pcaMessage.value = `PCA 降维到 ${components} 维后训练 SVM`
      const config: ExperimentConfig = {
        dataset: pcaDatasetId.value,
        split: 'stratified_80_20',
        preprocessing: `pca_k_${components}`,
        model: 'svm',
        kernel: 'rbf',
        c: 1,
        metrics: ['accuracy'],
        seed: 42,
      }
      const output = await runPipelineFromDisk(config, pcaDatasetId.value)
      pcaCurve.value.push({ components, accuracy: output.accuracy, duration: output.duration })
    }
    pcaState.value = 'done'
  } catch {
    pcaState.value = 'idle'
    pcaMessage.value = ''
  }
}

/** Map accuracy ∈ [0.3, 1.0] onto the SVG height (46 units). */
function accuracyLine(point: { accuracy: number }): number {
  const y = 100 - (point.accuracy - 0.3) * (100 / 0.7)
  return Math.min(100, Math.max(0, y)) * 0.46
}
</script>

<template>
  <div class="page benchmark-page">
    <header class="page-header split-header">
      <div>
        <span class="overline">BENCHMARK / UNIFIED COMPARISON</span>
        <h1>统一实验对比</h1>
        <p>同一 Pipeline、同一种子下运行全部已实现算法，整理跨数据集的性能对比结果。</p>
      </div>
      <div class="benchmark-actions">
        <button class="button ghost" :disabled="isRunningAll || doneCount === 0" @click="exportCsv">
          <AppIcon name="dataset" :size="15" />导出 CSV
        </button>
        <button class="button primary" :disabled="isRunningAll" @click="runAll">
          <AppIcon name="flask" :size="15" />{{ isRunningAll ? '基准运行中' : '运行全部基准' }}
        </button>
      </div>
    </header>

    <section class="metric-strip benchmark-strip">
      <div>
        <span>对比组合</span>
        <strong>{{ totalCombinations }}</strong>
        <small>{{ benchmarkModels.length }} 模型 × {{ datasets.length }} 数据集</small>
      </div>
      <div>
        <span>已完成</span>
        <strong>{{ doneCount }}</strong>
        <small>{{ doneCount === totalCombinations ? '矩阵已填满' : '等待运行' }}</small>
      </div>
      <div>
        <span>最佳准确率</span>
        <strong>{{ bestCell ? `${(bestCell.accuracy * 100).toFixed(1)}%` : '—' }}</strong>
        <small>{{ bestCell ? `${bestCell.label} · ${bestCell.dataset}` : '运行基准后更新' }}</small>
      </div>
      <div>
        <span>平均耗时</span>
        <strong>{{ averageDuration ? `${averageDuration.toFixed(2)}s` : '—' }}</strong>
        <small>浏览器本地 CPU</small>
      </div>
    </section>

    <div v-if="isRunningAll || (doneCount > 0 && doneCount < totalCombinations)" class="running-panel benchmark-progress">
      <span>{{ currentTask || '等待任务' }}</span>
      <b>{{ runProgress }}%</b>
      <div><i :style="{ width: `${runProgress}%` }"></i></div>
    </div>

    <section class="content-section">
      <div class="section-title-row">
        <div><span class="overline">PERFORMANCE MATRIX</span><h2>算法 × 数据集 性能矩阵</h2></div>
        <span class="matrix-hint">单元格内依次为 Accuracy · Macro F1 · 耗时</span>
      </div>
      <div class="data-sheet benchmark-matrix">
        <div class="benchmark-row benchmark-head">
          <span>模型</span>
          <span v-for="dataset in datasets" :key="dataset.id">{{ dataset.name }}</span>
        </div>
        <div v-for="spec in benchmarkModels" :key="spec.id" class="benchmark-row">
          <span class="benchmark-model">
            <b>{{ spec.label }}</b>
            <small>{{ spec.detail }}</small>
          </span>
          <span v-for="dataset in datasets" :key="dataset.id" class="benchmark-cell">
            <template v-if="matrix[spec.id][dataset.id].status === 'done'">
              <b>{{ matrix[spec.id][dataset.id].accuracy?.toFixed(3) }}</b>
              <small>
                F1 {{ matrix[spec.id][dataset.id].f1?.toFixed(3) }} ·
                {{ matrix[spec.id][dataset.id].duration?.toFixed(2) }}s
              </small>
            </template>
            <template v-else-if="matrix[spec.id][dataset.id].status === 'running'">
              <small class="cell-running">运行中…</small>
            </template>
            <template v-else-if="matrix[spec.id][dataset.id].status === 'failed'">
              <small class="cell-failed">失败</small>
            </template>
            <template v-else><small class="cell-idle">—</small></template>
          </span>
        </div>
      </div>
    </section>

    <section class="content-section pca-section">
      <div class="section-title-row">
        <div>
          <span class="overline">PCA DEEP DIVE</span>
          <h2>PCA 降维专项分析</h2>
        </div>
        <div class="pca-controls">
          <select v-model="pcaDatasetId" :disabled="pcaState === 'running'">
            <option v-for="dataset in datasets" :key="dataset.id" :value="dataset.id">{{ dataset.name }}</option>
          </select>
          <button class="button ghost" :disabled="pcaState === 'running'" @click="runPcaAnalysis">
            <AppIcon name="algorithm" :size="15" />{{ pcaState === 'running' ? '分析中' : '运行 PCA 分析' }}
          </button>
        </div>
      </div>

      <div v-if="pcaState === 'running'" class="running-panel pca-progress"><span>{{ pcaMessage }}</span><b>…</b></div>

      <template v-if="pcaState === 'done'">
        <div class="pca-grid">
          <div class="pca-card">
            <span class="pca-label">方差解释率（前 {{ pcaChartBars.length }} 个主成分）</span>
            <svg class="variance-chart" viewBox="0 0 100 46" preserveAspectRatio="none">
              <rect
                v-for="(ratio, index) in pcaChartBars"
                :key="`bar-${index}`"
                :x="(index / pcaChartBars.length) * 100"
                :y="44 - (ratio / pcaBarScale) * 42"
                :width="100 / pcaChartBars.length - 0.4"
                :height="(ratio / pcaBarScale) * 42"
                class="variance-bar"
              />
              <polyline
                :points="pcaCumulative.map((value, index) => `${((index + 0.5) / pcaChartBars.length) * 100},${44 - value * 44}`).join(' ')"
                class="cumulative-line"
              />
            </svg>
            <small>柱：单个成分方差占比 · 线：累计方差（全部成分累计 {{ pcaTotalKept.toFixed(3) }}）</small>
          </div>
          <div class="pca-card">
            <span class="pca-label">SVM-RBF 准确率 vs 主成分数</span>
            <svg class="variance-chart" viewBox="0 0 100 46" preserveAspectRatio="none">
              <polyline
                :points="pcaCurve.map((point, index) => `${(index / Math.max(pcaCurve.length - 1, 1)) * 100},${accuracyLine(point)}`).join(' ')"
                class="accuracy-line"
              />
              <circle
                v-for="(point, index) in pcaCurve"
                :key="`dot-${index}`"
                :cx="(index / Math.max(pcaCurve.length - 1, 1)) * 100"
                :cy="accuracyLine(point)"
                r="1.1"
                class="accuracy-dot"
              />
            </svg>
            <div class="curve-table">
              <span v-for="point in pcaCurve" :key="`k-${point.components}`">
                <b>{{ point.components }} 维</b>
                <small>{{ (point.accuracy * 100).toFixed(1) }}%</small>
              </span>
            </div>
            <small>降维后维度大幅减少，准确率先升后稳，说明 PCA 去除了噪声冗余维度。</small>
          </div>
        </div>
      </template>
      <div v-else-if="pcaState === 'idle'" class="empty-state">
        选择数据集并运行分析，查看方差解释率与「主成分数 → 准确率」曲线。
      </div>
    </section>
  </div>
</template>
