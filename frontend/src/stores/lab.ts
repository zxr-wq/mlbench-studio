import { computed, reactive, ref } from 'vue'
import { benchmarkClient } from '../services/benchmark'
import { algorithms, datasets, initialRuns } from '../data/catalog'
import type { ExperimentConfig, ExperimentResult } from '../types/benchmark'

const runs = ref<ExperimentResult[]>([...initialRuns])
const isRunning = ref(false)
const progress = ref(0)
const lastMessage = ref('')

const stats = computed(() => ({
  algorithms: algorithms.length,
  scratchAlgorithms: algorithms.filter((item) => item.implementation === '自主实现').length,
  datasets: datasets.length,
  completedRuns: runs.value.filter((run) => run.status === 'completed').length,
}))

const defaultConfig = reactive<ExperimentConfig>({
  dataset: 'breast_cancer',
  split: 'stratified_80_20',
  preprocessing: 'standard_scaler',
  model: 'svm',
  kernel: 'rbf',
  c: 1,
  k: 5,
  distance: 'euclidean',
  maxDepth: 4,
  nEstimators: 50,
  learningRate: 0.1,
  implementation: 'scratch',
  metrics: ['accuracy', 'macro_f1'],
  seed: 42,
})

async function runExperiment(config: ExperimentConfig) {
  if (isRunning.value) throw new Error('An experiment is already running')
  isRunning.value = true
  progress.value = 8
  lastMessage.value = '正在加载数据集'
  const timer = window.setInterval(() => {
    progress.value = Math.min(progress.value + 11, 89)
    if (progress.value > 60) lastMessage.value = '正在计算评价指标'
    else if (progress.value > 30) lastMessage.value = '正在训练模型'
  }, 150)

  try {
    const result = await benchmarkClient.runExperiment(config)
    const serial = Math.max(...runs.value.map((run) => Number(run.id.replace('EXP-', '')))) + 1
    result.id = `EXP-${String(serial).padStart(3, '0')}`
    result.createdAt = '刚刚'
    runs.value = [result, ...runs.value]
    progress.value = 100
    lastMessage.value = '实验已完成'
    return result
  } finally {
    window.clearInterval(timer)
    window.setTimeout(() => {
      isRunning.value = false
      progress.value = 0
    }, 300)
  }
}

export function useLabStore() {
  return { runs, stats, isRunning, progress, lastMessage, defaultConfig, runExperiment }
}
