import { computed, reactive, ref } from 'vue'
import { benchmarkClient } from '../services/benchmark'
import { algorithms, datasets } from '../data/catalog'
import type { ExperimentConfig, ExperimentResult } from '../types/benchmark'

const STORAGE_KEY = 'mlbench-studio-runs'
const DATABASE_NAME = 'mlbench-studio'
const RUN_STORE = 'experiment-results'
const MAX_SAVED_RUNS = 40

const savedRuns = (() => {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]') as ExperimentResult[] } catch { return [] as ExperimentResult[] }
})()
const runs = ref<ExperimentResult[]>(savedRuns)
const isRunning = ref(false)
const progress = ref(0)
const lastMessage = ref('')
const storageNotice = ref('')

function compactRun(run: ExperimentResult): ExperimentResult {
  return {
    ...run,
    visualization: undefined,
    evaluation: run.evaluation ? { ...run.evaluation, per_run: undefined } : undefined,
  }
}

function saveSummaries(items: ExperimentResult[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, MAX_SAVED_RUNS).map(compactRun)))
  } catch {
    storageNotice.value = '实验已完成，但浏览器摘要存储空间不足；请下载 JSON 保存本次结果。'
  }
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, 1)
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(RUN_STORE)) {
        request.result.createObjectStore(RUN_STORE, { keyPath: 'id' })
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

async function saveFullRun(run: ExperimentResult) {
  const database = await openDatabase()
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(RUN_STORE, 'readwrite')
    transaction.objectStore(RUN_STORE).put(run)
    transaction.oncomplete = () => resolve()
    transaction.onerror = () => reject(transaction.error)
  })
  database.close()
}

async function hydrateFullRuns() {
  if (typeof indexedDB === 'undefined') return
  try {
    // Migrate legacy records that used to place multi-megabyte chart payloads
    // directly in localStorage before rewriting that storage as summaries.
    for (const run of runs.value.filter((item) => item.visualization)) {
      await saveFullRun(run)
    }
    const database = await openDatabase()
    const stored = await new Promise<ExperimentResult[]>((resolve, reject) => {
      const request = database.transaction(RUN_STORE, 'readonly').objectStore(RUN_STORE).getAll()
      request.onsuccess = () => resolve(request.result as ExperimentResult[])
      request.onerror = () => reject(request.error)
    })
    database.close()
    const fullById = new Map(stored.map((run) => [run.id, run]))
    const merged = runs.value.map((run) => fullById.get(run.id) ?? run)
    const known = new Set(merged.map((run) => run.id))
    runs.value = [...stored.filter((run) => !known.has(run.id)), ...merged]
    saveSummaries(runs.value)
  } catch {
    storageNotice.value = '完整图表记录暂时无法读取，指标摘要仍可正常使用。'
  }
}

void hydrateFullRuns()

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
  maxIterations: 800,
  varSmoothing: 1e-9,
  pcaComponents: 2,
  fitIntercept: true,
  implementation: 'scratch',
  metrics: ['accuracy', 'macro_f1'],
  seed: 42,
  stabilityRuns: 1,
})

async function runExperiment(config: ExperimentConfig) {
  if (isRunning.value) throw new Error('An experiment is already running')
  isRunning.value = true
  storageNotice.value = ''
  progress.value = 0
  lastMessage.value = '实验正在排队'

  const stageLabels: Record<string, string> = {
    queued: '实验正在排队',
    'loading dataset': '正在加载与划分数据集',
    'training model': '正在执行真实模型训练与评价',
    completed: '实验已完成',
    failed: '实验运行失败',
  }

  try {
    const result = await benchmarkClient.runExperiment(config, (update) => {
      progress.value = update.progress
      lastMessage.value = stageLabels[update.stage] ?? update.stage
    })
    const serial = runs.value.reduce((largest, run) => {
      const value = Number(run.id.replace('EXP-', ''))
      return Number.isFinite(value) ? Math.max(largest, value) : largest
    }, 0) + 1
    result.id = `EXP-${String(serial).padStart(3, '0')}`
    result.createdAt = '刚刚'
    runs.value = [result, ...runs.value]
    saveSummaries(runs.value)
    void saveFullRun(result).catch(() => {
      storageNotice.value = '实验已完成并保留指标，但完整图表未能写入浏览器数据库；建议下载 JSON。'
    })
    progress.value = 100
    lastMessage.value = '实验已完成'
    return result
  } finally {
    window.setTimeout(() => {
      isRunning.value = false
      progress.value = 0
    }, 300)
  }
}

export function useLabStore() {
  return { runs, stats, isRunning, progress, lastMessage, storageNotice, defaultConfig, runExperiment }
}
