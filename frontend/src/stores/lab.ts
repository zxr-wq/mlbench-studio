import { computed, reactive, ref } from 'vue'
import { awaitExperiment, benchmarkClient } from '../services/benchmark'
import { algorithms, datasets } from '../data/catalog'
import type { ExperimentConfig, ExperimentResult } from '../types/benchmark'

const STORAGE_KEY = 'mlbench-studio-runs'
const DATABASE_NAME = 'mlbench-studio'
const RUN_STORE = 'experiment-results'
const MAX_SAVED_RUNS = 40

const savedRuns = (() => {
  try { const rows = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]'); return Array.isArray(rows) ? rows.filter((row) => row && typeof row.id === 'string' && typeof row.model === 'string') as ExperimentResult[] : [] } catch { return [] as ExperimentResult[] }
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
    request.onblocked = () => reject(new Error('浏览器数据库被其他页面占用'))
  })
}

async function saveFullRun(run: ExperimentResult) {
  const database = await openDatabase()
  try { await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(RUN_STORE, 'readwrite')
    // Vue proxies cannot be structured-cloned by IndexedDB. Serialize a plain snapshot.
    transaction.objectStore(RUN_STORE).put(JSON.parse(JSON.stringify(run)))
    transaction.oncomplete = () => resolve()
    transaction.onerror = () => reject(transaction.error)
    transaction.onabort = () => reject(transaction.error ?? new Error('保存中断'))
  }) } finally { database.close() }
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
    runs.value = [...stored.filter((run) => !known.has(run.id)), ...merged].sort((a,b) => (b.completedAt ?? '').localeCompare(a.completedAt ?? ''))
    saveSummaries(runs.value)
  } catch {
    storageNotice.value = '完整图表记录暂时无法读取，指标摘要仍可正常使用。'
  }
}

const isHydrating = ref(true)
const storageReady = hydrateFullRuns().finally(() => { isHydrating.value = false })

const stats = computed(() => ({
  algorithms: algorithms.length,
  scratchAlgorithms: algorithms.filter((item) => item.implementation === '自主实现').length,
  datasets: datasets.length,
  completedRuns: runs.value.filter((run) => run.status === 'completed').length,
}))

const defaultConfig = reactive<ExperimentConfig>({
  dataset: 'iris',
  split: 'stratified_80_20',
  preprocessing: 'standard_scaler',
  model: 'knn',
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
  await storageReady
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
    result.inputConfig = JSON.parse(JSON.stringify(config))
    await rememberRun(result)
    progress.value = 100
    lastMessage.value = '实验已完成'
    return result
  } finally {
    isRunning.value = false
  }
}

async function rememberRun(result: ExperimentResult) {
  if (!result.id.startsWith('EXP-')) result.id = `EXP-${result.backendId ?? result.id}`
  runs.value = [result, ...runs.value.filter((item) => item.id !== result.id)]
  saveSummaries(runs.value)
  try { await saveFullRun(result) } catch {
    storageNotice.value = '实验已完成，但完整图表未能写入浏览器数据库；请下载完整 JSON。'
  }
}

async function importResults(content: string) {
  await storageReady
  const payload = JSON.parse(content)
  const rows = payload.experiments ?? (payload.experiment ? [payload.experiment] : payload)
  if (!Array.isArray(rows) || !rows.length || rows.length > 500) throw new Error('请导入本平台导出的完整 JSON（1–500 条记录）。')
  for (const row of rows) {
    if (!row || !algorithms.some((item) => item.id === row.model) || !datasets.some((item) => item.id === row.dataset)
      || typeof row.id !== 'string' || row.status !== 'completed' || typeof row.metrics !== 'object' || row.metrics === null
      || !Number.isFinite(row.duration) || !Number.isFinite(row.accuracy) || !Number.isFinite(row.f1)
      || !['classification','regression','clustering','dimensionality_reduction'].includes(row.taskType)
      || !Object.values(row.metrics).every((value) => (Array.isArray(value) ? value : [value]).every((item) => typeof item === 'number' && Number.isFinite(item)))) {
      throw new Error('文件中存在无效记录，未导入。请使用本平台导出的结果 JSON。')
    }
  }
  for (const row of rows.slice().reverse()) await rememberRun(row)
  return rows.length
}

async function deleteRun(id: string) {
  const database = await openDatabase()
  await new Promise<void>((resolve,reject) => {
    const transaction=database.transaction(RUN_STORE,'readwrite'); transaction.objectStore(RUN_STORE).delete(id)
    transaction.oncomplete=()=>resolve(); transaction.onerror=()=>reject(transaction.error)
  })
  database.close()
  runs.value=runs.value.filter((run)=>run.id!==id); saveSummaries(runs.value)
}

async function resumePending() {
  await storageReady
  let pending: { id:string, config:ExperimentConfig } | undefined
  try { pending = JSON.parse(localStorage.getItem('mlbench-pending') ?? 'null') } catch { return }
  if (!pending?.id || isRunning.value) return
  isRunning.value=true; lastMessage.value='正在恢复上次实验连接'
  try {
    const result=await awaitExperiment(pending.id,(update)=>{ progress.value=update.progress; lastMessage.value='正在恢复上次实验 · '+update.stage })
    result.inputConfig=pending.config; await rememberRun(result)
    try { localStorage.removeItem('mlbench-pending') } catch { /* Result is already saved. */ }
  } catch (error) {
    storageNotice.value=`上次实验暂时无法恢复：${error instanceof Error ? error.message : '连接失败'}。已保存的运行记录不受影响。`
    if (error instanceof Error && /unknown experiment|404|failed/i.test(error.message)) { try { localStorage.removeItem('mlbench-pending') } catch { /* Best effort. */ } }
  } finally { isRunning.value=false }
}
void resumePending()

export function useLabStore() {
  return { runs, stats, isRunning, isHydrating, progress, lastMessage, storageNotice, defaultConfig, runExperiment, importResults, deleteRun }
}
