import type { BenchmarkClient, ExperimentConfig, ExperimentProgress, ExperimentResult, OverviewStats } from '../types/benchmark'

const apiBase = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')
function clearPending() { try { localStorage.removeItem('mlbench-pending') } catch { /* Best effort. */ } }
const request = async (path: string, options?: RequestInit) => {
  const response = await fetch(`${apiBase}${path}`, { ...options, signal: AbortSignal.timeout(15000) })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `Request failed (${response.status})`)
  }
  return response.json()
}

function makeConfig(config: ExperimentConfig) {
  const params: Record<string, unknown> = {
    linear_regression: { fit_intercept: config.fitIntercept },
    logistic_regression: { learning_rate: config.learningRate, max_iter: config.maxIterations, l2: 0.01 },
    naive_bayes: { var_smoothing: config.varSmoothing },
    svm: { C: config.c, kernel: config.kernel },
    knn: { k: config.k, distance: config.distance },
    decision_tree: { max_depth: config.maxDepth },
    random_forest: { n_estimators: config.nEstimators, max_depth: config.maxDepth, random_state: config.seed },
    gradient_boosting: { n_estimators: config.nEstimators, learning_rate: config.learningRate, max_depth: config.maxDepth },
    kmeans: { k: config.k, random_state: config.seed, n_init: 10, max_iter: 300, tol: 1e-4 },
    pca: { n_components: config.pcaComponents },
  }
  const metrics = config.metrics.map((metric) => metric === 'macro_f1' ? 'f1' : metric)
  return {
    dataset: config.dataset,
    split: { test_size: 0.2, random_state: config.seed },
    preprocessing: config.preprocessing === 'none' ? [] : [config.preprocessing],
    model: { name: config.model, implementation: config.implementation, params: params[config.model] ?? {} },
    metrics,
    evaluation: {
      strategy: config.split === 'kfold_5' ? 'kfold' : 'holdout',
      n_splits: 5,
      repetitions: config.stabilityRuns,
    },
  }
}

function scalarMetric(value: unknown) {
  if (Array.isArray(value)) return Number(value[value.length - 1] ?? 0)
  return Number(value ?? 0)
}

function mapRecord(record: any): ExperimentResult {
  const result = record.result
  if (!result) throw new Error(record.error ?? 'Experiment ended without a result')
  const metrics = result.metrics ?? {}
  const task = result.task_type as string
  const preferences = task === 'regression' ? ['r2', 'rmse', 'mae', 'mse']
    : task === 'clustering' ? ['silhouette_score', 'inertia']
      : task === 'dimensionality_reduction' ? ['cumulative_explained_variance', 'explained_variance_ratio']
        : ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
  const [primaryMetric, secondaryMetric] = preferences.filter((key) => metrics[key] != null)
  return {
    id: record.id, backendId: record.id, model: result.model, dataset: result.dataset,
    accuracy: Number(metrics.accuracy ?? 0), f1: Number(metrics.f1 ?? 0),
    duration: Number(result.training_time ?? 0) + Number(result.inference_time ?? 0),
    wallDuration: record.created_at && record.finished_at ? Math.max(0,(Date.parse(record.finished_at)-Date.parse(record.created_at))/1000) : undefined,
    trainingTime: Number(result.training_time ?? 0), inferenceTime: Number(result.inference_time ?? 0),
    status: record.status, createdAt: new Date(record.finished_at ?? Date.now()).toLocaleString('zh-CN'), completedAt: record.finished_at, taskType: result.task_type,
    implementation: result.implementation, metrics, metricStd: result.metric_std,
    evaluation: result.evaluation, config: record.config, params: result.params, visualization: result.visualization,
    primaryMetric, primaryValue: scalarMetric(metrics[primaryMetric]),
    secondaryMetric, secondaryValue: metrics[secondaryMetric] == null ? undefined : scalarMetric(metrics[secondaryMetric]),
  }
}

export async function awaitExperiment(id: string, onProgress?: (update: ExperimentProgress) => void): Promise<ExperimentResult> {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const socketUrl = `${apiBase ? apiBase.replace(/^http/, 'ws') : `${protocol}//${window.location.host}`}/api/ws/experiments/${id}`
  return new Promise((resolve, reject) => {
    let settled = false
    let polling = false
    let pollTimer: number | undefined
    const fallbackTimer = window.setTimeout(() => startPolling(), 3000)
    let socket:WebSocket | undefined
    const finish = (fn: (value: any) => void, value: any) => {
      if (!settled) { settled = true; window.clearTimeout(pollTimer); window.clearTimeout(fallbackTimer); socket?.close(); fn(value) }
    }
    const startPolling = () => {
      if (polling || settled) return
      polling = true
      const poll = async () => {
        if (settled) return
        try {
          const record = await request(`/api/experiments/${id}`)
          onProgress?.({ status: record.status, progress: Number(record.progress ?? 0), stage: String(record.stage ?? '') })
          if (record.status === 'completed') { clearPending(); return finish(resolve, mapRecord(record)) }
          if (record.status === 'failed') { clearPending(); return finish(reject, new Error(record.error ?? 'Experiment failed')) }
          pollTimer = window.setTimeout(poll, 1500)
        } catch (error) {
          finish(reject, error)
        }
      }
      void poll()
    }
    try { socket=new WebSocket(socketUrl) } catch {startPolling();return}
    socket.onmessage = (event) => {
      let record
      try { record = JSON.parse(event.data).experiment; if (!record?.status) return } catch { return }
      onProgress?.({ status: record.status, progress: Number(record.progress ?? 0), stage: String(record.stage ?? '') })
      if (record.status === 'completed') { clearPending(); finish(resolve, mapRecord(record)) }
      if (record.status === 'failed') { clearPending(); finish(reject, new Error(record.error ?? 'Experiment failed')) }
    }
    socket.onerror = startPolling
    socket.onclose = startPolling
  })
}

class ApiBenchmarkClient implements BenchmarkClient {
  async getOverview(): Promise<OverviewStats> {
    const [algorithms, datasets] = await Promise.all([request('/api/algorithms'), request('/api/datasets')])
    return { algorithms: algorithms.length, scratchAlgorithms: algorithms.length, datasets: datasets.length, completedRuns: 0 }
  }
  async getRecentRuns(): Promise<ExperimentResult[]> { return [] }
  async runExperiment(config: ExperimentConfig, onProgress?: (update: ExperimentProgress) => void): Promise<ExperimentResult> {
    const created = await request('/api/experiments', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(makeConfig(config)) })
    try { localStorage.setItem('mlbench-pending', JSON.stringify({ id:created.id, config })) } catch { /* In-memory execution still works. */ }
    onProgress?.({ status: created.status, progress: 0, stage: 'queued' })
    const result = await awaitExperiment(created.id, onProgress)
    try { localStorage.removeItem('mlbench-pending') } catch { /* Best effort. */ }
    return result
  }
}

export const benchmarkClient: BenchmarkClient = new ApiBenchmarkClient()

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const health = await request('/api/health')
    return health?.status === 'ok'
  } catch {
    return false
  }
}
