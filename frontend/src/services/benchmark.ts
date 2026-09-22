import type { BenchmarkClient, ExperimentConfig, ExperimentResult, OverviewStats } from '../types/benchmark'

const apiBase = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')
const request = async (path: string, options?: RequestInit) => {
  const response = await fetch(`${apiBase}${path}`, options)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `Request failed (${response.status})`)
  }
  return response.json()
}

function makeConfig(config: ExperimentConfig) {
  const params: Record<string, unknown> = {
    linear_regression: { fit_intercept: true },
    logistic_regression: { learning_rate: config.learningRate, max_iter: config.nEstimators * 10, l2: 0.01 },
    svm: { C: config.c, kernel: config.kernel },
    knn: { k: config.k, distance: config.distance },
    decision_tree: { max_depth: config.maxDepth },
    random_forest: { n_estimators: config.nEstimators, max_depth: config.maxDepth, random_state: config.seed },
    gradient_boosting: { n_estimators: config.nEstimators, learning_rate: config.learningRate, max_depth: config.maxDepth },
    kmeans: { k: config.k, random_state: config.seed },
    pca: { n_components: 2 },
  }
  const metrics = config.model === 'kmeans'
    ? ['silhouette_score', 'inertia']
    : config.model === 'pca'
      ? ['explained_variance_ratio', 'cumulative_explained_variance']
      : config.model === 'linear_regression'
        ? ['mse', 'rmse', 'mae', 'r2']
      : config.metrics.map((metric) => metric === 'macro_f1' ? 'f1' : metric)
  return {
    dataset: config.dataset,
    split: { test_size: 0.2, random_state: config.seed },
    preprocessing: config.preprocessing === 'none' ? [] : [config.preprocessing],
    model: { name: config.model, implementation: config.implementation, params: params[config.model] ?? {} },
    metrics,
  }
}

function mapRecord(record: any): ExperimentResult {
  const result = record.result
  if (!result) throw new Error(record.error ?? 'Experiment ended without a result')
  const metrics = result.metrics ?? {}
  const task = result.task_type as string
  const [primaryMetric, secondaryMetric] = task === 'regression' ? ['r2', 'rmse']
    : task === 'clustering' ? ['silhouette_score', 'inertia']
      : task === 'dimensionality_reduction' ? ['cumulative_explained_variance', 'explained_variance_ratio']
        : ['accuracy', 'f1']
  return {
    id: record.id, model: result.model, dataset: result.dataset,
    accuracy: Number(metrics.accuracy ?? 0), f1: Number(metrics.f1 ?? 0),
    duration: Number(result.training_time ?? 0) + Number(result.inference_time ?? 0),
    status: record.status, createdAt: '刚刚', taskType: result.task_type,
    implementation: result.implementation, metrics, params: result.params, visualization: result.visualization,
    primaryMetric, primaryValue: Number(metrics[primaryMetric] ?? 0),
    secondaryMetric, secondaryValue: typeof metrics[secondaryMetric] === 'number' ? Number(metrics[secondaryMetric]) : undefined,
  }
}

async function awaitExperiment(id: string): Promise<ExperimentResult> {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const socketUrl = `${apiBase ? apiBase.replace(/^http/, 'ws') : `${protocol}//${window.location.host}`}/api/ws/experiments/${id}`
  return new Promise((resolve, reject) => {
    let settled = false
    const socket = new WebSocket(socketUrl)
    const finish = (fn: (value: any) => void, value: any) => {
      if (!settled) { settled = true; socket.close(); fn(value) }
    }
    socket.onmessage = (event) => {
      const record = JSON.parse(event.data).experiment
      if (record.status === 'completed') finish(resolve, mapRecord(record))
      if (record.status === 'failed') finish(reject, new Error(record.error ?? 'Experiment failed'))
    }
    socket.onerror = async () => {
      try {
        const poll = async () => {
          const record = await request(`/api/experiments/${id}`)
          if (record.status === 'completed') return finish(resolve, mapRecord(record))
          if (record.status === 'failed') return finish(reject, new Error(record.error ?? 'Experiment failed'))
          window.setTimeout(poll, 400)
        }
        poll()
      } catch (error) { finish(reject, error) }
    }
  })
}

class ApiBenchmarkClient implements BenchmarkClient {
  async getOverview(): Promise<OverviewStats> {
    const [algorithms, datasets] = await Promise.all([request('/api/algorithms'), request('/api/datasets')])
    return { algorithms: algorithms.length, scratchAlgorithms: algorithms.length, datasets: datasets.length, completedRuns: 0 }
  }
  async getRecentRuns(): Promise<ExperimentResult[]> { return [] }
  async runExperiment(config: ExperimentConfig): Promise<ExperimentResult> {
    const created = await request('/api/experiments', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(makeConfig(config)) })
    return awaitExperiment(created.id)
  }
}

export const benchmarkClient: BenchmarkClient = new ApiBenchmarkClient()
