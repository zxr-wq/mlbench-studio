export type RunStatus = 'completed' | 'running' | 'failed'

export interface ExperimentConfig {
  dataset: string
  split: string
  preprocessing: string
  model: string
  kernel: string
  c: number
  k: number
  distance: 'euclidean' | 'manhattan'
  maxDepth: number
  nEstimators: number
  learningRate: number
  maxIterations: number
  varSmoothing: number
  pcaComponents: number
  fitIntercept: boolean
  implementation: 'scratch' | 'sklearn'
  metrics: string[]
  seed: number
  stabilityRuns: 1 | 3 | 5
}

export interface EvaluationSummary {
  strategy: 'holdout' | 'kfold'
  repetitions: number
  n_splits: number
  total_runs: number
  seeds: number[]
  per_run?: Array<{ seed: number, fold: number, metrics: Record<string, number | number[]> }>
}

export interface ExperimentResult {
  id: string
  model: string
  dataset: string
  accuracy: number
  f1: number
  duration: number
  status: RunStatus
  createdAt: string
  taskType?: string
  implementation?: string
  metrics?: Record<string, number | number[]>
  metricStd?: Record<string, number | number[]>
  evaluation?: EvaluationSummary
  params?: Record<string, unknown>
  visualization?: Record<string, unknown>
  primaryMetric?: string
  primaryValue?: number
  secondaryMetric?: string
  secondaryValue?: number
}

export interface OverviewStats {
  algorithms: number
  scratchAlgorithms: number
  datasets: number
  completedRuns: number
}

export interface BenchmarkClient {
  getOverview(): Promise<OverviewStats>
  getRecentRuns(): Promise<ExperimentResult[]>
  runExperiment(config: ExperimentConfig): Promise<ExperimentResult>
}
