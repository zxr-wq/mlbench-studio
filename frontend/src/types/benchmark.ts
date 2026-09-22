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
  implementation: 'scratch' | 'sklearn'
  metrics: string[]
  seed: number
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
