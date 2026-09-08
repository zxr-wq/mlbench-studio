export type RunStatus = 'completed' | 'running' | 'failed'

export interface ExperimentConfig {
  dataset: string
  split: string
  preprocessing: string
  model: string
  kernel: string
  c: number
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

