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

export interface PerClassMetric {
  label: string
  precision: number
  recall: number
  f1: number
  support: number
}

export interface RunDetail {
  confusionMatrix: number[][]
  classLabels: string[]
  perClass: PerClassMetric[]
  silhouette?: number
  pcaComponents?: number
  explainedVariance?: number[]
  config: ExperimentConfig
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
  detail?: RunDetail
}

export type StageListener = (stage: string, message: string) => void

export interface BenchmarkClient {
  runExperiment(config: ExperimentConfig, onStage?: StageListener): Promise<ExperimentResult>
}
