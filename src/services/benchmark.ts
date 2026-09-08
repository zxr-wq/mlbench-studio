import type {
  BenchmarkClient,
  ExperimentConfig,
  ExperimentResult,
  OverviewStats,
} from '../types/benchmark'

const pause = (milliseconds: number) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds))

const recentRuns: ExperimentResult[] = [
  {
    id: 'EXP-024',
    model: 'SVM',
    dataset: 'Breast Cancer',
    accuracy: 0.972,
    f1: 0.968,
    duration: 1.82,
    status: 'completed',
    createdAt: '今天 15:42',
  },
  {
    id: 'EXP-023',
    model: 'Random Forest',
    dataset: 'Wine',
    accuracy: 0.961,
    f1: 0.958,
    duration: 2.34,
    status: 'completed',
    createdAt: '今天 14:18',
  },
  {
    id: 'EXP-022',
    model: 'Decision Tree',
    dataset: 'Iris',
    accuracy: 0.947,
    f1: 0.944,
    duration: 0.46,
    status: 'completed',
    createdAt: '昨天 20:06',
  },
]

class MockBenchmarkClient implements BenchmarkClient {
  async getOverview(): Promise<OverviewStats> {
    await pause(140)
    return {
      algorithms: 10,
      scratchAlgorithms: 8,
      datasets: 4,
      completedRuns: recentRuns.length + 21,
    }
  }

  async getRecentRuns(): Promise<ExperimentResult[]> {
    await pause(180)
    return [...recentRuns]
  }

  async runExperiment(config: ExperimentConfig): Promise<ExperimentResult> {
    await pause(1200)

    const modelLabel: Record<string, string> = {
      svm: 'SVM',
      decision_tree: 'Decision Tree',
      knn: 'KNN',
    }
    const datasetLabel: Record<string, string> = {
      iris: 'Iris',
      wine: 'Wine',
      breast_cancer: 'Breast Cancer',
      digits: 'Digits',
    }

    return {
      id: `EXP-${String(recentRuns.length + 25).padStart(3, '0')}`,
      model: modelLabel[config.model] ?? config.model,
      dataset: datasetLabel[config.dataset] ?? config.dataset,
      accuracy: Number((0.935 + Math.random() * 0.045).toFixed(3)),
      f1: Number((0.928 + Math.random() * 0.045).toFixed(3)),
      duration: Number((0.7 + Math.random() * 1.8).toFixed(2)),
      status: 'completed',
      createdAt: '刚刚',
    }
  }
}

// 后续接入真实后端时，只需把这里替换为 WebSocketBenchmarkClient。
export const benchmarkClient: BenchmarkClient = new MockBenchmarkClient()

