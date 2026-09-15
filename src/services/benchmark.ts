/**
 * Local BenchmarkClient: 在浏览器内完成数据加载、训练与评估。
 * 数据集 JSON 由 scripts/export_datasets.py 从 scikit-learn 导出。
 */

import { algorithms, datasets } from '../data/catalog'
import { loadDataset } from '../ml/data'
import { runPipeline } from '../ml/pipeline'
import type {
  BenchmarkClient,
  ExperimentConfig,
  ExperimentResult,
  StageListener,
} from '../types/benchmark'

const datasetLabel = (id: string) => datasets.find((item) => item.id === id)?.name ?? id
const algorithmLabel = (id: string) => algorithms.find((item) => item.id === id)?.name ?? id

class LocalBenchmarkClient implements BenchmarkClient {
  async runExperiment(config: ExperimentConfig, onStage?: StageListener): Promise<ExperimentResult> {
    const bundle = await loadDataset(config.dataset)
    const output = await runPipeline(bundle, config, onStage)
    return {
      id: 'EXP-NEW',
      model: algorithmLabel(config.model),
      dataset: datasetLabel(config.dataset),
      accuracy: output.accuracy,
      f1: output.macroF1,
      duration: output.duration,
      status: 'completed',
      createdAt: '刚刚',
      detail: output.detail,
    }
  }
}

// 同一接口后续可替换为 WebSocketBenchmarkClient，接入 Python 后端。
export const benchmarkClient: BenchmarkClient = new LocalBenchmarkClient()
