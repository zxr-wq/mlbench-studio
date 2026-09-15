/** Registry of benchmark model configurations for the unified comparison page. */

import type { ExperimentConfig } from '../types/benchmark'
import type { SvmKernel } from './svm'

export interface BenchmarkModelSpec {
  id: string
  label: string
  detail: string
  model: 'svm' | 'kmeans'
  kernel?: SvmKernel
  c?: number
  pcaVariance?: number
}

export const benchmarkModels: BenchmarkModelSpec[] = [
  { id: 'svm_linear', label: 'SVM · Linear', detail: '线性核 · C=1', model: 'svm', kernel: 'linear', c: 1 },
  { id: 'svm_rbf', label: 'SVM · RBF', detail: '高斯核 · C=1', model: 'svm', kernel: 'rbf', c: 1 },
  { id: 'svm_poly', label: 'SVM · Poly', detail: '多项式核 d=3 · C=1', model: 'svm', kernel: 'poly', c: 1 },
  { id: 'kmeans', label: 'K-Means', detail: 'k=类别数 · k-means++ · 8 次重启', model: 'kmeans' },
  { id: 'pca_svm_rbf', label: 'PCA + SVM', detail: 'PCA 保留 95% 方差 + RBF 核', model: 'svm', kernel: 'rbf', c: 1, pcaVariance: 0.95 },
]

export function specToConfig(spec: BenchmarkModelSpec, datasetId: string, seed = 42): ExperimentConfig {
  return {
    dataset: datasetId,
    split: 'stratified_80_20',
    preprocessing: spec.pcaVariance ? `pca_${spec.pcaVariance}` : 'standard_scaler',
    model: spec.model,
    kernel: spec.kernel ?? 'rbf',
    c: spec.c ?? 1,
    metrics: ['accuracy', 'macro_f1'],
    seed,
  }
}
