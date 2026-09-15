/** Unified experiment pipeline: split → preprocess → train → predict → evaluate. */

import type { ExperimentConfig, PerClassMetric } from '../types/benchmark'
import { KMeansClassifier } from './kmeans'
import { evaluateClassification, silhouetteScore, type ClassificationReport } from './metrics'
import { fitPca, pcaTransform, type PcaModel } from './pca'
import {
  MinMaxScaler,
  mulberry32,
  StandardScaler,
  stratifiedFolds,
  stratifiedSplit,
  type FeatureScaler,
} from './preprocess'
import { SvmClassifier, type SvmKernel } from './svm'
import type { DatasetBundle, Row, StageReporter } from './types'
import { range } from './math'

const yieldToUi = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0)
  })

export interface RunDetail {
  confusionMatrix: number[][]
  classLabels: string[]
  perClass: PerClassMetric[]
  silhouette?: number
  pcaComponents?: number
  explainedVariance?: number[]
  config: ExperimentConfig
}

export interface PipelineOutput {
  accuracy: number
  macroF1: number
  macroPrecision: number
  macroRecall: number
  duration: number
  detail: RunDetail
}

interface PreparedFeatures {
  train: Row[]
  test: Row[]
  pca?: PcaModel
}

interface RunOutcome {
  report: ClassificationReport
  silhouette?: number
  pca?: PcaModel
}

function pickRows(rows: Row[], indices: number[]): Row[] {
  return indices.map((index) => rows[index])
}

function pickValues(values: number[], indices: number[]): number[] {
  return indices.map((index) => values[index])
}

/** PCA-aware preprocessing: `pca_<ratio>` keeps that variance share, `pca_k_<n>` keeps n dims. */
function prepareFeatures(XTrain: Row[], XTest: Row[], preprocessing: string): PreparedFeatures {
  if (preprocessing === 'none') return { train: XTrain, test: XTest }
  const scaler: FeatureScaler = preprocessing === 'minmax_scaler' ? new MinMaxScaler() : new StandardScaler()
  const train = scaler.fitTransform(XTrain)
  const test = scaler.transform(XTest)

  if (preprocessing.startsWith('pca')) {
    const fixedMatch = /^pca_k_(\d+)$/.exec(preprocessing)
    if (fixedMatch) {
      const pca = fitPca(train, { maxComponents: Number(fixedMatch[1]) })
      return { train: pcaTransform(pca, train), test: pcaTransform(pca, test), pca }
    }
    const ratio = Number(preprocessing.slice(4))
    const pca = fitPca(train, { varianceRatio: Number.isFinite(ratio) && ratio > 0 ? ratio : 0.95 })
    return { train: pcaTransform(pca, train), test: pcaTransform(pca, test), pca }
  }
  return { train, test }
}

function createModel(config: ExperimentConfig, classCount: number) {
  if (config.model === 'kmeans') return new KMeansClassifier(classCount, config.seed)
  const kernel: SvmKernel =
    config.kernel === 'linear' || config.kernel === 'poly' ? config.kernel : 'rbf'
  return new SvmClassifier({ kernel, c: config.c > 0 ? config.c : 1 })
}

async function runHoldout(
  bundle: DatasetBundle,
  config: ExperimentConfig,
  onStage?: StageReporter,
): Promise<RunOutcome> {
  const { trainIdx, testIdx } = stratifiedSplit(bundle.target, 0.2, config.seed)
  const XTrain = pickRows(bundle.data, trainIdx)
  const yTrain = pickValues(bundle.target, trainIdx)
  const XTest = pickRows(bundle.data, testIdx)
  const yTest = pickValues(bundle.target, testIdx)

  onStage?.('preprocess', '正在拟合预处理与降维')
  await yieldToUi()
  const prepared = prepareFeatures(XTrain, XTest, config.preprocessing)

  onStage?.('train', '正在训练模型')
  await yieldToUi()
  const model = createModel(config, bundle.targetNames.length)
  await model.fit(prepared.train, yTrain)

  onStage?.('predict', '正在生成测试集预测')
  await yieldToUi()
  const yPred = model.predict(prepared.test)
  const report = evaluateClassification(yTest, yPred, bundle.targetNames)

  let silhouette: number | undefined
  if (model instanceof KMeansClassifier) {
    silhouette = silhouetteScore(prepared.train, model.trainLabels, mulberry32(config.seed))
  }
  return { report, silhouette, pca: prepared.pca }
}

async function runCrossValidation(
  bundle: DatasetBundle,
  config: ExperimentConfig,
  onStage?: StageReporter,
): Promise<RunOutcome> {
  const foldCount = 5
  const folds = stratifiedFolds(bundle.target, foldCount, config.seed)
  const all = range(bundle.target.length)
  const yTrue: number[] = []
  const yPred: number[] = []
  for (let fold = 0; fold < foldCount; fold += 1) {
    onStage?.('train', `五折交叉验证 · 第 ${fold + 1}/${foldCount} 折`)
    await yieldToUi()
    const testIdx = folds[fold]
    const testSet = new Set(testIdx)
    const trainIdx = all.filter((index) => !testSet.has(index))
    const XTrain = pickRows(bundle.data, trainIdx)
    const XTest = pickRows(bundle.data, testIdx)
    const prepared = prepareFeatures(XTrain, XTest, config.preprocessing)
    const model = createModel(config, bundle.targetNames.length)
    await model.fit(prepared.train, pickValues(bundle.target, trainIdx))
    yPred.push(...model.predict(prepared.test))
    yTrue.push(...pickValues(bundle.target, testIdx))
  }
  return { report: evaluateClassification(yTrue, yPred, bundle.targetNames) }
}

export async function runPipeline(
  bundle: DatasetBundle,
  config: ExperimentConfig,
  onStage?: StageReporter,
): Promise<PipelineOutput> {
  const started = performance.now()
  onStage?.('split', '正在划分训练与测试集')
  await yieldToUi()

  const outcome =
    config.split === 'kfold_5'
      ? await runCrossValidation(bundle, config, onStage)
      : await runHoldout(bundle, config, onStage)

  onStage?.('metrics', '正在计算评价指标')
  await yieldToUi()

  return {
    accuracy: outcome.report.accuracy,
    macroF1: outcome.report.macroF1,
    macroPrecision: outcome.report.macroPrecision,
    macroRecall: outcome.report.macroRecall,
    duration: Number(((performance.now() - started) / 1000).toFixed(2)),
    detail: {
      confusionMatrix: outcome.report.confusionMatrix,
      classLabels: outcome.report.classLabels,
      perClass: outcome.report.perClass,
      silhouette: outcome.silhouette,
      pcaComponents: outcome.pca?.nComponents,
      explainedVariance: outcome.pca?.explainedVarianceRatio,
      config: { ...config },
    },
  }
}
