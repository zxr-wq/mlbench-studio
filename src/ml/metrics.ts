/** Evaluation metrics: classification report + silhouette score. */

import { range, shuffleInPlace, squaredDistance } from './math'
import type { Random } from './math'
import type { Row } from './types'

export interface PerClassMetric {
  label: string
  precision: number
  recall: number
  f1: number
  support: number
}

export interface ClassificationReport {
  accuracy: number
  macroPrecision: number
  macroRecall: number
  macroF1: number
  classLabels: string[]
  confusionMatrix: number[][]
  perClass: PerClassMetric[]
}

function clampIndex(value: number, classCount: number): number {
  const rounded = Math.round(value)
  if (Number.isNaN(rounded)) return 0
  return Math.min(Math.max(rounded, 0), classCount - 1)
}

export function evaluateClassification(
  yTrue: number[],
  yPred: number[],
  classLabels: string[],
): ClassificationReport {
  const classCount = classLabels.length
  const confusionMatrix: number[][] = range(classCount).map(() => new Array<number>(classCount).fill(0))
  let correct = 0
  for (let i = 0; i < yTrue.length; i += 1) {
    const truth = yTrue[i]
    const prediction = yPred[i] ?? 0
    if (truth === prediction) correct += 1
    confusionMatrix[clampIndex(truth, classCount)][clampIndex(prediction, classCount)] += 1
  }

  const perClass = classLabels.map((label, index) => {
    let tp = 0
    let fp = 0
    let fn = 0
    for (let r = 0; r < classCount; r += 1) {
      for (let c = 0; c < classCount; c += 1) {
        if (r === index && c === index) tp += confusionMatrix[r][c]
        else if (c === index) fp += confusionMatrix[r][c]
        else if (r === index) fn += confusionMatrix[r][c]
      }
    }
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0
    return { label, precision, recall, f1, support: tp + fn }
  })

  const total = yTrue.length || 1
  const average = (pick: (metric: PerClassMetric) => number) =>
    perClass.reduce((sum, metric) => sum + pick(metric), 0) / (classCount || 1)

  return {
    accuracy: correct / total,
    macroPrecision: average((metric) => metric.precision),
    macroRecall: average((metric) => metric.recall),
    macroF1: average((metric) => metric.f1),
    classLabels,
    confusionMatrix,
    perClass,
  }
}

/**
 * Mean silhouette coefficient over (a deterministic sample of) the samples.
 * Unsupervised quality measure used for clustering runs.
 */
export function silhouetteScore(X: Row[], labels: number[], rand: Random, sampleLimit = 1200): number {
  const total = X.length
  if (total < 2) return 0
  let indices = range(total)
  if (total > sampleLimit) {
    shuffleInPlace(indices, rand)
    indices = indices.slice(0, sampleLimit)
  }
  const n = indices.length
  const clusters = new Map<number, number[]>()
  indices.forEach((sampleIndex, position) => {
    const list = clusters.get(labels[sampleIndex]) ?? []
    list.push(position)
    clusters.set(labels[sampleIndex], list)
  })

  let sum = 0
  let counted = 0
  for (let i = 0; i < n; i += 1) {
    const ownCluster = clusters.get(labels[indices[i]]) ?? []
    if (ownCluster.length < 2) continue
    const distances: Map<number, number> = new Map()
    let ownSum = 0
    for (let j = 0; j < n; j += 1) {
      if (j === i) continue
      const distance = Math.sqrt(squaredDistance(X[indices[i]], X[indices[j]]))
      ownSum += distance
      const other = labels[indices[j]]
      if (other !== labels[indices[i]]) distances.set(other, (distances.get(other) ?? 0) + distance)
    }
    const a = ownSum / (ownCluster.length - 1)
    let b = Infinity
    for (const [, clusterSum] of distances) b = Math.min(b, clusterSum)
    sum += (b - a) / Math.max(a, b)
    counted += 1
  }
  return counted > 0 ? sum / counted : 0
}
