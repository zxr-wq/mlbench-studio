/** Reproducible randomness, stratified splitting and feature scalers. */

import { range, shuffleInPlace, uniqueSorted, type Random } from './math'
import type { Row } from './types'

/** Deterministic PRNG so identical seeds reproduce identical experiments. */
export function mulberry32(seed: number): Random {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export interface IndexSplit {
  trainIdx: number[]
  testIdx: number[]
}

function classIndices(y: number[], cls: number): number[] {
  const indices: number[] = []
  for (let i = 0; i < y.length; i += 1) if (y[i] === cls) indices.push(i)
  return indices
}

/** Per-class shuffle, then take a fixed ratio of every class as the test side. */
export function stratifiedSplit(y: number[], testRatio: number, seed: number): IndexSplit {
  const rand = mulberry32(seed)
  const trainIdx: number[] = []
  const testIdx: number[] = []
  for (const cls of uniqueSorted(y)) {
    const indices = classIndices(y, cls)
    shuffleInPlace(indices, rand)
    const testCount = Math.max(1, Math.round(indices.length * testRatio))
    testIdx.push(...indices.slice(0, testCount))
    trainIdx.push(...indices.slice(testCount))
  }
  return { trainIdx, testIdx }
}

/** Deal per-class shuffled indices round-robin into stratified folds. */
export function stratifiedFolds(y: number[], foldCount: number, seed: number): number[][] {
  const rand = mulberry32(seed)
  const folds: number[][] = range(foldCount).map(() => [])
  for (const cls of uniqueSorted(y)) {
    const indices = classIndices(y, cls)
    shuffleInPlace(indices, rand)
    indices.forEach((sample, position) => folds[position % foldCount].push(sample))
  }
  return folds
}

export interface FeatureScaler {
  transform(X: Row[]): Row[]
}

export class StandardScaler implements FeatureScaler {
  private mean: Row = []

  private std: Row = []

  fit(X: Row[]): this {
    const sampleCount = X.length
    const dim = X[0]?.length ?? 0
    this.mean = new Array<number>(dim).fill(0)
    this.std = new Array<number>(dim).fill(1)
    if (!sampleCount) return this
    for (const row of X) for (let j = 0; j < dim; j += 1) this.mean[j] += row[j]
    for (let j = 0; j < dim; j += 1) this.mean[j] /= sampleCount
    for (let j = 0; j < dim; j += 1) this.std[j] = 0
    for (const row of X) {
      for (let j = 0; j < dim; j += 1) {
        const delta = row[j] - this.mean[j]
        this.std[j] += delta * delta
      }
    }
    for (let j = 0; j < dim; j += 1) {
      const s = Math.sqrt(this.std[j] / sampleCount)
      this.std[j] = s > 1e-12 ? s : 1
    }
    return this
  }

  transform(X: Row[]): Row[] {
    return X.map((row) => row.map((value, j) => (value - this.mean[j]) / this.std[j]))
  }

  fitTransform(X: Row[]): Row[] {
    return this.fit(X).transform(X)
  }
}

export class MinMaxScaler implements FeatureScaler {
  private min: Row = []

  private span: Row = []

  fit(X: Row[]): this {
    const dim = X[0]?.length ?? 0
    this.min = new Array<number>(dim).fill(Infinity)
    this.span = new Array<number>(dim).fill(1)
    for (const row of X) {
      for (let j = 0; j < dim; j += 1) this.min[j] = Math.min(this.min[j], row[j])
    }
    for (const row of X) {
      for (let j = 0; j < dim; j += 1) this.span[j] = Math.max(this.span[j], row[j] - this.min[j])
    }
    for (let j = 0; j < dim; j += 1) if (this.span[j] <= 1e-12) this.span[j] = 1
    return this
  }

  transform(X: Row[]): Row[] {
    return X.map((row) => row.map((value, j) => (value - this.min[j]) / this.span[j]))
  }

  fitTransform(X: Row[]): Row[] {
    return this.fit(X).transform(X)
  }
}
