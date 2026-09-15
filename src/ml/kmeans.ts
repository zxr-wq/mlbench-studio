/**
 * K-Means with k-means++ seeding and multiple restarts.
 *
 * Used directly as an unsupervised model and, via a majority-vote mapping of
 * clusters to labels, as a baseline classifier for the benchmark matrix.
 */

import { range, squaredDistance, uniqueSorted } from './math'
import type { Random } from './math'
import { mulberry32 } from './preprocess'
import type { Row, SupervisedModel } from './types'

export interface KMeansFit {
  labels: number[]
  centroids: Row[]
  inertia: number
  iterations: number
}

function kmeansPlusPlusInit(X: Row[], k: number, rand: Random): Row[] {
  const centroids: Row[] = [X[Math.floor(rand() * X.length)].slice()]
  const nearest = new Float64Array(X.length).fill(Infinity)
  for (let c = 1; c < k; c += 1) {
    const latest = centroids[c - 1]
    let total = 0
    for (let i = 0; i < X.length; i += 1) {
      const d = squaredDistance(X[i], latest)
      if (d < nearest[i]) nearest[i] = d
      total += nearest[i]
    }
    if (total <= 0) {
      centroids.push(X[Math.floor(rand() * X.length)].slice())
      continue
    }
    const target = rand() * total
    let acc = 0
    let pick = X.length - 1
    for (let i = 0; i < X.length; i += 1) {
      acc += nearest[i]
      if (acc >= target) {
        pick = i
        break
      }
    }
    centroids.push(X[pick].slice())
  }
  return centroids
}

function kmeansSingle(X: Row[], k: number, rand: Random, maxIter: number, tol: number): KMeansFit {
  const n = X.length
  let centroids = kmeansPlusPlusInit(X, k, rand)
  const labels = new Array<number>(n).fill(0)
  let iterations = 0
  for (let iter = 0; iter < maxIter; iter += 1) {
    iterations = iter + 1
    let moved = false
    for (let i = 0; i < n; i += 1) {
      let best = 0
      let bestDistance = Infinity
      for (let c = 0; c < k; c += 1) {
        const d = squaredDistance(X[i], centroids[c])
        if (d < bestDistance) {
          bestDistance = d
          best = c
        }
      }
      if (labels[i] !== best) {
        labels[i] = best
        moved = true
      }
    }
    const nextCentroids: Row[] = []
    for (let c = 0; c < k; c += 1) {
      const dim = X[0]?.length ?? 0
      const sums = new Array<number>(dim).fill(0)
      let members = 0
      for (let i = 0; i < n; i += 1) {
        if (labels[i] !== c) continue
        members += 1
        for (let j = 0; j < dim; j += 1) sums[j] += X[i][j]
      }
      if (members > 0) nextCentroids.push(sums.map((value) => value / members))
      else nextCentroids.push(X[Math.floor(rand() * n)].slice())
    }
    let shift = 0
    for (let c = 0; c < k; c += 1) {
      for (let j = 0; j < centroids[c].length; j += 1) {
        shift = Math.max(shift, Math.abs(nextCentroids[c][j] - centroids[c][j]))
      }
    }
    centroids = nextCentroids
    if (!moved && shift < tol) break
  }
  let inertia = 0
  for (let i = 0; i < n; i += 1) inertia += squaredDistance(X[i], centroids[labels[i]])
  return { labels: labels.slice(), centroids, inertia, iterations }
}

export function kmeansFit(X: Row[], k: number, seed: number, nInit = 8, maxIter = 300, tol = 1e-6): KMeansFit {
  const rand = mulberry32(seed)
  let best: KMeansFit | null = null
  for (let init = 0; init < nInit; init += 1) {
    const fit = kmeansSingle(X, k, rand, maxIter, tol)
    if (!best || fit.inertia < best.inertia) best = fit
  }
  return best ?? { labels: [], centroids: [], inertia: 0, iterations: 0 }
}

/** Cluster-then-majority-vote classifier used for supervised comparison. */
export class KMeansClassifier implements SupervisedModel {
  private centroids: Row[] = []

  private mapping: number[] = []

  private classes: number[] = []

  trainLabels: number[] = []

  inertia = 0

  iterations = 0

  constructor(private readonly k: number, private readonly seed: number, private readonly nInit = 8) {}

  async fit(X: Row[], y: number[]): Promise<void> {
    this.classes = uniqueSorted(y)
    const targetClusters = Math.max(1, Math.min(this.k, X.length))
    const fit = kmeansFit(X, targetClusters, this.seed, this.nInit)
    this.centroids = fit.centroids
    this.trainLabels = fit.labels
    this.inertia = fit.inertia
    this.iterations = fit.iterations

    const labelCounts = new Map<number, number>()
    for (const value of y) labelCounts.set(value, (labelCounts.get(value) ?? 0) + 1)
    const globalMajority =
      [...labelCounts.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0]?.[0] ?? 0

    this.mapping = range(targetClusters).map((cluster) => {
      const tally = new Map<number, number>()
      for (let i = 0; i < y.length; i += 1) {
        if (fit.labels[i] !== cluster) continue
        tally.set(y[i], (tally.get(y[i]) ?? 0) + 1)
      }
      const sorted = [...tally.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0])
      return sorted[0]?.[0] ?? globalMajority
    })
  }

  predict(X: Row[]): number[] {
    return X.map((row) => {
      let best = 0
      let bestDistance = Infinity
      for (let c = 0; c < this.centroids.length; c += 1) {
        const d = squaredDistance(row, this.centroids[c])
        if (d < bestDistance) {
          bestDistance = d
          best = c
        }
      }
      return this.mapping[best] ?? this.classes[0] ?? 0
    })
  }
}
