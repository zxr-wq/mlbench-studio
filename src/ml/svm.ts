/**
 * Support Vector Machine trained with a simplified SMO routine (Platt).
 *
 * - Kernels: linear / rbf / poly with gamma = 'scale' (1 / (d * var(X))).
 * - Multiclass via one-vs-rest.
 * - The full kernel matrix is precomputed once per binary problem and the
 *   error vector f is updated incrementally, matching the validated
 *   prototype in scripts/prototype_validate.py.
 */

import { dot, squaredDistance, uniqueSorted } from './math'
import type { Row, SupervisedModel } from './types'

export type SvmKernel = 'linear' | 'rbf' | 'poly'

export interface SvmOptions {
  kernel?: SvmKernel
  c?: number
  degree?: number
  tol?: number
  maxIter?: number
}

export const yieldToUi = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0)
  })

interface KernelConfig {
  kernel: SvmKernel
  gamma: number
  degree: number
}

function kernelValue(a: Row, b: Row, cfg: KernelConfig): number {
  if (cfg.kernel === 'linear') return dot(a, b)
  if (cfg.kernel === 'poly') return (cfg.gamma * dot(a, b) + 1) ** cfg.degree
  return Math.exp(-cfg.gamma * squaredDistance(a, b))
}

/** gamma='scale', mirroring scikit-learn's default. */
function scaleGamma(X: Row[]): number {
  const n = X.length
  const dim = X[0]?.length ?? 0
  if (!n || !dim) return 1
  let total = 0
  for (let j = 0; j < dim; j += 1) {
    let mean = 0
    for (let i = 0; i < n; i += 1) mean += X[i][j]
    mean /= n
    let variance = 0
    for (let i = 0; i < n; i += 1) {
      const delta = X[i][j] - mean
      variance += delta * delta
    }
    total += variance / n
  }
  const meanVariance = total / dim
  return meanVariance > 1e-12 ? 1 / (dim * meanVariance) : 1
}

interface BinaryModel {
  supportVectors: Row[]
  coefficients: number[]
  intercept: number
}

function smoBinary(X: Row[], y: number[], cfg: KernelConfig, c: number, options: SvmOptions): BinaryModel {
  const n = X.length
  const tol = options.tol ?? 1e-3
  const eps = 1e-5
  const maxIter = options.maxIter ?? 40000

  const K = new Float64Array(n * n)
  for (let i = 0; i < n; i += 1) {
    for (let j = i; j < n; j += 1) {
      const value = kernelValue(X[i], X[j], cfg)
      K[i * n + j] = value
      K[j * n + i] = value
    }
  }

  const alpha = new Float64Array(n)
  const f = new Float64Array(n)
  for (let i = 0; i < n; i += 1) f[i] = -y[i]
  let b = 0
  let applied = 0
  let attempts = 0
  let streak = 0
  const attemptLimit = maxIter * 2 + 5000

  while (applied < maxIter && attempts < attemptLimit) {
    attempts += 1

    // i1: the largest KKT violation.
    const candidates: number[] = []
    let i1 = -1
    let worst = 0
    for (let i = 0; i < n; i += 1) {
      const r = y[i] * f[i]
      const violating = (r < -tol && alpha[i] < c - eps) || (r > tol && alpha[i] > eps)
      if (!violating) continue
      candidates.push(i)
      const size = Math.abs(r)
      if (size > worst) {
        worst = size
        i1 = i
      }
    }
    if (i1 < 0) break

    // i2: second-choice heuristic argmax |E1 - E2|, with a cycling fallback.
    const E1 = f[i1]
    let i2 = i1
    if (streak > 20) {
      i2 = candidates[(candidates.indexOf(i1) + 1 + attempts) % candidates.length]
    } else {
      let bestDiff = 0
      for (const j of candidates) {
        if (j === i1) continue
        const diff = Math.abs(E1 - f[j])
        if (diff > bestDiff) {
          bestDiff = diff
          i2 = j
        }
      }
      if (i2 === i1) i2 = candidates[(candidates.indexOf(i1) + 1) % candidates.length]
    }

    const y1 = y[i1]
    const y2 = y[i2]
    const a1o = alpha[i1]
    const a2o = alpha[i2]
    const s = y1 * y2
    let L: number
    let H: number
    if (s < 0) {
      L = Math.max(0, a2o - a1o)
      H = Math.min(c, c + a2o - a1o)
    } else {
      L = Math.max(0, a1o + a2o - c)
      H = Math.min(c, a1o + a2o)
    }
    if (L === H) {
      streak += 1
      continue
    }
    const k11 = K[i1 * n + i1]
    const k12 = K[i1 * n + i2]
    const k22 = K[i2 * n + i2]
    const eta = 2 * k12 - k11 - k22
    if (eta >= -1e-12) {
      streak += 1
      continue
    }
    const E2 = f[i2]
    const a2n = Math.min(Math.max(a2o - (y2 * (E1 - E2)) / eta, L), H)
    if (Math.abs(a2n - a2o) < eps * (a2n + a2o + eps)) {
      streak += 1
      continue
    }
    const a1n = a1o + s * (a2o - a2n)
    alpha[i1] = a1n
    alpha[i2] = a2n

    // Threshold update (sign matters: b absorbs -E, not +E).
    const da1 = a1n - a1o
    const da2 = a2n - a2o
    const b1 = b - E1 - y1 * da1 * k11 - y2 * da2 * k12
    const b2 = b - E2 - y1 * da1 * k12 - y2 * da2 * k22
    let bNew: number
    if (a1n > eps && a1n < c - eps) bNew = b1
    else if (a2n > eps && a2n < c - eps) bNew = b2
    else bNew = (b1 + b2) / 2

    const dA1 = y1 * da1
    const dA2 = y2 * da2
    const dB = bNew - b
    for (let k = 0; k < n; k += 1) f[k] += dA1 * K[i1 * n + k] + dA2 * K[i2 * n + k] + dB
    b = bNew
    applied += 1
    streak = 0
  }

  const supportVectors: Row[] = []
  const coefficients: number[] = []
  for (let i = 0; i < n; i += 1) {
    if (alpha[i] > 1e-6) {
      supportVectors.push(X[i])
      coefficients.push(alpha[i] * y[i])
    }
  }
  return { supportVectors, coefficients, intercept: b }
}

export class SvmClassifier implements SupervisedModel {
  private classes: number[] = []

  private models: BinaryModel[] = []

  private cfg: KernelConfig = { kernel: 'rbf', gamma: 1, degree: 3 }

  supportVectorCount = 0

  constructor(private readonly options: SvmOptions = {}) {}

  async fit(X: Row[], y: number[]): Promise<void> {
    const kernel = this.options.kernel ?? 'rbf'
    const c = this.options.c ?? 1
    this.cfg = { kernel, gamma: scaleGamma(X), degree: this.options.degree ?? 3 }
    this.classes = uniqueSorted(y)
    this.models = []
    this.supportVectorCount = 0
    for (const cls of this.classes) {
      const binaryTarget = y.map((value) => (value === cls ? 1 : -1))
      await yieldToUi()
      const model = smoBinary(X, binaryTarget, this.cfg, c, this.options)
      this.models.push(model)
      this.supportVectorCount += model.supportVectors.length
    }
  }

  predict(X: Row[]): number[] {
    const fallback = this.classes[0] ?? 0
    return X.map((row) => {
      let bestClass = fallback
      let bestScore = -Infinity
      for (let idx = 0; idx < this.models.length; idx += 1) {
        const model = this.models[idx]
        let score = model.intercept
        for (let s = 0; s < model.supportVectors.length; s += 1) {
          score += model.coefficients[s] * kernelValue(row, model.supportVectors[s], this.cfg)
        }
        if (score > bestScore) {
          bestScore = score
          bestClass = this.classes[idx]
        }
      }
      return bestClass
    })
  }
}
