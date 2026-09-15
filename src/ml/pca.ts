/**
 * Principal Component Analysis via cyclic Jacobi eigen-decomposition of the
 * covariance matrix (validated against numpy.linalg.eigh to machine precision).
 */

import type { Row } from './types'

export interface PcaOptions {
  varianceRatio?: number
  maxComponents?: number
}

export interface PcaModel {
  mean: Row
  /** Eigenvectors sorted by decreasing eigenvalue; components[j] is the j-th direction. */
  components: Row[]
  eigenvalues: number[]
  explainedVarianceRatio: number[]
  nComponents: number
}

interface EigenDecomposition {
  eigenvalues: number[]
  /** vectors[j] is the eigenvector belonging to eigenvalues[j]. */
  vectors: Row[]
}

function jacobiEigen(matrix: number[][]): EigenDecomposition {
  const d = matrix.length
  const a = matrix.map((row) => row.slice())
  // v[j][i] = i-th coordinate of the j-th eigenvector (column layout).
  const v: Row[] = Array.from({ length: d }, (_, j) => {
    const column = new Array<number>(d).fill(0)
    column[j] = 1
    return column
  })

  for (let sweep = 0; sweep < 60; sweep += 1) {
    let offSquare = 0
    for (let p = 0; p < d; p += 1) {
      for (let q = p + 1; q < d; q += 1) offSquare += a[p][q] * a[p][q]
    }
    if (Math.sqrt(offSquare * 2) < 1e-10) break
    for (let p = 0; p < d - 1; p += 1) {
      for (let q = p + 1; q < d; q += 1) {
        const apq = a[p][q]
        if (Math.abs(apq) < 1e-14) continue
        const theta = (a[q][q] - a[p][p]) / (2 * apq)
        const sign = theta >= 0 ? 1 : -1
        const t = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const cos = 1 / Math.sqrt(t * t + 1)
        const sin = t * cos
        for (let j = 0; j < d; j += 1) {
          const vp = a[p][j]
          const vq = a[q][j]
          a[p][j] = cos * vp - sin * vq
          a[q][j] = sin * vp + cos * vq
        }
        for (let i = 0; i < d; i += 1) {
          const vp = a[i][p]
          const vq = a[i][q]
          a[i][p] = cos * vp - sin * vq
          a[i][q] = sin * vp + cos * vq
        }
        for (let i = 0; i < d; i += 1) {
          const vp = v[p][i]
          const vq = v[q][i]
          v[p][i] = cos * vp - sin * vq
          v[q][i] = sin * vp + cos * vq
        }
      }
    }
  }

  const eigenvalues = range0(d).map((i) => a[i][i])
  return { eigenvalues, vectors: v }
}

function range0(count: number): number[] {
  return Array.from({ length: count }, (_, index) => index)
}

export function fitPca(X: Row[], options: PcaOptions = {}): PcaModel {
  const n = X.length
  const dim = X[0]?.length ?? 0
  const mean = new Array<number>(dim).fill(0)
  for (const row of X) for (let j = 0; j < dim; j += 1) mean[j] += row[j]
  for (let j = 0; j < dim; j += 1) mean[j] /= n || 1

  const covariance: number[][] = Array.from({ length: dim }, () => new Array<number>(dim).fill(0))
  if (n > 1) {
    for (const row of X) {
      const centered = row.map((value, j) => value - mean[j])
      for (let i = 0; i < dim; i += 1) {
        for (let j = i; j < dim; j += 1) covariance[i][j] += centered[i] * centered[j]
      }
    }
    for (let i = 0; i < dim; i += 1) {
      for (let j = i; j < dim; j += 1) {
        covariance[i][j] /= n - 1
        covariance[j][i] = covariance[i][j]
      }
    }
  }

  const { eigenvalues: raw, vectors } = jacobiEigen(covariance)
  const order = range0(dim).sort((a, b) => raw[b] - raw[a])
  const eigenvalues = order.map((index) => Math.max(raw[index], 0))
  const total = eigenvalues.reduce((sum, value) => sum + value, 0)
  const explainedVarianceRatio = total > 0 ? eigenvalues.map((value) => value / total) : eigenvalues.map(() => 0)

  let nComponents = dim
  if (options.varianceRatio !== undefined) {
    let acc = 0
    nComponents = dim
    for (let j = 0; j < dim; j += 1) {
      acc += explainedVarianceRatio[j]
      if (acc >= options.varianceRatio) {
        nComponents = j + 1
        break
      }
    }
  } else if (options.maxComponents !== undefined) {
    nComponents = Math.min(options.maxComponents, dim)
  }
  nComponents = Math.max(1, nComponents)

  return {
    mean,
    components: order.slice(0, nComponents).map((index) => vectors[index]),
    eigenvalues: eigenvalues.slice(0, nComponents),
    explainedVarianceRatio: explainedVarianceRatio.slice(0, nComponents),
    nComponents,
  }
}

export function pcaTransform(model: PcaModel, X: Row[]): Row[] {
  const dim = model.mean.length
  return X.map((row) =>
    model.components.map((component) => {
      let sum = 0
      for (let j = 0; j < dim; j += 1) sum += (row[j] - model.mean[j]) * component[j]
      return sum
    }),
  )
}
