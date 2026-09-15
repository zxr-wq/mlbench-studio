/** Small numeric helpers shared by the ML core. */

export type Random = () => number

export function dot(a: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i]
  return sum
}

export function squaredDistance(a: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < a.length; i += 1) {
    const delta = a[i] - b[i]
    sum += delta * delta
  }
  return sum
}

export function uniqueSorted(values: number[]): number[] {
  return [...new Set(values)].sort((a, b) => a - b)
}

export function shuffleInPlace<T>(items: T[], rand: Random): void {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1))
    const tmp = items[i]
    items[i] = items[j]
    items[j] = tmp
  }
}

export function range(count: number): number[] {
  return Array.from({ length: count }, (_, index) => index)
}
