/** Shared type vocabulary for the in-browser ML core. */

export type Row = number[]

export interface DatasetBundle {
  id: string
  name: string
  featureNames: string[]
  targetNames: string[]
  data: Row[]
  target: number[]
}

export interface SupervisedModel {
  fit(X: Row[], y: number[]): Promise<void>
  predict(X: Row[]): number[]
}

export type StageReporter = (stage: string, message: string) => void
