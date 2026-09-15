/** Dataset loading: fetch + cache the JSON bundles exported by scripts/export_datasets.py. */

import type { DatasetBundle } from './types'

const cache = new Map<string, Promise<DatasetBundle>>()

export function loadDataset(id: string): Promise<DatasetBundle> {
  const existing = cache.get(id)
  if (existing) return existing
  const request = fetch(`${import.meta.env.BASE_URL}data/${id}.json`)
    .then((response) => {
      if (!response.ok) throw new Error(`数据集 ${id} 加载失败（HTTP ${response.status}）`)
      return response.json() as Promise<DatasetBundle>
    })
    .catch((error: unknown) => {
      cache.delete(id)
      throw error
    })
  cache.set(id, request)
  return request
}
