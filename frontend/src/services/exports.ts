import type { ExperimentResult } from '../types/benchmark'

export function downloadFile(filename: string, data: string, type = 'application/json') {
  const blob = new Blob([type.startsWith('text/csv') ? '\uFEFF' + data : data], { type })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename
  anchor.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
export const metricName = (key: string) => ({ accuracy:'Accuracy', f1:'Macro F1', macro_f1:'Macro F1', precision:'Macro Precision', recall:'Macro Recall', r2:'R²', rmse:'RMSE', mse:'MSE', mae:'MAE', silhouette_score:'Silhouette', inertia:'Inertia', explained_variance_ratio:'单个主成分方差', cumulative_explained_variance:'累计解释方差', roc_auc:'ROC-AUC' }[key] ?? key)
export function metricValue(run: ExperimentResult, key: string) {
  const value = run.metrics?.[key]
  return value === undefined ? undefined : Number(Array.isArray(value) ? value.at(-1) : value)
}
export function metricsCsv(runs: ExperimentResult[]) {
  const rows: Array<Array<string | number>> = [['experiment','model','dataset','implementation','metric','mean','std','seed','strategy','total_runs']]
  for (const run of runs) for (const [key, value] of Object.entries(run.metrics ?? {})) {
    const values = Array.isArray(value) ? value : [value]
    const deviation = run.metricStd?.[key]
    values.forEach((item,i) => rows.push([run.id,run.model,run.dataset,run.implementation ?? '',Array.isArray(value) ? `${key}[${i+1}]` : key,item,Array.isArray(deviation) ? deviation[i] ?? 0 : deviation ?? 0,run.evaluation?.seeds.join('/') ?? '',run.evaluation?.strategy ?? '',run.evaluation?.total_runs ?? 1]))
  }
  return rows.map((row)=>row.map((value)=>`"${String(value).replaceAll('"','""')}"`).join(',')).join('\r\n')
}
