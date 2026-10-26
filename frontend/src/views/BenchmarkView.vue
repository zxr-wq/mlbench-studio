<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { useRouter } from 'vue-router'
import { algorithms, datasets } from '../data/catalog'
import { useLabStore } from '../stores/lab'
import { downloadFile, metricName, metricValue, metricsCsv } from '../services/exports'
import type { ExperimentConfig, ExperimentResult } from '../types/benchmark'
const router=useRouter()
const { runs, defaultConfig, runExperiment, isRunning }=useLabStore()
const selectedModels=ref(['knn','naive_bayes','decision_tree'])
const selectedDatasets=ref(['iris','wine'])
const versions=ref(['scratch'])
const repeats=ref<1|3|5>(1)
const batchRunning=ref(false)
const stopRequested=ref(false)
const errors=ref<string[]>([])
const batchProgress=ref('')
const task=ref('classification')
const metric=ref('accuracy')
const cohort=ref('')
const seed=ref(42)
onBeforeUnmount(()=>{stopRequested.value=true})
const taskFor=(model:string)=> model==='linear_regression' ? 'regression' : model==='kmeans' ? 'clustering' : model==='pca' ? 'dimensionality_reduction' : 'classification'
const metricsFor=(type:string)=>type==='classification' ? ['accuracy','f1','precision','recall'] : type==='regression' ? ['r2','rmse','mae','mse'] : type==='clustering' ? ['silhouette_score','inertia'] : ['cumulative_explained_variance']
const pairs=computed(()=>selectedDatasets.value.flatMap((dataset)=>selectedModels.value.filter((model)=> !['linear_regression','kmeans','pca'].includes(model) ? !['diabetes','friedman1'].includes(dataset) : model!=='linear_regression' || ['diabetes','friedman1'].includes(dataset)).flatMap((model)=>versions.value.map((implementation)=>({dataset,model,implementation})))))
function taskChanged() { metric.value=metricsFor(task.value)[0]; cohort.value='' }
function cohortKey(run:ExperimentResult) {
  const config=run.config as any
  if(!config?.split || !config?.evaluation) return 'legacy-unverified'
  return JSON.stringify({ split:config.split, preprocessing:config.preprocessing, evaluation:config.evaluation, k:run.model==='kmeans' ? config.model?.params?.k : undefined, dimensions:run.model==='pca' ? config.model?.params?.n_components : undefined })
}
const taskRuns=computed(()=>runs.value.filter((run)=>run.status==='completed' && run.taskType===task.value && run.metrics?.[metric.value]!=null))
const cohorts=computed(()=>[...new Set(taskRuns.value.map(cohortKey))])
const activeCohort=computed(()=>cohort.value || cohorts.value[0] || '')
const comparable=computed(()=>taskRuns.value.filter((run)=>cohortKey(run)===activeCohort.value))
const columns=computed(()=>[...new Set(comparable.value.map((run)=>run.dataset))])
const rows=computed(()=>[...new Set(comparable.value.map((run)=>`${run.model}|${run.implementation}`))])
const cell=(key:string,dataset:string)=>comparable.value.find((run)=>`${run.model}|${run.implementation}`===key && run.dataset===dataset)
const latest=computed(()=>rows.value.flatMap((key)=>columns.value.flatMap((dataset)=>{const run=cell(key,dataset); return run ? [run] : []})))
const modelName=(id:string)=>algorithms.find((item)=>item.id===id)?.name ?? id
const datasetName=(id:string)=>datasets.find((item)=>item.id===id)?.name ?? id
function cohortLabel(key:string) {
  if(key==='legacy-unverified') return '旧版记录 · 配置未核实，仅供查看'
  const parsed=JSON.parse(key)
  return `${parsed.evaluation.strategy==='kfold' ? `${parsed.evaluation.n_splits} 折` : `测试 ${Math.round(parsed.split.test_size*100)}%`} · seed ${parsed.split.random_state} · ${parsed.evaluation.repetitions} 组 · ${parsed.preprocessing.join('+') || '无预处理'}${parsed.k ? ` · K=${parsed.k}` : ''}${parsed.dimensions ? ` · PC=${parsed.dimensions}` : ''}`
}
function heat(run:ExperimentResult|undefined,dataset:string) {
  if(!run) return { background:'#eeeee8',color:'#88917e' }
  const values=comparable.value.filter((row)=>row.dataset===dataset).map((row)=>metricValue(row,metric.value)!).filter(Number.isFinite)
  const lo=Math.min(...values),hi=Math.max(...values),current=metricValue(run,metric.value)!
  let score=hi===lo ? .5 : (current-lo)/(hi-lo)
  if(['rmse','mae','mse','inertia'].includes(metric.value)) score=1-score
  return { background:`rgba(53,115,72,${.08+.75*score})`,color:score>.6 ? '#fff' : '#294732' }
}
async function startBatch() {
  if(batchRunning.value || isRunning.value) return
  if(!Number.isInteger(seed.value)||seed.value<0||seed.value>4294967290) { errors.value=['随机种子需为 0 到 4294967290 的整数']; return }
  const jobs=[...pairs.value]
  batchRunning.value=true; stopRequested.value=false; errors.value=[]
  try {
    for(let i=0;i<jobs.length;i++) {
      if(stopRequested.value) break
      const job=jobs[i], type=taskFor(job.model)
      batchProgress.value=`${i+1} / ${jobs.length} · ${modelName(job.model)} × ${datasetName(job.dataset)} · ${job.implementation}`
      const config:ExperimentConfig={...defaultConfig,dataset:job.dataset,model:job.model,implementation:job.implementation as 'scratch'|'sklearn',seed:seed.value,stabilityRuns:repeats.value,split:'stratified_80_20',preprocessing:'standard_scaler',k:job.model==='kmeans' ? Math.max(2,Math.min(8,datasets.find((d)=>d.id===job.dataset)?.classes ?? 3)) : 5,metrics:metricsFor(type).map((m)=>m==='f1' ? 'macro_f1' : m)}
      if(type==='dimensionality_reduction') config.metrics=['explained_variance_ratio','cumulative_explained_variance']
      try { await runExperiment(config) } catch(error) { errors.value.push(`${modelName(job.model)} × ${job.dataset}: ${error instanceof Error ? error.message : '运行失败'}`) }
    }
    batchProgress.value=stopRequested.value ? '已在当前实验完成后停止，已完成结果均已保存。' : `批量实验完成，${errors.value.length} 项失败。`
    cohort.value=''
  } finally { batchRunning.value=false }
}
function exportResults(kind:'csv'|'json') { downloadFile(`mlbench-benchmark.${kind}`,kind==='csv' ? metricsCsv(latest.value) : JSON.stringify({schema_version:2,exported_at:new Date().toISOString(),experiments:latest.value},null,2),kind==='csv' ? 'text/csv;charset=utf-8' : 'application/json') }
</script>

<template>
  <div class="page benchmark-page">
    <header class="page-header"><span class="overline">BENCHMARK / EVIDENCE</span><h1>让模型在相同条件下比较</h1><p>选择算法、数据集与实现，运行真实批量实验；下方按任务与相同评价条件分组显示结果。</p></header>
    <section class="batch-config" aria-label="批量实验设置">
      <fieldset :disabled="batchRunning"><legend>算法（10 种）</legend><div class="batch-choices"><label v-for="item in algorithms" :key="item.id"><input type="checkbox" :value="item.id" v-model="selectedModels"/>{{ item.name }}</label></div></fieldset>
      <fieldset :disabled="batchRunning"><legend>数据集</legend><div class="batch-choices"><label v-for="item in datasets" :key="item.id"><input type="checkbox" :value="item.id" v-model="selectedDatasets"/>{{ item.name }}</label></div></fieldset>
      <div class="chart-controls batch-actions"><label><input type="checkbox" value="scratch" v-model="versions" :disabled="batchRunning"/>Scratch</label><label><input type="checkbox" value="sklearn" v-model="versions" :disabled="batchRunning"/>sklearn 对照</label><label>重复 <select v-model.number="repeats" :disabled="batchRunning"><option :value="1">1 次</option><option :value="3">3 组种子</option><option :value="5">5 组种子</option></select></label><label>起始种子 <input type="number" v-model.number="seed" min="0" max="4294967290" :disabled="batchRunning"/></label></div>
      <div class="batch-actions"><button class="button primary" :disabled="batchRunning || isRunning || !pairs.length" @click="startBatch">运行 {{ pairs.length }} 组实验 · 共 {{ pairs.length*repeats }} 次训练</button><button v-if="batchRunning" class="button ghost" :disabled="stopRequested" @click="stopRequested=true">{{ stopRequested ? '当前完成后停止' : '完成当前后停止' }}</button><span class="batch-status" role="status">{{ batchProgress }}</span></div>
      <p class="run-message">统一使用 Standard Scaler、80/20 划分和所选连续随机种子；不兼容的算法与数据集组合自动跳过。Scratch SVM 在较大数据集上耗时较长。不同算法各有自己的参数，点击热力图单元格可核对完整配置。</p>
      <ul v-if="errors.length" class="batch-errors"><li v-for="error in errors" :key="error">{{ error }}</li></ul>
    </section>
    <section class="benchmark-filters"><div class="chart-controls"><label>任务 <select v-model="task" @change="taskChanged"><option value="classification">分类</option><option value="regression">回归</option><option value="clustering">聚类</option><option value="dimensionality_reduction">降维</option></select></label><label>指标 <select v-model="metric" @change="cohort=''"><option v-for="item in metricsFor(task)" :key="item" :value="item">{{ metricName(item) }}</option></select></label><label>评价条件 <select v-model="cohort"><option value="">最新一组条件</option><option v-for="key in cohorts" :key="key" :value="key">{{ cohortLabel(key) }}</option></select></label><button class="button ghost" :disabled="!latest.length" @click="exportResults('csv')">导出比较 CSV</button><button class="button ghost" :disabled="!latest.length" @click="exportResults('json')">下载完整实验集</button></div><p class="run-message">{{ activeCohort ? cohortLabel(activeCohort) : '完成一次实验后即可对比。' }}。每个“算法 × 实现 × 数据集”显示该条件下最新一次运行；颜色仅在同一数据集列内比较，深色较优。缺失项表示尚未运行。</p></section>
    <div v-if="latest.length" class="benchmark-matrix"><table class="benchmark-table"><thead><tr><th>算法 / 实现</th><th v-for="dataset in columns" :key="dataset">{{ datasetName(dataset) }}</th></tr></thead><tbody><tr v-for="key in rows" :key="key"><th>{{ modelName(key.split('|')[0]) }}<br/><small>{{ key.split('|')[1] }}</small></th><td v-for="dataset in columns" :key="dataset" :style="heat(cell(key,dataset),dataset)"><button :disabled="!cell(key,dataset)" @click="router.push(`/runs/${cell(key,dataset)!.id}`)"><template v-if="cell(key,dataset)"><strong>{{ metricValue(cell(key,dataset)!,metric)?.toFixed(3) }}</strong><small>{{ metricName(metric) }} · {{ cell(key,dataset)!.evaluation?.total_runs ?? 1 }} 次运行</small></template><span v-else>— 未运行</span></button></td></tr></tbody></table></div>
    <div v-else class="empty-evidence">当前条件下还没有 {{ metricName(metric) }} 结果。<br/>上方批量运行或在创建实验中运行模型后，这里会显示真实对比。</div>
  </div>
</template>
