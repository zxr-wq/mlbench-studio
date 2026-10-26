<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import { algorithms, datasets } from '../data/catalog'
import { useLabStore } from '../stores/lab'
import { downloadFile, metricsCsv, metricName } from '../services/exports'

const router = useRouter()
const { runs, importResults, isHydrating } = useLabStore()
const importMessage=ref('')
async function importFile(event:Event) {
  const input=event.target as HTMLInputElement
  const file=input.files?.[0]; if(!file) return
  try {
    if(file.size>30*1024*1024) throw new Error('文件应小于 30MB。')
    const count=await importResults(await file.text()); importMessage.value=`已成功导入 ${count} 条实验记录。`
  } catch(error) {importMessage.value=error instanceof Error ? error.message : '导入失败'}
  input.value=''
}
function exportAll(kind:'csv'|'json') { downloadFile(`mlbench-all-results.${kind}`,kind==='csv' ? metricsCsv(filteredRuns.value) : JSON.stringify({schema_version:2,exported_at:new Date().toISOString(),experiments:filteredRuns.value},null,2),kind==='csv' ? 'text/csv;charset=utf-8' : 'application/json') }
const query = ref('')
const modelFilter = ref('全部模型')

const models = computed(() => ['全部模型', ...new Set(runs.value.map((run) => run.model))])
const modelName = (id: string) => algorithms.find((item) => item.id === id)?.name ?? id
const datasetName = (id: string) => datasets.find((item) => item.id === id)?.name ?? id
const filteredRuns = computed(() => runs.value.filter((run) => {
  const matchQuery = `${run.id} ${run.model} ${modelName(run.model)} ${run.dataset} ${datasetName(run.dataset)}`.toLowerCase().includes(query.value.toLowerCase())
  const matchModel = modelFilter.value === '全部模型' || run.model === modelFilter.value
  return matchQuery && matchModel
}))
const label = (run: { primaryMetric?: string }, fallback: string) => metricName(run.primaryMetric ?? fallback)
const value = (run: { primaryValue?: number, accuracy: number }, fallback: number) => Number(run.primaryValue ?? fallback).toFixed(3)
</script>

<template>
  <div class="page runs-page">
    <header class="page-header split-header">
      <div><span class="overline">EXPERIMENTS / RUNS</span><h1>运行记录</h1><p>筛选、比较并回溯每一次模型实验。</p></div>
      <RouterLink to="/experiments/new" class="button primary"><AppIcon name="plus" :size="16" />新建实验</RouterLink>
    </header>
    <div class="run-tools"><label class="button ghost import-control">导入结果 JSON<input type="file" accept=".json,application/json" @change="importFile" :disabled="isHydrating"/></label><button class="button ghost" :disabled="isHydrating || !filteredRuns.length" @click="exportAll('json')">备份全部筛选结果</button><button class="button ghost" :disabled="isHydrating || !filteredRuns.length" @click="exportAll('csv')">导出指标 CSV</button><RouterLink class="button ghost" to="/benchmark">进入 Benchmark 对比</RouterLink></div><p class="run-message" role="status">{{ isHydrating ? '正在读取完整实验记录…' : importMessage || '记录保存在此浏览器；更换电脑前请备份完整 JSON，再在新浏览器导入。' }}</p>

    <section class="runs-toolbar">
      <label class="search-box"><AppIcon name="search" :size="16" /><input v-model="query" placeholder="搜索实验、模型或数据集" /></label>
      <select v-model="modelFilter"><option v-for="model in models" :key="model" :value="model">{{ model === '全部模型' ? model : modelName(model) }}</option></select>
      <span class="row-count">{{ filteredRuns.length }} runs</span>
    </section>

    <section class="data-sheet">
      <div class="sheet-row sheet-head">
        <span>运行</span><span>模型</span><span>数据集</span><span>主要指标</span><span>辅助指标</span><span>耗时</span><span>状态</span>
      </div>
      <button v-for="run in filteredRuns" :key="run.id" class="sheet-row" @click="router.push(`/runs/${run.id}`)">
        <span class="run-cell"><i></i><b>{{ run.id }}</b><small>{{ run.createdAt }}</small></span>
        <span><b>{{ modelName(run.model) }}</b></span>
        <span>{{ datasetName(run.dataset) }}</span>
        <span class="metric-cell"><small>{{ label(run, 'accuracy') }}</small>{{ value(run, run.accuracy) }}</span>
        <span class="metric-cell secondary"><small>{{ run.secondaryMetric ? metricName(run.secondaryMetric) : '—' }}</small>{{ run.secondaryValue?.toFixed(3) ?? '—' }}</span>
        <span>{{ run.duration.toFixed(2) }}s</span>
        <span class="status-label"><i></i>{{ run.status === 'completed' ? '完成' : run.status === 'failed' ? '失败' : '运行中' }}</span>
      </button>
      <div v-if="!filteredRuns.length" class="empty-state"><b>{{ runs.length ? '没有符合筛选条件的实验' : '还没有真实实验记录' }}</b><p>{{ runs.length ? '请调整关键词或模型筛选。' : '实验完成后，配置、指标和可视化会出现在这里。' }}</p><RouterLink v-if="!runs.length" to="/experiments/new" class="button primary">创建实验</RouterLink></div>
    </section>
  </div>
</template>
