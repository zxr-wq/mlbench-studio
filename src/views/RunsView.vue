<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import { useLabStore } from '../stores/lab'

const router = useRouter()
const { runs } = useLabStore()
const query = ref('')
const modelFilter = ref('全部模型')

const models = computed(() => ['全部模型', ...new Set(runs.value.map((run) => run.model))])
const filteredRuns = computed(() => runs.value.filter((run) => {
  const matchQuery = `${run.id} ${run.model} ${run.dataset}`.toLowerCase().includes(query.value.toLowerCase())
  const matchModel = modelFilter.value === '全部模型' || run.model === modelFilter.value
  return matchQuery && matchModel
}))
</script>

<template>
  <div class="page runs-page">
    <header class="page-header split-header">
      <div><span class="overline">EXPERIMENTS / RUNS</span><h1>运行记录</h1><p>筛选、比较并回溯每一次模型实验。</p></div>
      <RouterLink to="/experiments/new" class="button primary"><AppIcon name="plus" :size="16" />新建实验</RouterLink>
    </header>

    <section class="runs-toolbar">
      <label class="search-box"><AppIcon name="search" :size="16" /><input v-model="query" placeholder="搜索实验、模型或数据集" /></label>
      <select v-model="modelFilter"><option v-for="model in models" :key="model">{{ model }}</option></select>
      <button class="button ghost compact-button"><AppIcon name="filter" :size="15" />更多筛选</button>
      <span class="row-count">{{ filteredRuns.length }} runs</span>
    </section>

    <section class="data-sheet">
      <div class="sheet-row sheet-head">
        <span>运行</span><span>模型</span><span>数据集</span><span>Accuracy</span><span>Macro F1</span><span>耗时</span><span>状态</span>
      </div>
      <button v-for="run in filteredRuns" :key="run.id" class="sheet-row" @click="router.push(`/runs/${run.id}`)">
        <span class="run-cell"><i></i><b>{{ run.id }}</b><small>{{ run.createdAt }}</small></span>
        <span><b>{{ run.model }}</b></span>
        <span>{{ run.dataset }}</span>
        <span class="metric-cell">{{ run.accuracy.toFixed(3) }}</span>
        <span class="metric-cell secondary">{{ run.f1.toFixed(3) }}</span>
        <span>{{ run.duration.toFixed(2) }}s</span>
        <span class="status-label"><i></i>完成</span>
      </button>
      <div v-if="!filteredRuns.length" class="empty-state">没有符合筛选条件的实验。</div>
    </section>
  </div>
</template>
