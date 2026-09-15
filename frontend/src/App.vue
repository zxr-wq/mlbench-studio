<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import AlgorithmLibrary from './components/AlgorithmLibrary.vue'
import ArchitectureView from './components/ArchitectureView.vue'
import ConfigPanel from './components/ConfigPanel.vue'
import HistoryView from './components/HistoryView.vue'
import ResultsDashboard from './components/ResultsDashboard.vue'
import BenchmarkView from './components/BenchmarkView.vue'
import { experimentApi, subscribeToExperiment } from './services/api'
import { demoMode } from './services/demo'

const views = [
  { id: 'lab', label: '实验工作台', index: '01' },
  { id: 'algorithms', label: '算法图鉴', index: '02' },
  { id: 'history', label: '运行记录', index: '03' },
  { id: 'benchmark', label: '性能对比', index: '04' },
  { id: 'architecture', label: '架构说明', index: '05' },
]

const initialView = new URLSearchParams(window.location.search).get('view')
const activeView = ref(views.some((item) => item.id === initialView) ? initialView : 'lab')
const datasets = ref([])
const algorithms = ref([])
const history = ref([])
const experiment = ref(null)
const loading = ref(true)
const running = ref(false)
const message = ref('')
let unsubscribeExperiment = null

const config = reactive({
  dataset_id: 'iris',
  model_ids: ['knn', 'naive_bayes'],
  implementations: ['scratch', 'sklearn'],
  split_method: 'stratified',
  test_size: 0.2,
  folds: 5,
  standardize: true,
  seed: 42,
  parameters: {},
  metrics: ['accuracy', 'precision', 'recall', 'f1'],
})

const pageTitle = computed(() => views.find((item) => item.id === activeView.value)?.label || '实验工作台')

async function loadCatalog() {
  loading.value = true
  try {
    const [datasetItems, algorithmItems, historyItems] = await Promise.all([
      experimentApi.listDatasets(),
      experimentApi.listAlgorithms(),
      experimentApi.listExperiments(),
    ])
    datasets.value = datasetItems
    algorithms.value = algorithmItems
    history.value = historyItems
    for (const algorithm of algorithmItems) {
      config.parameters[algorithm.id] = { ...algorithm.defaults }
    }
  } catch (error) {
    message.value = error.message
  } finally {
    loading.value = false
  }
}

function closeLiveConnections() {
  unsubscribeExperiment?.()
  unsubscribeExperiment = null
}

function handleRecord(record) {
  experiment.value = record
  running.value = ['queued', 'running'].includes(record.status)
  if (!running.value) {
    closeLiveConnections()
    loadHistory()
    if (record.status === 'failed') message.value = record.error || '实验运行失败'
  }
}

function connectExperiment(id) {
  closeLiveConnections()
  unsubscribeExperiment = subscribeToExperiment(
    id,
    handleRecord,
    (error) => { message.value = error.message },
  )
}

async function runExperiment() {
  if (running.value) return
  message.value = ''
  running.value = true
  experiment.value = {
    status: 'queued',
    progress: 0,
    stage: '正在创建实验',
    results: [],
    config: { ...config },
  }
  try {
    const created = await experimentApi.createExperiment(config)
    connectExperiment(created.id)
  } catch (error) {
    running.value = false
    message.value = error.message
  }
}

async function loadHistory() {
  try {
    history.value = await experimentApi.listExperiments()
  } catch (error) {
    message.value = error.message
  }
}

async function openHistoryItem(id) {
  try {
    experiment.value = await experimentApi.getExperiment(id)
    activeView.value = 'lab'
  } catch (error) {
    message.value = error.message
  }
}

onMounted(async () => {
  await loadCatalog()
  const experimentId = new URLSearchParams(window.location.search).get('experiment')
  if (experimentId) await openHistoryItem(experimentId)
})
onBeforeUnmount(closeLiveConnections)
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <button class="brand" aria-label="返回实验工作台" @click="activeView = 'lab'">
        <span class="brand-mark"><i></i><i></i><i></i></span>
        <span><strong>智模工坊</strong><small>ML STUDIO</small></span>
      </button>

      <nav class="main-nav" aria-label="主导航">
        <button
          v-for="view in views"
          :key="view.id"
          :class="{ active: activeView === view.id }"
          @click="activeView = view.id"
        >
          <span>{{ view.index }}</span>{{ view.label }}
        </button>
      </nav>

      <div class="sidebar-note">
        <span class="eyebrow">课程实践 · 2026</span>
        <p>从原理出发，用实验验证每一个判断。</p>
        <div class="note-line"><span></span><em>5 个自主实现模型</em></div>
      </div>
    </aside>

    <main>
      <header class="topbar">
        <div>
          <span class="eyebrow">MACHINE LEARNING LABORATORY</span>
          <h1>{{ pageTitle }}</h1>
        </div>
        <div class="topbar-actions">
          <span class="status-dot"><i></i>{{ demoMode ? '静态展示模式' : '本地计算引擎' }}</span>
          <a href="/docs" target="_blank" rel="noreferrer">API 文档 ↗</a>
        </div>
      </header>

      <div v-if="message" class="toast" role="alert">
        <span>{{ message }}</span><button @click="message = ''">×</button>
      </div>

      <section v-if="loading" class="loading-state">
        <span class="loader"></span>
        <p>正在准备实验环境…</p>
      </section>

      <section v-else-if="activeView === 'lab'" class="lab-layout">
        <ConfigPanel
          v-model="config"
          :datasets="datasets"
          :algorithms="algorithms"
          :running="running"
          @run="runExperiment"
        />
        <ResultsDashboard :experiment="experiment" :algorithms="algorithms" />
      </section>

      <AlgorithmLibrary v-else-if="activeView === 'algorithms'" :algorithms="algorithms" />
      <HistoryView
        v-else-if="activeView === 'history'"
        :history="history"
        @refresh="loadHistory"
        @open="openHistoryItem"
      />
      <BenchmarkView v-else-if="activeView === 'benchmark'" :history="history" />
      <ArchitectureView v-else-if="activeView === 'architecture'" />
    </main>
  </div>
</template>
