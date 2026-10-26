<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import AppIcon from './components/AppIcon.vue'
import { checkBackendHealth } from './services/benchmark'
import { algorithms, datasets } from './data/catalog'
import { useLabStore } from './stores/lab'

const route = useRoute()
const { isRunning, progress, lastMessage, runs } = useLabStore()
const collapsed = ref(false)
const searchOpen = ref(false)
const searchQuery = ref('')
const backendStatus = ref<'checking' | 'online' | 'offline'>('checking')
let healthTimer: number | undefined

const nav = [
  { label: '概览', to: '/', icon: 'overview' },
  { label: '创建实验', to: '/experiments/new', icon: 'flask' },
  { label: '运行记录', to: '/runs', icon: 'runs' },
  { label: 'Benchmark', to: '/benchmark', icon: 'runs' },
  { label: '算法库', to: '/algorithms', icon: 'algorithm' },
  { label: '数据集', to: '/datasets', icon: 'dataset' },
]

const pageTitle = computed(() => String(route.meta.title || 'MLBench'))
const searchResults=computed(()=>[
  ...nav.map((item)=>({to:item.to,title:item.label,type:'页面'})),
  ...algorithms.map((item)=>({to:`/experiments/new?model=${item.id}`,title:item.name,type:'算法'})),
  ...datasets.map((item)=>({to:`/experiments/new?dataset=${item.id}`,title:item.name,type:'数据集'})),
  ...runs.value.map((item)=>({to:`/runs/${item.id}`,title:`${item.id} ${item.model} ${item.dataset}`,type:'实验'})),
].filter((item)=>item.title.toLowerCase().includes(searchQuery.value.toLowerCase())).slice(0,12))
function shortcut(event:KeyboardEvent) { if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='k') { event.preventDefault(); searchOpen.value=!searchOpen.value } if(event.key==='Escape') searchOpen.value=false }

async function refreshBackendStatus() {
  backendStatus.value = await checkBackendHealth() ? 'online' : 'offline'
}

onMounted(() => {
  window.addEventListener('keydown',shortcut)
  refreshBackendStatus()
  healthTimer = window.setInterval(refreshBackendStatus, 15000)
})
onBeforeUnmount(() => {window.clearInterval(healthTimer); window.removeEventListener('keydown',shortcut)})
</script>

<template>
  <div class="app-shell" :class="{ 'is-collapsed': collapsed }">
    <aside class="app-sidebar">
      <RouterLink to="/" class="wordmark">
        <span class="wordmark-symbol">M</span>
        <span class="wordmark-text"><b>MLBench</b><small>实验工作台</small></span>
      </RouterLink>

      <button class="project-switcher">
        <span class="project-icon">01</span>
        <span class="project-copy"><small>当前项目</small><b>十大算法 Benchmark</b></span>
        <span class="switcher-caret">⌄</span>
      </button>

      <nav class="app-nav">
        <span class="nav-label">WORKSPACE</span>
        <RouterLink v-for="item in nav.slice(0, 3)" :key="item.to" :to="item.to" :title="item.label">
          <AppIcon :name="item.icon" :size="17" /><span>{{ item.label }}</span>
        </RouterLink>
        <span class="nav-label registry-label">REGISTRY</span>
        <RouterLink v-for="item in nav.slice(3)" :key="item.to" :to="item.to" :title="item.label">
          <AppIcon :name="item.icon" :size="17" /><span>{{ item.label }}</span>
        </RouterLink>
      </nav>

      <div class="sidebar-status" :class="`is-${backendStatus}`" :title="backendStatus === 'online' ? 'FastAPI 服务连接正常' : '请双击启动本地实验台.bat'">
        <span class="backend-light"></span>
        <span><b>{{ backendStatus === 'online' ? '后端已连接' : backendStatus === 'offline' ? '后端未启动' : '正在检测后端' }}</b><small>{{ backendStatus === 'online' ? 'FastAPI · WebSocket' : '双击启动本地实验台' }}</small></span>
      </div>
      <button class="collapse-button" @click="collapsed = !collapsed"><AppIcon :name="collapsed ? 'expand' : 'collapse'" :size="15" /></button>
    </aside>

    <section class="app-frame">
      <div v-if="isRunning" class="active-run-banner" role="status"><b>真实实验运行中 · {{ progress }}%</b><span>{{ lastMessage }}</span><RouterLink to="/experiments/new">查看运行状态 →</RouterLink></div>
      <header class="app-topbar">
        <div class="breadcrumb"><span>机器学习实践</span><i>/</i><b>{{ pageTitle }}</b></div>
        <div class="topbar-actions">
          <button class="command-search" @click="searchOpen = true"><AppIcon name="search" :size="15" /><span>搜索实验、算法、数据集</span><kbd>⌘ K</kbd></button>
          <span class="term-label">机器学习实践 · 课程项目</span>
          <div class="member-stack"><span>ZX</span><span>+3</span></div>
        </div>
      </header>
      <RouterView v-slot="{ Component }">
        <Transition name="page" mode="out-in"><component :is="Component" :key="route.fullPath" /></Transition>
      </RouterView>
    </section>

    <div v-if="searchOpen" class="command-backdrop" @click.self="searchOpen = false">
      <div class="command-dialog">
        <label><AppIcon name="search" :size="19" /><input v-model="searchQuery" autofocus placeholder="搜索实验、算法或数据集…" /><button @click="searchOpen = false">ESC</button></label>
        <div class="command-results">
          <span>快速跳转</span>
          <RouterLink v-for="item in searchResults" :key="item.to" :to="item.to" @click="searchOpen=false"><AppIcon name="search" :size="16"/><b>{{ item.title }}</b><small>{{ item.type }}</small></RouterLink>
          <p v-if="!searchResults.length" class="run-message">未找到相关记录。</p>
        </div>
      </div>
    </div>
  </div>
</template>
