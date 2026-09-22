<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute } from 'vue-router'
import AppIcon from './components/AppIcon.vue'

const route = useRoute()
const collapsed = ref(false)
const searchOpen = ref(false)
const searchQuery = ref('')

const nav = [
  { label: '概览', to: '/', icon: 'overview' },
  { label: '创建实验', to: '/experiments/new', icon: 'flask' },
  { label: '运行记录', to: '/runs', icon: 'runs' },
  { label: '算法库', to: '/algorithms', icon: 'algorithm' },
  { label: '数据集', to: '/datasets', icon: 'dataset' },
]

const pageTitle = computed(() => String(route.meta.title || 'MLBench'))
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

      <div class="sidebar-status">
        <span class="backend-light"></span>
        <span><b>Mock runtime</b><small>后端接口已预留</small></span>
      </div>
      <button class="collapse-button" @click="collapsed = !collapsed"><AppIcon :name="collapsed ? 'expand' : 'collapse'" :size="15" /></button>
    </aside>

    <section class="app-frame">
      <header class="app-topbar">
        <div class="breadcrumb"><span>机器学习实践</span><i>/</i><b>{{ pageTitle }}</b></div>
        <div class="topbar-actions">
          <button class="command-search" @click="searchOpen = true"><AppIcon name="search" :size="15" /><span>搜索实验、算法、数据集</span><kbd>⌘ K</kbd></button>
          <span class="term-label">2026 夏季学期</span>
          <div class="member-stack"><span>ZX</span><span>+3</span></div>
        </div>
      </header>
      <RouterView v-slot="{ Component }">
        <Transition name="page" mode="out-in"><component :is="Component" /></Transition>
      </RouterView>
    </section>

    <div v-if="searchOpen" class="command-backdrop" @click.self="searchOpen = false">
      <div class="command-dialog">
        <label><AppIcon name="search" :size="19" /><input v-model="searchQuery" autofocus placeholder="搜索实验、算法或数据集…" /><button @click="searchOpen = false">ESC</button></label>
        <div class="command-results">
          <span>快速跳转</span>
          <RouterLink to="/experiments/new" @click="searchOpen = false"><AppIcon name="flask" :size="16" /><b>创建新实验</b><small>Builder</small></RouterLink>
          <RouterLink to="/runs" @click="searchOpen = false"><AppIcon name="runs" :size="16" /><b>查看运行记录</b><small>Experiments</small></RouterLink>
          <RouterLink to="/algorithms" @click="searchOpen = false"><AppIcon name="algorithm" :size="16" /><b>打开算法库</b><small>Registry</small></RouterLink>
        </div>
      </div>
    </div>
  </div>
</template>
