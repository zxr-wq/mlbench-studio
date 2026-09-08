<script setup lang="ts">
import { computed, ref } from 'vue'
import AppIcon from '../components/AppIcon.vue'
import { algorithms, type AlgorithmItem } from '../data/catalog'

const query = ref('')
const family = ref('全部')
const selected = ref<AlgorithmItem | null>(null)
const families = ['全部', '分类', '聚类', '降维', '集成', '神经网络']
const filtered = computed(() => algorithms.filter((item) => {
  const matchesFamily = family.value === '全部' || item.family === family.value
  const matchesQuery = `${item.name} ${item.shortName}`.toLowerCase().includes(query.value.toLowerCase())
  return matchesFamily && matchesQuery
}))
</script>

<template>
  <div class="page catalog-page">
    <header class="page-header split-header">
      <div><span class="overline">REGISTRY / ALGORITHMS</span><h1>算法注册表</h1><p>算法不是散落的脚本，而是遵循统一协议的可替换模块。</p></div>
      <RouterLink to="/experiments/new" class="button primary"><AppIcon name="flask" :size="15" />使用算法</RouterLink>
    </header>
    <div class="catalog-toolbar">
      <label class="search-box"><AppIcon name="search" :size="16" /><input v-model="query" placeholder="搜索算法" /></label>
      <div class="filter-tabs"><button v-for="item in families" :key="item" :class="{ active: family === item }" @click="family = item">{{ item }}</button></div>
    </div>
    <section class="algorithm-grid">
      <button v-for="item in filtered" :key="item.id" class="algorithm-card" @click="selected = item">
        <div class="algorithm-card-top"><span class="algorithm-glyph" :style="{ color: item.accent, borderColor: `${item.accent}55`, background: `${item.accent}0c` }">{{ item.shortName }}</span><span class="readiness" :class="item.status"><i></i>{{ item.status === 'ready' ? '可运行' : '开发中' }}</span></div>
        <span class="family-label">{{ item.family }}</span><h2>{{ item.name }}</h2><p>{{ item.description }}</p>
        <div class="algorithm-card-foot"><span>{{ item.implementation }}</span><AppIcon name="arrow" :size="15" /></div>
      </button>
    </section>

    <div v-if="selected" class="drawer-backdrop" @click.self="selected = null">
      <aside class="detail-drawer">
        <button class="drawer-close" @click="selected = null">×</button>
        <span class="algorithm-glyph large" :style="{ color: selected.accent, borderColor: `${selected.accent}55`, background: `${selected.accent}0c` }">{{ selected.shortName }}</span>
        <span class="overline">{{ selected.family }} / {{ selected.implementation }}</span><h1>{{ selected.name }}</h1><p>{{ selected.description }}</p>
        <div class="tag-row"><span v-for="tag in selected.tags" :key="tag">{{ tag }}</span></div>
        <dl><div><dt>接口</dt><dd>BaseModel</dd></div><div><dt>状态</dt><dd>{{ selected.status === 'ready' ? '通过接口测试' : '正在开发' }}</dd></div><div><dt>版本</dt><dd>v0.1.0</dd></div></dl>
        <RouterLink :to="`/experiments/new?model=${selected.id}`" class="button primary full-button">用这个算法创建实验 <AppIcon name="arrow" :size="15" /></RouterLink>
      </aside>
    </div>
  </div>
</template>

