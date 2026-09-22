<script setup lang="ts">
import AppIcon from '../components/AppIcon.vue'
import { datasets } from '../data/catalog'
</script>

<template>
  <div class="page catalog-page">
    <header class="page-header"><span class="overline">REGISTRY / DATASETS</span><h1>数据集目录</h1><p>少而经典的数据集，用一致的数据契约服务全部算法。</p></header>
    <section class="dataset-list">
      <article v-for="(item, index) in datasets" :key="item.id" class="dataset-row">
        <span class="dataset-index">0{{ index + 1 }}</span>
        <div class="dataset-swatch" :style="{ background: item.accent }"><span v-for="dot in 12" :key="dot"></span></div>
        <div class="dataset-copy"><span>{{ item.task }}</span><h2>{{ item.name }}</h2><p>{{ item.description }}</p></div>
        <dl><div><dt>样本</dt><dd>{{ item.samples.toLocaleString() }}</dd></div><div><dt>特征</dt><dd>{{ item.features }}</dd></div><div><dt>类别</dt><dd>{{ item.classes }}</dd></div></dl>
        <RouterLink :to="`/experiments/new?dataset=${item.id}`" class="round-link"><AppIcon name="arrow" :size="17" /></RouterLink>
      </article>
    </section>
  </div>
</template>
