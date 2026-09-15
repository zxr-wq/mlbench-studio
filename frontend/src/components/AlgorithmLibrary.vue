<script setup>
import { ref } from 'vue'

defineProps({ algorithms: { type: Array, required: true } })
const expanded = ref('logistic_regression')
</script>

<template>
  <section class="content-view algorithm-view">
    <div class="view-intro">
      <div><span class="section-number">MODEL ATLAS</span><h2>从公式，到可以运行的代码</h2></div>
      <p>算法目录由后端 Registry 动态生成。分类、回归、聚类与降维模型共用统一的参数和可视化约定。</p>
    </div>
    <div class="algorithm-atlas">
      <article
        v-for="(algorithm, index) in algorithms"
        :key="algorithm.id"
        class="atlas-card paper-panel"
        :class="{ expanded: expanded === algorithm.id }"
        @click="expanded = expanded === algorithm.id ? '' : algorithm.id"
      >
        <div class="atlas-index">{{ String(index + 1).padStart(2, '0') }}</div>
        <div class="atlas-summary">
          <span>{{ algorithm.family }}</span>
          <h3>{{ algorithm.name }}</h3>
          <p>{{ algorithm.description }}</p>
          <div class="tag-list"><i v-for="tag in algorithm.strengths" :key="tag">{{ tag }}</i></div>
        </div>
        <button class="expand-button" :aria-label="`展开${algorithm.name}`">{{ expanded === algorithm.id ? '−' : '+' }}</button>
        <div v-if="expanded === algorithm.id" class="atlas-detail">
          <div><span>核心原理</span><p>{{ algorithm.principle }}</p></div>
          <div><span>关键公式</span><code>{{ algorithm.formula }}</code></div>
          <div><span>默认参数</span><p class="parameter-copy"><b v-for="(value, key) in algorithm.defaults" :key="key">{{ key }} = {{ value }}</b></p></div>
        </div>
      </article>
    </div>
  </section>
</template>
