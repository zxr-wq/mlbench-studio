<script setup lang="ts">
import { computed } from 'vue'
import { algorithmEducation, metricEducation, modelKey } from '../data/algorithmEducation'

const props = defineProps<{ model: string, taskType?: string, metrics?: Record<string, number | number[]> }>()
const key = computed(() => modelKey(props.model))
const education = computed(() => algorithmEducation[key.value])
const title = computed(() => ({ classification: '分类结果怎么读？', regression: '回归结果怎么读？', clustering: '聚类结果怎么读？', dimensionality_reduction: 'PCA 结果怎么读？' }[props.taskType ?? 'classification'] ?? '结果怎么读？'))
const metricTips = computed(() => metricEducation[props.taskType as keyof typeof metricEducation] ?? metricEducation.classification)
</script>

<template>
  <section class="algorithm-guide">
    <header><span>LEARN BY THIS RUN</span><h2>小白也能读懂的实验结论</h2><p>先看图，再用下面的原则判断“这次到底跑得好不好”。</p></header>
    <div class="guide-columns">
      <article><b>① {{ props.model }} 的原理</b><p>{{ education?.principle ?? '该模型使用统一训练与评价流程。' }}</p><small>适合什么情况</small><p>{{ education?.useWhen }}</p></article>
      <article><b>② {{ title }}</b><div v-for="tip in metricTips" :key="tip[0]" class="metric-tip"><strong>{{ tip[0] }}</strong><p>{{ tip[1] }}</p></div></article>
      <article><b>③ 本算法可以看哪些图</b><ul><li v-for="chart in education?.visuals" :key="chart">{{ chart }}</li></ul><p class="guide-note">图表来自本次实验数据；没有统计意义的图不会强行生成。</p></article>
    </div>
  </section>
</template>
