<script setup>
import { computed } from 'vue'

const props = defineProps({
  points: { type: Array, default: () => [] },
  classNames: { type: Array, default: () => [] },
})

const palette = ['#7157d9', '#d06a50', '#2e947e', '#d6a038', '#5478bd', '#ad5e9f', '#4d9dbe', '#8d7c58', '#a5a84c', '#69707f']

const plotted = computed(() => {
  if (!props.points.length) return []
  const xs = props.points.map((point) => point.x)
  const ys = props.points.map((point) => point.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const width = Math.max(maxX - minX, 1e-8)
  const height = Math.max(maxY - minY, 1e-8)
  return props.points.map((point) => ({
    ...point,
    cx: 22 + ((point.x - minX) / width) * 456,
    cy: 18 + (1 - (point.y - minY) / height) * 224,
  }))
})
</script>

<template>
  <div class="scatter-wrap">
    <svg viewBox="0 0 500 260" role="img" aria-label="测试样本 PCA 二维投影">
      <defs>
        <pattern id="grid" width="50" height="52" patternUnits="userSpaceOnUse">
          <path d="M 50 0 L 0 0 0 52" fill="none" stroke="#d8d2c7" stroke-width="0.6" />
        </pattern>
      </defs>
      <rect x="0" y="0" width="500" height="260" fill="url(#grid)" />
      <circle
        v-for="(point, index) in plotted"
        :key="index"
        :cx="point.cx"
        :cy="point.cy"
        :r="point.correct ? 4.2 : 6.2"
        :fill="palette[point.label % palette.length]"
        :stroke="point.correct ? '#fffdf7' : '#1c1c20'"
        :stroke-width="point.correct ? 1 : 2.3"
        :opacity="point.correct ? 0.78 : 1"
      >
        <title>{{ classNames[point.label] }} → {{ classNames[point.prediction] }}</title>
      </circle>
    </svg>
    <div class="plot-legend">
      <span v-for="(name, index) in classNames" :key="name"><i :style="{ background: palette[index % palette.length] }"></i>{{ name }}</span>
      <span class="error-legend"><i></i>误判（黑色描边）</span>
    </div>
  </div>
</template>
