<script setup>
import { computed } from 'vue'

const props = defineProps({ visualization: { type: Object, required: true } })
const palette = ['#7157d9', '#d06a50', '#2e947e', '#d6a038', '#5478bd']
const allPoints = computed(() => [...(props.visualization.points || []), ...(props.visualization.support_vectors || [])])
const scale = computed(() => {
  if (!allPoints.value.length) return { x: () => 0, y: () => 0 }
  const xs = allPoints.value.map((point) => point.x)
  const ys = allPoints.value.map((point) => point.y)
  const minX = Math.min(...xs); const maxX = Math.max(...xs)
  const minY = Math.min(...ys); const maxY = Math.max(...ys)
  return {
    x: (value) => 24 + (value - minX) / Math.max(maxX - minX, 1e-9) * 452,
    y: (value) => 18 + (1 - (value - minY) / Math.max(maxY - minY, 1e-9)) * 224,
  }
})
</script>

<template>
  <svg class="boundary-plot" viewBox="0 0 500 260" role="img" aria-label="SVM 决策边界与支持向量">
    <rect x="0" y="0" width="500" height="260" fill="#f8f6f0" />
    <circle v-for="(point, index) in visualization.points || []" :key="`p-${index}`" :cx="scale.x(point.x)" :cy="scale.y(point.y)" r="4.5" :fill="palette[point.label % palette.length]" opacity=".78" />
    <circle v-for="(point, index) in visualization.support_vectors || []" :key="`s-${index}`" :cx="scale.x(point.x)" :cy="scale.y(point.y)" r="8" fill="none" stroke="#1d1824" stroke-width="2.3"><title>支持向量</title></circle>
    <path v-if="visualization.boundary_path" :d="visualization.boundary_path" fill="none" stroke="#1d1824" stroke-width="2" />
  </svg>
</template>
