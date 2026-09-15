<script setup>
import { computed } from 'vue'

const props = defineProps({ visualization: { type: Object, required: true } })
const palette = ['#7157d9', '#d06a50', '#2e947e', '#d6a038', '#5478bd', '#ad5e9f']
const histories = computed(() => props.visualization.centroid_history || [])
const latest = computed(() => histories.value.at(-1) || [])
const all = computed(() => histories.value.flat())
const scale = computed(() => {
  if (!all.value.length) return { x: () => 250, y: () => 130 }
  const xs = all.value.map((point) => Array.isArray(point) ? point[0] : point.x)
  const ys = all.value.map((point) => Array.isArray(point) ? point[1] : point.y)
  const minX = Math.min(...xs); const maxX = Math.max(...xs)
  const minY = Math.min(...ys); const maxY = Math.max(...ys)
  return {
    x: (point) => 28 + ((Array.isArray(point) ? point[0] : point.x) - minX) / Math.max(maxX - minX, 1e-9) * 444,
    y: (point) => 20 + (1 - ((Array.isArray(point) ? point[1] : point.y) - minY) / Math.max(maxY - minY, 1e-9)) * 215,
  }
})
function path(clusterIndex) {
  return histories.value.map((centroids, index) => `${index ? 'L' : 'M'} ${scale.value.x(centroids[clusterIndex])} ${scale.value.y(centroids[clusterIndex])}`).join(' ')
}
</script>

<template>
  <svg class="centroid-plot" viewBox="0 0 500 260" role="img" aria-label="K-Means 质心迭代路径">
    <rect x="0" y="0" width="500" height="260" fill="#f8f6f0" />
    <template v-for="(_, clusterIndex) in latest" :key="clusterIndex">
      <path :d="path(clusterIndex)" fill="none" :stroke="palette[clusterIndex % palette.length]" stroke-width="2" stroke-dasharray="4 3" />
      <circle v-for="(centroids, iteration) in histories" :key="iteration" :cx="scale.x(centroids[clusterIndex])" :cy="scale.y(centroids[clusterIndex])" :r="iteration === histories.length - 1 ? 8 : 3.5" :fill="palette[clusterIndex % palette.length]"><title>簇 {{ clusterIndex }} · 第 {{ iteration + 1 }} 轮</title></circle>
    </template>
  </svg>
</template>
