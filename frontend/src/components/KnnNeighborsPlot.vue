<script setup>
import { computed } from 'vue'

const props = defineProps({
  visualization: { type: Object, required: true },
})

const palette = ['#7157d9', '#d06a50', '#2e947e', '#d6a038', '#5478bd']
const plotted = computed(() => {
  const points = [props.visualization.query, ...(props.visualization.neighbors || [])]
  if (!points.length) return { query: null, neighbors: [] }
  const xs = points.map((point) => point.x)
  const ys = points.map((point) => point.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const width = Math.max(maxX - minX, 1e-8)
  const height = Math.max(maxY - minY, 1e-8)
  const project = (point) => ({
    ...point,
    cx: 34 + ((point.x - minX) / width) * 432,
    cy: 25 + (1 - (point.y - minY) / height) * 205,
  })
  return { query: project(points[0]), neighbors: points.slice(1).map(project) }
})
</script>

<template>
  <div class="knn-neighbor-wrap">
    <svg viewBox="0 0 500 260" role="img" aria-label="KNN 最近邻二维投影">
      <line
        v-for="neighbor in plotted.neighbors"
        :key="`line-${neighbor.rank}`"
        :x1="plotted.query.cx"
        :y1="plotted.query.cy"
        :x2="neighbor.cx"
        :y2="neighbor.cy"
        stroke="#aaa0b7"
        stroke-width="1.2"
        stroke-dasharray="4 3"
      />
      <circle
        v-for="neighbor in plotted.neighbors"
        :key="neighbor.rank"
        :cx="neighbor.cx"
        :cy="neighbor.cy"
        r="7"
        :fill="palette[neighbor.label % palette.length]"
        stroke="#fff"
        stroke-width="2"
      >
        <title>第 {{ neighbor.rank }} 邻居 · {{ visualization.class_names[neighbor.label] }} · 距离 {{ neighbor.distance.toFixed(4) }}</title>
      </circle>
      <circle v-if="plotted.query" :cx="plotted.query.cx" :cy="plotted.query.cy" r="10" fill="#201a2b" stroke="#fff" stroke-width="3" />
      <text v-if="plotted.query" :x="plotted.query.cx" :y="plotted.query.cy + 3" text-anchor="middle" fill="#fff" font-size="8">Q</text>
    </svg>
    <div class="neighbor-list">
      <span v-for="neighbor in visualization.neighbors" :key="neighbor.rank">
        #{{ neighbor.rank }} {{ visualization.class_names[neighbor.label] }} <b>{{ neighbor.distance.toFixed(3) }}</b>
      </span>
    </div>
  </div>
</template>
