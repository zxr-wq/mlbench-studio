<script setup>
import { computed } from 'vue'

const props = defineProps({ points: { type: Array, default: () => [] } })

const line = computed(() => {
  if (!props.points.length) return ''
  const losses = props.points.map((item) => item.loss)
  const min = Math.min(...losses)
  const max = Math.max(...losses)
  const span = Math.max(max - min, 1e-9)
  return props.points.map((point, index) => {
    const x = 12 + (index / Math.max(1, props.points.length - 1)) * 376
    const y = 16 + (1 - (point.loss - min) / span) * 128
    return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
  }).join(' ')
})
</script>

<template>
  <div class="curve-wrap">
    <svg viewBox="0 0 400 164" role="img" aria-label="训练损失曲线">
      <line v-for="n in 4" :key="n" x1="12" x2="388" :y1="n * 32" :y2="n * 32" stroke="#ded8ce" stroke-width="1" />
      <path :d="line" fill="none" stroke="#7157d9" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
      <circle v-if="points.length" cx="388" :cy="16 + 128 * 0" r="0" />
    </svg>
    <div class="curve-labels"><span>开始</span><span>{{ points.at(-1)?.epoch || 0 }} epochs</span></div>
  </div>
</template>
