<script setup>
import { computed } from 'vue'

const props = defineProps({
  matrix: { type: Array, default: () => [] },
  classNames: { type: Array, default: () => [] },
})

const maxValue = computed(() => Math.max(1, ...props.matrix.flat()))
function cellStyle(value) {
  const intensity = 0.1 + (value / maxValue.value) * 0.82
  return { backgroundColor: `rgba(113, 87, 217, ${intensity})`, color: intensity > 0.52 ? '#ffffff' : '#312d38' }
}
</script>

<template>
  <div class="matrix-shell" :style="{ '--matrix-size': matrix.length }">
    <div class="matrix-axis predicted">预测标签 →</div>
    <div class="matrix-axis actual">真实标签 ↓</div>
    <div class="matrix-grid">
      <div class="matrix-corner"></div>
      <div v-for="(name, index) in classNames" :key="`head-${name}`" class="matrix-label top" :title="name">C{{ index }}</div>
      <template v-for="(row, rowIndex) in matrix" :key="rowIndex">
        <div class="matrix-label side" :title="classNames[rowIndex]">C{{ rowIndex }}</div>
        <div
          v-for="(value, colIndex) in row"
          :key="`${rowIndex}-${colIndex}`"
          class="matrix-cell"
          :class="{ diagonal: rowIndex === colIndex }"
          :style="cellStyle(value)"
          :title="`${classNames[rowIndex]} → ${classNames[colIndex]}: ${value}`"
        >{{ value }}</div>
      </template>
    </div>
    <div class="matrix-key">
      <span v-for="(name, index) in classNames" :key="name">C{{ index }} {{ name }}</span>
    </div>
  </div>
</template>
