<script setup>
import ScatterPlot from './ScatterPlot.vue'
defineProps({ visualization: { type: Object, required: true } })
</script>

<template>
  <div class="pca-viz">
    <ScatterPlot :points="visualization.projection || []" :class-names="visualization.class_names || []" />
    <div class="variance-list">
      <span v-for="(value, index) in visualization.explained_variance_ratio || []" :key="index">
        <small>PC{{ index + 1 }}</small><i><b :style="{ width: `${value * 100}%` }"></b></i><em>{{ (value * 100).toFixed(1) }}%</em>
      </span>
    </div>
  </div>
</template>
