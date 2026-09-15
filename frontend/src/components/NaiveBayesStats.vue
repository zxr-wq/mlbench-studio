<script setup>
import { computed } from 'vue'

const props = defineProps({
  visualization: { type: Object, required: true },
})

const featureRows = computed(() => {
  const names = props.visualization.feature_names || []
  const means = props.visualization.means || []
  const variances = props.visualization.variances || []
  return names.slice(0, 8).map((name, featureIndex) => ({
    name,
    values: means.map((classMeans, classIndex) => ({
      mean: classMeans[featureIndex],
      variance: variances[classIndex][featureIndex],
    })),
  }))
})
</script>

<template>
  <div class="nb-stats">
    <div class="prior-list">
      <span v-for="(prior, index) in visualization.class_priors" :key="index">
        <small>{{ visualization.class_names[index] }}</small>
        <i><b :style="{ width: `${prior * 100}%` }"></b></i>
        <em>{{ (prior * 100).toFixed(1) }}%</em>
      </span>
    </div>
    <div class="nb-table">
      <div class="nb-row header" :style="{ gridTemplateColumns: `minmax(120px, 1.2fr) repeat(${visualization.class_names.length}, minmax(90px, 1fr))` }"><b>特征</b><span v-for="name in visualization.class_names" :key="name">{{ name }}</span></div>
      <div v-for="row in featureRows" :key="row.name" class="nb-row" :style="{ gridTemplateColumns: `minmax(120px, 1.2fr) repeat(${visualization.class_names.length}, minmax(90px, 1fr))` }">
        <b :title="row.name">{{ row.name }}</b>
        <span v-for="(value, index) in row.values" :key="index" :title="`variance ${value.variance.toFixed(4)}`">
          μ {{ value.mean.toFixed(2) }}<small>σ² {{ value.variance.toFixed(2) }}</small>
        </span>
      </div>
    </div>
  </div>
</template>
