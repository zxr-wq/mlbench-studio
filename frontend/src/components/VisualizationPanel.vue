<script setup>
import { computed } from 'vue'
import CentroidHistoryPlot from './CentroidHistoryPlot.vue'
import DecisionBoundaryPlot from './DecisionBoundaryPlot.vue'
import KnnNeighborsPlot from './KnnNeighborsPlot.vue'
import NaiveBayesStats from './NaiveBayesStats.vue'
import PcaVisualization from './PcaVisualization.vue'
import TrainingCurve from './TrainingCurve.vue'
import TreeVisualization from './TreeVisualization.vue'

const props = defineProps({ visualization: { type: Object, default: () => ({}) } })
const type = computed(() => {
  if (props.visualization.type) return props.visualization.type
  if (props.visualization.tree) return 'decision_tree'
  if (props.visualization.centroid_history) return 'kmeans_clusters'
  if (props.visualization.explained_variance_ratio) return 'pca_projection'
  if (props.visualization.loss_history) return 'loss_history'
  return ''
})
const lossPoints = computed(() => (props.visualization.loss_history || []).map((loss, index) => ({ epoch: index + 1, loss })))
</script>

<template>
  <div v-if="type" class="unified-visualization">
    <KnnNeighborsPlot v-if="type === 'knn_neighbors'" :visualization="visualization" />
    <NaiveBayesStats v-else-if="type === 'naive_bayes_stats'" :visualization="visualization" />
    <TreeVisualization v-else-if="type === 'decision_tree' || type === 'tree'" :tree="visualization.tree || visualization" />
    <DecisionBoundaryPlot v-else-if="type === 'svm_boundary' || type === 'decision_boundary'" :visualization="visualization" />
    <CentroidHistoryPlot v-else-if="type === 'kmeans_clusters' || type === 'centroid_history'" :visualization="visualization" />
    <PcaVisualization v-else-if="type === 'pca_projection' || type === 'pca'" :visualization="visualization" />
    <TrainingCurve v-else-if="type === 'loss_history' || type === 'gradient_boosting_loss'" :points="lossPoints" />
  </div>
</template>
