<script setup lang="ts">
import ResultExplorer from './ResultExplorer.vue'
defineProps<{ visualization?: Record<string, any>, taskType?: string }>()
</script>

<template>
  <ResultExplorer :visualization="visualization" :task-type="taskType" />
</template>
