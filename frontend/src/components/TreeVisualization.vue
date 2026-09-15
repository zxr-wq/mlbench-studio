<script setup>
import TreeNode from './TreeNode.vue'
defineProps({ tree: { type: Object, required: true } })
</script>

<template>
  <div class="tree-visualization"><TreeNode :node="tree" /></div>
</template>
