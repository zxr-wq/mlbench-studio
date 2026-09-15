<script setup>
import { computed } from 'vue'

defineOptions({ name: 'TreeNode' })
const props = defineProps({ node: { type: Object, required: true }, depth: { type: Number, default: 0 } })
const isLeaf = computed(() => !props.node.left && !props.node.right)
const valueLabel = computed(() => Array.isArray(props.node.value) ? props.node.value.join(' / ') : String(props.node.value ?? '—'))
</script>

<template>
  <div class="tree-branch">
    <div class="tree-node" :class="{ leaf: isLeaf }">
      <strong>{{ isLeaf ? '叶节点' : (node.feature ?? '特征') }}</strong>
      <span v-if="!isLeaf">≤ {{ Number(node.threshold).toFixed(3) }}</span>
      <small>samples {{ node.samples ?? '—' }}</small>
      <em>value [{{ valueLabel }}]</em>
    </div>
    <div v-if="!isLeaf" class="tree-children">
      <div v-if="node.left"><b>True</b><TreeNode :node="node.left" :depth="depth + 1" /></div>
      <div v-if="node.right"><b>False</b><TreeNode :node="node.right" :depth="depth + 1" /></div>
    </div>
  </div>
</template>
