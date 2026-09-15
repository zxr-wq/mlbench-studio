<script setup>
defineProps({ history: { type: Array, required: true } })
defineEmits(['refresh', 'open'])

const statusLabel = { completed: '已完成', running: '运行中', queued: '等待中', failed: '失败' }
function formatDate(value) {
  return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}
</script>

<template>
  <section class="content-view history-view">
    <div class="view-intro">
      <div><span class="section-number">EXPERIMENT LOG</span><h2>每一次尝试，都留下证据</h2></div>
      <button class="outline-button" @click="$emit('refresh')">刷新记录</button>
    </div>

    <div v-if="history.length" class="history-table paper-panel">
      <div class="history-head"><span>实验编号</span><span>数据集</span><span>模型组合</span><span>最佳主指标</span><span>状态</span><span>时间</span></div>
      <button v-for="item in history" :key="item.id" class="history-row" @click="$emit('open', item.id)">
        <span><b>#{{ item.id.slice(0, 6).toUpperCase() }}</b><small>seed {{ item.config?.seed }}</small></span>
        <span>{{ item.dataset?.name || item.dataset_id }}</span>
        <span>{{ item.model_ids.length }} 个模型</span>
        <span class="accuracy-cell">{{ (item.best_score ?? item.best_accuracy) == null ? '—' : `${((item.best_score ?? item.best_accuracy) * 100).toFixed(1)}%` }}</span>
        <span><i class="status-chip" :class="item.status">{{ statusLabel[item.status] }}</i></span>
        <span>{{ formatDate(item.created_at) }} <b class="open-arrow">↗</b></span>
      </button>
    </div>
    <div v-else class="empty-history paper-panel">
      <span>0</span><h3>还没有实验记录</h3><p>在实验工作台完成第一次模型比较后，记录会出现在这里。</p>
    </div>
  </section>
</template>
