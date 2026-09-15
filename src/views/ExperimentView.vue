<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import { algorithms, datasets } from '../data/catalog'
import { useLabStore } from '../stores/lab'
import type { ExperimentConfig } from '../types/benchmark'

const route = useRoute()
const router = useRouter()
const { defaultConfig, isRunning, progress, lastMessage, runExperiment } = useLabStore()
const step = ref(1)
const config = reactive<ExperimentConfig>({ ...defaultConfig, metrics: [...defaultConfig.metrics] })

const implementedModels = new Set(['svm', 'kmeans'])
const availableModels = algorithms
const preprocessingLabels: Record<string, string> = {
  standard_scaler: 'Standard Scaler',
  minmax_scaler: 'Min-Max Scaler',
  pca_0.95: 'PCA · 95% 方差',
  none: '不处理',
}

function toggleMetric(metric: string) {
  if (config.metrics.includes(metric)) {
    if (config.metrics.length > 1) config.metrics = config.metrics.filter((item) => item !== metric)
  } else config.metrics = [...config.metrics, metric]
}

async function submit() {
  const result = await runExperiment({ ...config })
  router.push(`/runs/${result.id}`)
}

onMounted(() => {
  if (typeof route.query.dataset === 'string' && datasets.some((item) => item.id === route.query.dataset)) {
    config.dataset = route.query.dataset
  }
  if (typeof route.query.model === 'string' && implementedModels.has(route.query.model)) {
    config.model = route.query.model
    step.value = 2
  }
})
</script>

<template>
  <div class="builder-layout">
    <main class="builder-main">
      <header class="page-header">
        <span class="overline">EXPERIMENT BUILDER</span>
        <h1>创建一次可复现实验</h1>
        <p>配置会随实验结果保存，之后可完整回放。</p>
      </header>

      <ol class="step-nav">
        <li v-for="item in [{ n: 1, t: '选择数据' }, { n: 2, t: '选择算法' }, { n: 3, t: '评价设置' }]" :key="item.n" :class="{ active: step === item.n, done: step > item.n }" @click="step = item.n">
          <span>{{ step > item.n ? '✓' : item.n }}</span><b>{{ item.t }}</b>
        </li>
      </ol>

      <section v-if="step === 1" class="builder-stage">
        <div class="stage-heading"><span>01</span><div><h2>选择数据集</h2><p>先决定这次实验要回答什么问题。</p></div></div>
        <div class="choice-grid dataset-choice-grid">
          <button v-for="item in datasets" :key="item.id" :class="{ selected: config.dataset === item.id }" @click="config.dataset = item.id">
            <i :style="{ background: item.accent }"></i>
            <span><b>{{ item.name }}</b><small>{{ item.task }} · {{ item.samples }} 样本 · {{ item.features }} 特征</small></span>
            <em><AppIcon name="check" :size="13" /></em>
          </button>
        </div>
        <div class="inline-fields">
          <label><span>划分策略</span><select v-model="config.split"><option value="stratified_80_20">分层随机划分 · 80 / 20</option><option value="kfold_5">五折交叉验证</option></select></label>
          <label><span>预处理</span><select v-model="config.preprocessing"><option value="standard_scaler">Standard Scaler</option><option value="minmax_scaler">Min-Max Scaler</option><option value="pca_0.95">PCA 降维 · 保留 95% 方差</option><option value="none">不处理</option></select></label>
        </div>
      </section>

      <section v-else-if="step === 2" class="builder-stage">
        <div class="stage-heading"><span>02</span><div><h2>选择算法</h2><p>同一接口下可以随时替换实现。</p></div></div>
        <div class="choice-grid model-choice-grid">
          <button
            v-for="item in availableModels"
            :key="item.id"
            :class="{ selected: config.model === item.id, 'not-ready': !implementedModels.has(item.id) }"
            :disabled="!implementedModels.has(item.id)"
            :title="implementedModels.has(item.id) ? '' : '该算法尚未接入本地运行时'"
            @click="config.model = item.id"
          >
            <span class="model-monogram" :style="{ color: item.accent, borderColor: `${item.accent}55` }">{{ item.shortName }}</span>
            <span><b>{{ item.name }}</b><small>{{ implementedModels.has(item.id) ? item.implementation : '待接入' }}</small></span>
            <em><AppIcon name="check" :size="13" /></em>
          </button>
        </div>
        <div v-if="config.model === 'svm'" class="parameter-panel">
          <div><span>模型参数</span><small>来自 SVM 参数 Schema</small></div>
          <label><span>Kernel</span><select v-model="config.kernel"><option value="rbf">RBF</option><option value="linear">Linear</option><option value="poly">Polynomial</option></select></label>
          <label class="range-control"><span>正则化系数 C <output>{{ config.c.toFixed(1) }}</output></span><input v-model.number="config.c" type="range" min="0.1" max="5" step="0.1" /></label>
        </div>
        <div v-else-if="config.model === 'kmeans'" class="parameter-panel">
          <div><span>模型参数</span><small>来自 K-Means 参数 Schema</small></div>
          <p class="parameter-note">k 自动等于类别数，采用 k-means++ 初始化并做 8 次重启取最小簇内平方误差。</p>
        </div>
      </section>

      <section v-else class="builder-stage">
        <div class="stage-heading"><span>03</span><div><h2>设置评价方式</h2><p>选择能够回答实验问题的指标。</p></div></div>
        <div class="metric-choice-list">
          <button v-for="item in [{ id: 'accuracy', name: 'Accuracy', desc: '整体预测正确的样本比例' }, { id: 'macro_f1', name: 'Macro F1', desc: '平等看待每一个类别' }, { id: 'precision', name: 'Precision', desc: '关注阳性预测的可信程度' }]" :key="item.id" :class="{ selected: config.metrics.includes(item.id) }" @click="toggleMetric(item.id)">
            <em><AppIcon name="check" :size="14" /></em><span><b>{{ item.name }}</b><small>{{ item.desc }}</small></span>
          </button>
        </div>
        <label class="seed-field"><span>随机种子</span><input v-model.number="config.seed" type="number" /><small>固定随机种子，保证数据划分和模型结果可以复现。</small></label>
      </section>

      <footer class="builder-actions">
        <button class="button ghost" :disabled="step === 1" @click="step -= 1">上一步</button>
        <button v-if="step < 3" class="button primary" @click="step += 1">继续 <AppIcon name="arrow" :size="15" /></button>
        <button v-else class="button primary" :disabled="isRunning" @click="submit"><AppIcon name="flask" :size="15" />{{ isRunning ? '实验运行中' : '运行实验' }}</button>
      </footer>
    </main>

    <aside class="builder-summary">
      <span class="overline">RUN PREVIEW</span>
      <h2>实验摘要</h2>
      <div class="summary-pipeline">
        <div><i>1</i><span><small>DATASET</small><b>{{ datasets.find((item) => item.id === config.dataset)?.name }}</b></span></div>
        <div><i>2</i><span><small>PREPROCESS</small><b>{{ preprocessingLabels[config.preprocessing] ?? config.preprocessing }}</b></span></div>
        <div><i>3</i><span><small>MODEL</small><b>{{ algorithms.find((item) => item.id === config.model)?.name }}</b></span></div>
        <div><i>4</i><span><small>EVALUATION</small><b>{{ config.metrics.join(' + ') }}</b></span></div>
      </div>
      <div class="repro-note"><AppIcon name="check" :size="16" /><span><b>可复现实验</b><small>配置、环境版本和随机种子将一起保存。</small></span></div>
      <div v-if="isRunning" class="running-panel"><span>{{ lastMessage }}</span><b>{{ progress }}%</b><div><i :style="{ width: `${progress}%` }"></i></div></div>
    </aside>
  </div>
</template>
