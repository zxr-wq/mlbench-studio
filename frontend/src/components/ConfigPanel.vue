<script setup>
import { computed, ref, watch } from 'vue'

const props = defineProps({
  datasets: { type: Array, required: true },
  algorithms: { type: Array, required: true },
  running: { type: Boolean, default: false },
})
const emit = defineEmits(['run'])
const model = defineModel({ type: Object, required: true })
const advancedOpen = ref(false)

const selectedDataset = computed(() => props.datasets.find((item) => item.id === model.value.dataset_id))
const selectedAlgorithms = computed(() => props.algorithms.filter((item) => model.value.model_ids.includes(item.id)))
const referenceCount = computed(() => selectedAlgorithms.value.filter((item) => item.implementations?.includes('sklearn')).length)
const metricCatalog = {
  classification: [{ id: 'accuracy', label: 'Accuracy' }, { id: 'precision', label: 'Precision' }, { id: 'recall', label: 'Recall' }, { id: 'f1', label: 'F1' }],
  regression: [{ id: 'mse', label: 'MSE' }, { id: 'rmse', label: 'RMSE' }, { id: 'mae', label: 'MAE' }, { id: 'r2', label: 'R²' }],
  clustering: [{ id: 'silhouette_score', label: 'Silhouette' }, { id: 'inertia', label: 'Inertia' }],
  dimensionality_reduction: [{ id: 'cumulative_explained_variance', label: '累计解释方差' }],
}
const taskType = computed(() => selectedDataset.value?.task_type || 'classification')
const metricOptions = computed(() => metricCatalog[taskType.value] || metricCatalog.classification)

function isCompatible(algorithm) {
  return algorithm.available !== false && (!algorithm.task_type || algorithm.task_type === taskType.value)
}

function toggleAlgorithm(id) {
  const algorithm = props.algorithms.find((item) => item.id === id)
  if (algorithm && !isCompatible(algorithm)) return
  const values = [...model.value.model_ids]
  if (values.includes(id)) {
    if (values.length === 1) return
    model.value.model_ids = values.filter((item) => item !== id)
  } else if (values.length < 5) {
    model.value.model_ids = [...values, id]
  }
}

watch(taskType, () => {
  const compatible = props.algorithms.filter(isCompatible)
  const retained = model.value.model_ids.filter((id) => compatible.some((item) => item.id === id))
  model.value.model_ids = retained.length ? retained : compatible.slice(0, 1).map((item) => item.id)
  model.value.metrics = metricOptions.value.map((item) => item.id)
})

function toggleImplementation(implementation) {
  const values = [...model.value.implementations]
  if (values.includes(implementation)) {
    if (values.length === 1) return
    model.value.implementations = values.filter((item) => item !== implementation)
  } else {
    model.value.implementations = [...values, implementation]
  }
}

function toggleMetric(metric) {
  const values = [...model.value.metrics]
  if (values.includes(metric)) {
    if (values.length === 1) return
    model.value.metrics = values.filter((item) => item !== metric)
  } else {
    model.value.metrics = [...values, metric]
  }
}

function resetParameters() {
  for (const algorithm of props.algorithms) {
    model.value.parameters[algorithm.id] = { ...algorithm.defaults }
  }
  model.value.seed = 42
  model.value.test_size = 0.2
  model.value.folds = 5
}
</script>

<template>
  <aside class="config-panel paper-panel">
    <div class="panel-heading">
      <div>
        <span class="section-number">SETUP / 01</span>
        <h2>配置一次实验</h2>
      </div>
      <span class="selection-count">{{ model.model_ids.length }} 个模型</span>
    </div>

    <div class="config-section">
      <label class="field-label">01 · 选择数据集</label>
      <div class="dataset-list">
        <button
          v-for="dataset in datasets"
          :key="dataset.id"
          class="dataset-option"
          :class="{ selected: model.dataset_id === dataset.id }"
          @click="model.dataset_id = dataset.id"
        >
          <span class="dataset-swatch" :style="{ background: dataset.accent }"></span>
          <span class="dataset-copy">
            <strong>{{ dataset.name }}</strong>
            <small>{{ dataset.samples }} 样本 · {{ dataset.features }} 特征 · {{ dataset.classes }} 类</small>
          </span>
          <span class="radio-mark"></span>
        </button>
      </div>
      <p v-if="selectedDataset" class="field-hint">{{ selectedDataset.description }}</p>
    </div>

    <div class="config-section">
      <div class="field-row">
        <label class="field-label">02 · 数据划分</label>
        <span class="field-value">{{ model.split_method === 'kfold' ? `${model.folds} 折` : `${Math.round(model.test_size * 100)}% 测试集` }}</span>
      </div>
      <div class="segmented-control three">
        <button :class="{ active: model.split_method === 'stratified' }" @click="model.split_method = 'stratified'">分层抽样</button>
        <button :class="{ active: model.split_method === 'random' }" @click="model.split_method = 'random'">随机划分</button>
        <button :class="{ active: model.split_method === 'kfold' }" @click="model.split_method = 'kfold'">交叉验证</button>
      </div>
      <div v-if="model.split_method === 'kfold'" class="range-row">
        <input v-model.number="model.folds" type="range" min="3" max="10" step="1" />
        <output>{{ model.folds }} folds</output>
      </div>
      <div v-else class="range-row">
        <input v-model.number="model.test_size" type="range" min="0.1" max="0.5" step="0.05" />
        <output>{{ Math.round(model.test_size * 100) }}%</output>
      </div>
    </div>

    <div class="config-section">
      <div class="field-row">
        <label class="field-label">03 · 评价指标</label>
        <span class="field-value">宏平均</span>
      </div>
      <div class="metric-picker" :style="{ gridTemplateColumns: `repeat(${metricOptions.length}, 1fr)` }">
        <button
          v-for="item in metricOptions"
          :key="item.id"
          :class="{ selected: model.metrics.includes(item.id) }"
          @click="toggleMetric(item.id)"
        >{{ item.label }}</button>
      </div>
    </div>

    <div class="config-section">
      <div class="field-row">
        <label class="field-label">04 · 对比算法</label>
        <span class="field-value">最多 5 项</span>
      </div>
      <div class="algorithm-picker">
        <button
          v-for="algorithm in algorithms"
          :key="algorithm.id"
          :class="{ selected: model.model_ids.includes(algorithm.id), disabled: !isCompatible(algorithm) }"
          :disabled="!isCompatible(algorithm)"
          @click="toggleAlgorithm(algorithm.id)"
        >
          <span class="check-mark">{{ model.model_ids.includes(algorithm.id) ? '✓' : '+' }}</span>
          <span><strong>{{ algorithm.short_name }}</strong><small>{{ algorithm.family }}</small></span>
        </button>
      </div>
    </div>

    <div class="config-section compact-section">
      <div class="field-row">
        <label class="field-label">05 · 实现版本</label>
        <span class="field-value">{{ referenceCount }} 个模型有 sklearn 对照</span>
      </div>
      <div class="implementation-picker">
        <button
          v-for="item in [{ id: 'scratch', label: 'Scratch 自实现' }, { id: 'sklearn', label: 'Sklearn 对照' }]"
          :key="item.id"
          :class="{ selected: model.implementations.includes(item.id) }"
          @click="toggleImplementation(item.id)"
        >
          <span class="check-mark">{{ model.implementations.includes(item.id) ? '✓' : '+' }}</span>
          <b>{{ item.label }}</b>
        </button>
      </div>
      <p class="field-hint">选择两项会在完全相同的数据划分上运行；暂未提供 sklearn Adapter 的模型只运行 Scratch。</p>
    </div>

    <div class="config-section compact-section">
      <button class="advanced-toggle" @click="advancedOpen = !advancedOpen">
        <span><b>06 · 训练参数</b><small>随机种子、标准化与模型超参数</small></span>
        <i :class="{ open: advancedOpen }">⌄</i>
      </button>

      <div v-if="advancedOpen" class="advanced-body">
        <div class="inline-fields">
          <label>随机种子<input v-model.number="model.seed" type="number" min="0" max="1000000" /></label>
          <label class="switch-field">
            特征标准化
            <button class="switch" :class="{ on: model.standardize }" @click="model.standardize = !model.standardize"><span></span></button>
          </label>
        </div>
        <div v-for="algorithm in selectedAlgorithms" :key="algorithm.id" class="parameter-group">
          <h4>{{ algorithm.short_name }}</h4>
          <div class="parameter-grid">
            <label v-for="parameter in algorithm.parameter_schema" :key="parameter.key">
              {{ parameter.label }}
              <button
                v-if="parameter.type === 'boolean'"
                class="switch mini"
                :class="{ on: model.parameters[algorithm.id][parameter.key] }"
                @click="model.parameters[algorithm.id][parameter.key] = !model.parameters[algorithm.id][parameter.key]"
              ><span></span></button>
              <select
                v-else-if="parameter.type === 'select'"
                v-model="model.parameters[algorithm.id][parameter.key]"
              >
                <option v-for="option in parameter.options" :key="option" :value="option">{{ option }}</option>
              </select>
              <input
                v-else
                v-model.number="model.parameters[algorithm.id][parameter.key]"
                type="number"
                :min="parameter.min"
                :max="parameter.max"
                :step="parameter.step"
              />
            </label>
          </div>
        </div>
        <button class="text-button" @click="resetParameters">恢复推荐参数</button>
      </div>
    </div>

    <button class="run-button" :disabled="running" @click="emit('run')">
      <span v-if="running" class="button-spinner"></span>
      <span v-else class="play-icon">▶</span>
      {{ running ? '实验进行中' : '开始训练与评估' }}
      <small>{{ running ? '请留在当前页面' : '实时返回各模型结果' }}</small>
    </button>
  </aside>
</template>
