<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import AppIcon from '../components/AppIcon.vue'
import { datasets } from '../data/catalog'
import { useLabStore } from '../stores/lab'

const { runs, stats } = useLabStore()
const bestRun = computed(() => [...runs.value]
  .filter((run) => (run.taskType ?? 'classification') === 'classification')
  .sort((a, b) => (b.primaryValue ?? b.accuracy) - (a.primaryValue ?? a.accuracy))[0])
const recentRuns = computed(() => runs.value.slice(0, 4))
</script>

<template>
  <div class="page overview-page">
    <section class="hero-grid">
      <div class="hero-copy">
        <span class="overline">MACHINE LEARNING LAB · 01</span>
        <h1>让每一次实验<br><em>都成为证据。</em></h1>
        <p>用同一套 Pipeline 复现、比较并解释经典机器学习算法，而不是把结果散落在十个脚本里。</p>
        <div class="hero-actions">
          <RouterLink to="/experiments/new" class="button primary">
            <AppIcon name="plus" :size="16" />创建实验
          </RouterLink>
          <RouterLink to="/runs" class="button ghost">浏览全部运行<AppIcon name="arrow" :size="15" /></RouterLink>
        </div>
      </div>

      <div class="hero-figure">
        <div class="figure-head">
          <span>{{ bestRun ? '最佳测试准确率' : '等待首次真实实验' }}</span>
          <span class="live-mark"><i></i> REAL RUNS</span>
        </div>
        <strong v-if="bestRun">{{ ((bestRun.primaryValue ?? bestRun.accuracy) * 100).toFixed(1) }}<small>%</small></strong>
        <strong v-else>--<small>%</small></strong>
        <p>{{ bestRun ? `${bestRun.model} · ${bestRun.dataset}` : '运行一次分类实验后，这里将显示真实结果' }}</p>
        <svg class="score-line" viewBox="0 0 460 110" preserveAspectRatio="none">
          <path class="line-area" d="M0 90 C55 86 60 70 110 75 S180 38 225 55 S300 22 345 36 S410 10 460 18 L460 110 L0 110Z"/>
          <path class="line-stroke" d="M0 90 C55 86 60 70 110 75 S180 38 225 55 S300 22 345 36 S410 10 460 18"/>
          <circle cx="460" cy="18" r="4"/>
        </svg>
        <div class="figure-scale"><span>TRAIN</span><span>EVALUATE</span></div>
      </div>
    </section>

    <section class="metric-strip">
      <div><span>已登记算法</span><strong>{{ stats.algorithms }}</strong><small>5 个类别</small></div>
      <div><span>自主实现</span><strong>{{ stats.scratchAlgorithms }}</strong><small>十大算法完成度 100%</small></div>
      <div><span>可用数据集</span><strong>{{ stats.datasets }}</strong><small>{{ datasets.reduce((sum, item) => sum + item.samples, 0).toLocaleString() }} 条样本</small></div>
      <div><span>实验运行</span><strong>{{ stats.completedRuns }}</strong><small>全部可复现</small></div>
    </section>

    <div class="overview-columns">
      <section class="content-section recent-section">
        <div class="section-title-row">
          <div><span class="overline">RECENT RUNS</span><h2>刚刚发生的实验</h2></div>
          <RouterLink to="/runs" class="text-link">查看全部 <AppIcon name="arrow" :size="14" /></RouterLink>
        </div>
        <div class="recent-list">
          <RouterLink v-for="run in recentRuns" :key="run.id" :to="`/runs/${run.id}`" class="recent-row">
            <span class="run-state"><i></i></span>
            <span class="run-name"><b>{{ run.id }}</b><small>{{ run.model }} on {{ run.dataset }}</small></span>
            <span class="run-metric"><small>Accuracy</small><b>{{ run.accuracy.toFixed(3) }}</b></span>
            <span class="run-time"><small>{{ run.createdAt }}</small><b>{{ run.duration.toFixed(2) }}s</b></span>
            <AppIcon name="arrow" :size="15" />
          </RouterLink>
          <div v-if="!recentRuns.length" class="overview-empty"><b>还没有真实实验记录</b><p>启动本地后端并创建实验，结果会自动保存到这里。</p><RouterLink to="/experiments/new" class="text-link">创建第一次实验 <AppIcon name="arrow" :size="14" /></RouterLink></div>
        </div>
      </section>

      <aside class="content-section focus-section">
        <div class="section-title-row">
          <div><span class="overline">THIS WEEK</span><h2>本周焦点</h2></div>
        </div>
        <div class="focus-card">
          <span class="focus-number">10</span>
          <p>十大算法均已注册并可运行</p>
          <div class="focus-progress"><span style="width: 100%"></span></div>
          <small>Scratch 与 sklearn 共用统一 Runner</small>
        </div>
        <div class="next-item">
          <span>验收建议</span>
          <b>完成多随机种子 Benchmark</b>
          <small>报告均值与标准差，减少单次划分的偶然性</small>
        </div>
        <RouterLink to="/algorithms" class="plain-link">打开算法注册表 <AppIcon name="arrow" :size="14" /></RouterLink>
      </aside>
    </div>
  </div>
</template>
