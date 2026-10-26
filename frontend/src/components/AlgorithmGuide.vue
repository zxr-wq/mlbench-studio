<script setup lang="ts">
import { computed } from 'vue'
import { algorithmEducation, metricEducation, modelKey } from '../data/algorithmEducation'
import { algorithms } from '../data/catalog'

const props = defineProps<{ model: string, taskType?: string, metrics?: Record<string, number | number[]>, visualization?: Record<string, any> }>()
const key = computed(() => modelKey(props.model))
const education = computed(() => algorithmEducation[key.value])
const modelName = computed(() => algorithms.find((item) => item.id === key.value)?.name ?? props.model)
const title = computed(() => ({ classification: '分类结果怎么读？', regression: '回归结果怎么读？', clustering: '聚类结果怎么读？', dimensionality_reduction: 'PCA 结果怎么读？' }[props.taskType ?? 'classification'] ?? '结果怎么读？'))
const metricTips = computed(() => metricEducation[props.taskType as keyof typeof metricEducation] ?? metricEducation.classification)
const conclusion=computed(()=>{
  const metrics=props.metrics ?? {},chart=props.visualization ?? {},lines:string[]=[]
  const value=(key:string)=>{const item=metrics[key];return Array.isArray(item) ? item.at(-1) : item}
  if(props.taskType==='classification') {
    if(value('accuracy')!=null) lines.push(`本次 Accuracy 为 ${(Number(value('accuracy'))*100).toFixed(1)}%，表示在所选评价设置下约有这些比例的样本被正确分类。`)
    if(value('f1')!=null) lines.push(`Macro F1 为 ${Number(value('f1')).toFixed(3)}，它先分别评价每个类别，再等权平均。`)
    const matrix=chart.classification?.confusion_matrix
    if(matrix) {
      let wrong=0,total=0;matrix.forEach((row:number[],r:number)=>row.forEach((n,c)=>{total+=n;if(r!==c)wrong+=n}))
      lines.push(`代表性运行的混淆矩阵中，${total} 个测试样本有 ${wrong} 个误判。多次实验时，这个数量只对应图表注明的种子与折。`)
    }
  } else if(props.taskType==='regression') {
    if(value('r2')!=null) lines.push(`R² 为 ${Number(value('r2')).toFixed(3)}；${Number(value('r2'))<0 ? '当前模型比用测试集目标均值作常数预测还差，应重新检查特征与模型选择。' : '越接近 1，测试数据中被模型解释的目标变化越多。'}`)
    if(value('rmse')!=null) lines.push(`RMSE 为 ${Number(value('rmse')).toFixed(2)}，与目标量纲一致；是否足够小要结合目标取值范围和应用允许误差判断。`)
  } else if(props.taskType==='clustering') {
    if(value('silhouette_score')!=null) lines.push(`轮廓系数为 ${Number(value('silhouette_score')).toFixed(3)}，用于衡量当前聚类的紧密与分离程度，不等于标签预测准确率。`)
    lines.push('播放过程中簇规模与 Inertia 会随轮次更新；顶部指标始终对应最终聚类。改变 K 后，Inertia 不能直接当成公平排名。')
  } else {
    if(value('cumulative_explained_variance')!=null) lines.push(`所保留主成分累计解释约 ${(Number(value('cumulative_explained_variance'))*100).toFixed(1)}% 的训练方差；损失的信息是否重要取决于后续任务。`)
    lines.push('散点图只显示前两个主成分；当保留维度大于 2 时，散点不会展示所有保留信息。')
  }
  return lines
})
</script>

<template>
  <section class="algorithm-guide">
    <header><span>LEARN BY THIS RUN</span><h2>本次实验结论与解读</h2><p>结合指标与图表，分析本次实验的表现、适用场景与结果含义。</p></header>
    <div class="experiment-conclusion"><p v-for="line in conclusion" :key="line">{{ line }}</p></div>
    <div class="guide-columns">
      <article><b>① {{ modelName }} 的原理</b><p>{{ education?.principle ?? '该模型使用统一训练与评价流程。' }}</p><small>适合什么情况</small><p>{{ education?.useWhen }}</p></article>
      <article><b>② {{ title }}</b><div v-for="tip in metricTips" :key="tip[0]" class="metric-tip"><strong>{{ tip[0] }}</strong><p>{{ tip[1] }}</p></div></article>
      <article><b>③ 本算法可以看哪些图</b><ul><li v-for="chart in education?.visuals" :key="chart">{{ chart }}</li></ul><p class="guide-note">可用图表取决于实现版本和数据集，缺少真实输出的项目不会生成。</p></article>
    </div>
  </section>
</template>
