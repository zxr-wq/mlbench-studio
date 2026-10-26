<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import ChartCard from './ChartCard.vue'
import EvidencePlot, { type PlotSeries } from './EvidencePlot.vue'
const props = defineProps<{ visualization?: Record<string, any>, taskType?: string }>()
const v = computed(() => props.visualization ?? {})
const palette = ['#297958', '#3679b5', '#c77735', '#8855a2', '#bf4d70', '#94752d', '#397f87', '#6a7098', '#b5682c', '#4b6e3b']
const color = (label: number) => palette[Number(label) % palette.length]
const labels = computed<string[]>(() => v.value.target_names ?? [])
const name = (label: number) => labels.value[label] ?? String(label)
const cls = computed(() => v.value.classification)
const reg = computed(() => v.value.regression)
const cluster = computed(() => v.value.clustering)
const pca = computed(() => v.value.pca)
const source = computed(() => v.value.provenance)
const percent = ref(false)
const round = (value: number) => Number(value.toPrecision(3)).toString()
function matrixStrength(value:number,row:number[]) { return value / Math.max(1,percent.value ? row.reduce((a,b)=>a+b,0) : Math.max(...cls.value.confusion_matrix.flat())) }
function byClass(points: any[], labelKey: string, hollow = false): PlotSeries[] {
  const classes = [...new Set(points.map((point) => Number(point[labelKey])))].sort((a, b) => a - b)
  return classes.map((label) => ({ name: name(label), color: color(label), kind: 'scatter', points: points.filter((point) => Number(point[labelKey]) === label).map((point) => ({ ...point, hollow: hollow && point.actual !== point.predicted, label: `${name(label)}${point.actual !== point.predicted && hollow ? ' · 误判' : ''}` })) }))
}
const classSeries = computed(() => byClass(cls.value?.points ?? [], 'actual', true))
const rocSeries = computed<PlotSeries[]>(() => [{ name: `ROC · AUC ${cls.value?.roc_curve?.auc}`, color: palette[0], points: (cls.value?.roc_curve?.points ?? []).map((p: any) => ({ x: p.fpr, y: p.tpr })) }])
const regressionSeries = computed<PlotSeries[]>(() => [{ name: '测试样本', color: palette[0], kind: 'scatter', points: (reg.value?.pairs ?? []).map((p: any) => ({ x: p.actual, y: p.predicted })) }])
const residualSeries = computed<PlotSeries[]>(() => [{ name: '真实值 − 预测值', color: palette[1], kind: 'scatter', points: (reg.value?.residuals ?? []).map((p: any) => ({ x: p.predicted, y: p.residual })) }])
const histogramSeries = computed<PlotSeries[]>(() => {
  const h = reg.value?.residual_histogram
  return [{ name: '残差频数', color: palette[2], kind: 'bar', points: h ? h.counts.map((count: number, i: number) => ({ x: (h.edges[i] + h.edges[i + 1]) / 2, y: count, label: `[${round(h.edges[i])}, ${round(h.edges[i + 1])}]` })) : [] }]
})
const lossSeries = computed<PlotSeries[]>(() => [{ name: '训练损失', color: palette[2], points: (v.value.loss_curve ?? []).map((y: number, index: number) => ({ x:v.value.loss_iterations?.[index] ?? index, y })) }])
const ensembleSeries = computed<PlotSeries[]>(() => [{ name: '测试准确率（同一已训练集成的前 N 棵树）', color: palette[0], points: (v.value.ensemble_curve ?? []).map((p: any) => ({ x: p.iteration, y: p.accuracy })) }])
const cumulative = computed<PlotSeries[]>(() => {
  let total = 0
  return [{ name: '累计解释方差', color: palette[0], points: (pca.value?.explained_variance_ratio ?? []).map((y: number, i: number) => ({ x: i + 1, y: total += y })) }]
})
const variance = computed<PlotSeries[]>(() => [{ name: '单个主成分解释方差', color: palette[2], kind: 'bar', points: (pca.value?.explained_variance_ratio ?? []).map((y: number, i: number) => ({ x: i + 1, y })) }])
const probabilitiesIndex = ref(0)
const probabilities = computed(() => v.value.probabilities)
const probabilityRow = computed(() => probabilities.value?.rows?.[probabilitiesIndex.value])
const gaussianIndex = ref(0)
const gaussian = computed(() => v.value.gaussian_features?.[gaussianIndex.value])
const gaussianSeries = computed<PlotSeries[]>(() => (gaussian.value?.curves ?? []).map((curve: number[], label: number) => ({ name: name(label), color: color(label), points: curve.map((y, i) => ({ x: gaussian.value.x[i], y })) })))

const iteration = ref(0)
const playing = ref(false)
let timer: number | undefined
const history = computed<any[][]>(() => cluster.value?.centroid_history ?? [])
const currentCenters = computed(() => history.value[iteration.value] ?? [])
const currentAssignment = computed(() => cluster.value?.iterations?.[iteration.value])
const clusterColors = computed(() => cluster.value?.points?.map((point: any, index: number) => ({ ...point, label: currentAssignment.value?.labels?.[index] ?? point.cluster })) ?? [])
const clusterSeries = computed<PlotSeries[]>(() => {
  const groups = [...new Set<number>(clusterColors.value.map((p: any) => p.label))].sort((a,b)=>a-b)
  const points: PlotSeries[] = groups.map((label) => ({ name: `簇 ${label + 1}`, color: color(label), kind: 'scatter', points: clusterColors.value.filter((p: any) => p.label === label) }))
  currentCenters.value.forEach((point, index) => {
    points.push({ name: `质心 ${index + 1}`, color: color(index), kind: 'scatter', points: [{ ...point, hollow: true, label: `质心 ${index + 1}` }] })
    points.push({ name: `轨迹 ${index + 1}`, color: color(index), points: history.value.slice(0, iteration.value + 1).map((step) => step[index]).filter(Boolean), dashed: true })
  })
  return points
})
const clusterRange = computed(() => {
  const points = [...(cluster.value?.points ?? []), ...history.value.flat()]
  const range = (key: string) => { const a = points.map((p: any) => Number(p[key])); const lo = Math.min(...a), hi = Math.max(...a); const pad = (hi-lo || 1)*.08; return [lo-pad,hi+pad] }
  return { x: range('x'), y: range('y') }
})
const clusterSizes = computed(() => {
  if (!currentAssignment.value) return cluster.value?.sizes ?? []
  return currentCenters.value.map((_, label) => currentAssignment.value.labels.filter((item: number) => item === label).length)
})
function stop() { playing.value = false; window.clearInterval(timer); timer = undefined }
function play() {
  if (playing.value) { stop(); return }
  if (iteration.value >= history.value.length - 1) iteration.value = 0
  playing.value = true
  timer = window.setInterval(() => { if (iteration.value >= history.value.length - 1) stop(); else iteration.value++ }, 800)
}
watch(() => props.visualization, () => { stop(); iteration.value = Math.max(0, history.value.length - 1); neighborIndex.value = 0; probabilitiesIndex.value = 0 }, { immediate: false })
watch(history, () => { iteration.value = Math.max(0, history.value.length - 1) }, { immediate: true })
onBeforeUnmount(stop)

const neighborIndex = ref(0)
const neighborK = ref(5)
const knn = computed(() => v.value.knn)
watch(knn, () => { neighborK.value = knn.value?.k ?? 5 }, { immediate: true })
const example = computed(() => knn.value?.examples?.[neighborIndex.value])
const activeNeighbors = computed<any[]>(() => example.value?.neighbors?.slice(0, neighborK.value) ?? [])
const voteRows = computed(() => {
  const rows = new Map<number, number>()
  activeNeighbors.value.forEach((p) => rows.set(p.label, (rows.get(p.label) ?? 0) + (knn.value.weights === 'distance' ? 1 / Math.max(p.distance, 1e-12) : 1)))
  return [...rows].sort((a,b) => b[1]-a[1] || a[0]-b[0])
})
const neighborSeries = computed<PlotSeries[]>(() => {
  const points: PlotSeries[] = [{ name: '训练样本（投影）', color: '#bcc4bb', kind: 'scatter', points: knn.value?.training ?? [] }]
  activeNeighbors.value.forEach((p, i) => {
    const neighbor = knn.value.training[p.index]
    points.push({ name: `邻居 ${i+1} · ${name(p.label)}`, color: color(p.label), points: [example.value, neighbor] })
  })
  if (example.value) points.push({ name: `待预测点 · 实际 ${name(example.value.actual)}`, color: '#1b2520', kind: 'scatter', points: [{ ...example.value, hollow: true }] })
  return points
})

const treeDepth = ref(4)
const treeWidth = computed(() => Math.max(1000, 2 ** (treeDepth.value - 1) * 145 + 100))
const treeNodes = computed(() => {
  const nodes: any[] = []
  const visit = (node: any, depth: number, lo: number, hi: number, parent?: any) => {
    if (!node || depth >= treeDepth.value) return
    const counts: number[] = node.value ?? []
    const label = counts.indexOf(Math.max(...counts))
    const leaf = !node.left && !node.right
    const truncated = !leaf && depth === treeDepth.value - 1
    const title = leaf ? `预测 ${name(label)}` : `${node.feature} ≤ ${Number(node.threshold).toFixed(3)}`
    const current = { id: nodes.length, x: (lo+hi)/2, y: 38+depth*90, px: parent?.x, py: parent?.y, title, label: title.length > 22 ? title.slice(0,20)+'…' : title, detail: `${node.samples} 样本 · ${counts.join('/')}`, leaf, truncated }
    nodes.push(current)
    visit(node.left, depth+1, lo, current.x, current); visit(node.right, depth+1, current.x, hi, current)
  }
  visit(v.value.tree_structure, 0, 70, treeWidth.value-70)
  return nodes
})

const boundary = computed(() => v.value.svm_boundary)
const svmTraining = ref(true)
const svmCells = computed(() => {
  const b = boundary.value
  if (!b) return []
  return b.labels.flatMap((row: number[], yi: number) => row.map((label, xi) => ({ x: xi, y: yi, label, score: b.scores[yi][xi] })))
})
const sx = (value: number) => 66 + (value-boundary.value.xs[0])/(boundary.value.xs.at(-1)-boundary.value.xs[0])*540
const sy = (value: number) => 266 - (value-boundary.value.ys[0])/(boundary.value.ys.at(-1)-boundary.value.ys[0])*236
const boundarySegments = computed(() => {
  const b = boundary.value; if (!b) return []
  const segments: Array<{ x1:number,y1:number,x2:number,y2:number }> = []
  // Marching squares at f(x)=0, interpolated from the actual decision scores.
  for(let y=0;y<b.ys.length-1;y++) for(let x=0;x<b.xs.length-1;x++) {
    const vertices = [[x,y],[x+1,y],[x+1,y+1],[x,y+1]]
    const intersections: number[][] = []
    for(let edge=0;edge<4;edge++) {
      const a=vertices[edge], c=vertices[(edge+1)%4], av=b.scores[a[1]][a[0]], cv=b.scores[c[1]][c[0]]
      if((av < 0) !== (cv < 0)) { const t=av/(av-cv); intersections.push([sx(b.xs[a[0]]+t*(b.xs[c[0]]-b.xs[a[0]])),sy(b.ys[a[1]]+t*(b.ys[c[1]]-b.ys[a[1]]))]) }
    }
    for(let i=0;i+1<intersections.length;i+=2) segments.push({x1:intersections[i][0],y1:intersections[i][1],x2:intersections[i+1][0],y2:intersections[i+1][1]})
  }
  return segments
})
const contributionLabels: Record<string,string> = { 'absolute coefficient': '系数绝对值', 'tree split count': '分裂次数', 'forest split count': '森林分裂次数', 'boosting split count': '提升树分裂次数', 'feature importance': '不纯度重要性' }
</script>

<template>
  <section class="visual-suite" v-if="Object.keys(v).length">
    <div class="visual-suite-head"><div><span class="section-kicker">RESULT EXPLORER</span><h2>实验可视化</h2></div><p>可放大查看、悬停读数并下载 SVG</p></div>
    <p class="evidence-provenance" v-if="source"><b>图表来源：seed={{ source.seed }}{{ source.strategy === 'kfold' ? ` · 第 ${source.fold} 折` : '' }}</b><span v-if="source.total_runs > 1">；图表对应这一次代表性运行，顶部指标为 {{ source.total_runs }} 次运行的均值 ± 标准差。</span><span v-else>；所有图表与顶部指标来自同一次运行。</span></p>
    <p v-else class="evidence-provenance">这是一条旧版记录，未保存图表种子信息；请重新运行以获得完整来源说明。</p>
    <div class="evidence-grid">
      <ChartCard v-if="cls" title="混淆矩阵" description="行：真实类别；列：预测类别。悬停读取原始数量与该类召回比例。">
        <div class="chart-controls"><button :class="{ active: !percent }" @click="percent=false">原始数量</button><button :class="{ active: percent }" @click="percent=true">按真实类别归一化</button></div>
        <div class="matrix-scroll" tabindex="0" aria-label="可横向滚动的混淆矩阵"><svg :viewBox="`0 0 ${110+cls.labels.length*57} ${70+cls.labels.length*43}`" :style="{ minWidth: `${110+cls.labels.length*57}px` }" class="matrix-svg" role="img" aria-label="混淆矩阵">
          <text v-for="(label,i) in cls.labels" :key="`c-${i}`" :x="110+Number(i)*57+26" y="35" text-anchor="middle" fill="#535e51" font-size="10">{{ label.length > 11 ? label.slice(0,10)+'…' : label }}<title>{{ label }}</title></text>
          <g v-for="(row,ri) in cls.confusion_matrix" :key="ri"><text x="102" :y="67+Number(ri)*43" text-anchor="end" fill="#535e51" font-size="11">{{ cls.labels[ri].length > 14 ? cls.labels[ri].slice(0,13)+'…' : cls.labels[ri] }}</text><g v-for="(value,ci) in row" :key="ci"><rect :x="110+Number(ci)*57" :y="45+Number(ri)*43" width="53" height="39" rx="4" :fill="`rgba(41,121,88,${.08+.9*matrixStrength(Number(value),row)})`"/><text :x="136+Number(ci)*57" :y="69+Number(ri)*43" text-anchor="middle" :fill="matrixStrength(Number(value),row)>.5 ? '#fff' : '#284837'" font-size="12">{{ percent ? `${(100*Number(value)/Math.max(1,row.reduce((a:number,b:number)=>a+b,0))).toFixed(1)}%` : value }}</text><title>{{ cls.labels[ri] }} → {{ cls.labels[ci] }}：{{ value }} 个，{{ (100*Number(value)/Math.max(1,row.reduce((a:number,b:number)=>a+b,0))).toFixed(1) }}%</title></g></g>
        </svg></div>
      </ChartCard>
      <ChartCard v-if="cls" title="真实与预测类别分布" description="比较模型是否过多预测某个类别；数量来自本次测试集。"><EvidencePlot :series="[{ name:'真实',color:palette[0],kind:'bar',barWidth:.3,points:cls.labels.map((label:string,i:number)=>({x:i-.17,y:cls.actual_distribution[i],label})) }, {name:'预测',color:palette[2],kind:'bar',barWidth:.3,points:cls.labels.map((label:string,i:number)=>({x:i+.17,y:cls.predicted_distribution[i],label}))}]" x-label="类别" y-label="样本数" :x-range="[-.6,cls.labels.length-.4]" :y-range="[0,Math.max(1,...cls.actual_distribution,...cls.predicted_distribution)*1.15]" :x-ticks="cls.labels.map((label:string,i:number)=>({value:i,label}))"/></ChartCard>
      <ChartCard v-if="cls" title="测试集二维投影" description="PCA 方向仅在训练集拟合。颜色表示真实类别，空心点表示误判；此图不表示模型决策边界。"><EvidencePlot :series="classSeries" x-label="训练 PCA 第一主成分" y-label="训练 PCA 第二主成分"/></ChartCard>
      <ChartCard v-if="cls?.roc_curve" title="ROC 曲线" description="以第二个类别为正类，虚线为随机排序基线。AUC 越接近 1 越好。"><EvidencePlot :series="rocSeries" x-label="假阳性率 FPR" y-label="真阳性率 TPR" :x-range="[0,1]" :y-range="[0,1]" diagonal/></ChartCard>
      <ChartCard v-if="reg" title="真实值与预测值" description="横轴是真实目标，纵轴是预测目标；越接近 y=x 参考线越准确。"><EvidencePlot :series="regressionSeries" x-label="真实值" y-label="预测值" diagonal/></ChartCard>
      <ChartCard v-if="reg?.residuals" title="残差诊断" description="残差 = 真实值 − 预测值；围绕零线随机散布较理想，弯曲或漏斗形提示模型不足。"><EvidencePlot :series="residualSeries" x-label="预测值" y-label="残差" zero-line/></ChartCard>
      <ChartCard v-if="reg" title="残差分布" description="横轴为误差区间，纵轴为区间内样本数；悬停柱体查看精确区间。"><EvidencePlot :series="histogramSeries" x-label="残差（真实值 − 预测值）" y-label="样本频数" :y-range="[0,Math.max(1,...reg.residual_histogram.counts)*1.15]"/></ChartCard>
      <ChartCard v-if="cluster" title="K-Means 质心迭代" :description="history.length ? '颜色按该轮真实分配更新；空心点是质心，虚线是质心历史轨迹。' : '该实现未返回逐轮轨迹，展示最终聚类结果。'" wide>
        <div v-if="history.length" class="chart-controls playback"><button @click="stop(); iteration=0">重置</button><button :disabled="iteration===0" @click="stop(); iteration--">上一步</button><button @click="play">{{ playing ? '暂停' : '播放' }}</button><button :disabled="iteration===history.length-1" @click="stop(); iteration++">下一步</button><label>轮次 {{ iteration }} / {{ history.length-1 }}<input aria-label="质心迭代轮次" type="range" min="0" :max="history.length-1" v-model.number="iteration" @input="stop"/></label><span v-if="currentAssignment">该轮 Inertia {{ currentAssignment.inertia.toFixed(3) }}</span></div>
        <EvidencePlot :series="clusterSeries" :x-range="clusterRange.x" :y-range="clusterRange.y" x-label="训练 PCA 第一主成分" y-label="训练 PCA 第二主成分" :legend="false"/><div class="plot-legend"><span v-for="(_,i) in clusterSizes" :key="i"><i :style="{background:color(Number(i))}"></i>簇 {{ Number(i)+1 }} · {{ clusterSizes[i] }} 样本</span></div>
      </ChartCard>
      <ChartCard v-if="pca" title="主成分解释方差" :description="`展示全部 ${pca.explained_variance_ratio.length} 个保留主成分；柱高是各自保留的训练集方差比例。`"><EvidencePlot :series="variance" x-label="保留主成分序号" y-label="方差比例" :x-range="[.5,pca.explained_variance_ratio.length+.5]" integer-x :y-range="[0,1]" percent-y/></ChartCard>
      <ChartCard v-if="pca" title="累计解释方差" description="观察保留信息随主成分增加的变化。解释方差来自训练集，不代表分类准确率。"><EvidencePlot :series="cumulative" x-label="保留主成分数" y-label="累计方差比例" :x-range="[1,pca.explained_variance_ratio.length]" integer-x :y-range="[0,1]" percent-y/></ChartCard>
      <ChartCard v-if="pca" title="PCA 测试集投影" description="测试样本映射到训练得到的前两个主成分。标签颜色只用于解释，未参与 PCA 拟合。"><EvidencePlot :series="byClass(pca.points,'label')" x-label="第一主成分" y-label="第二主成分"/></ChartCard>
      <ChartCard v-if="v.loss_curve?.length>1" title="训练损失曲线" description="横轴为实际记录的训练轮次；下降不等于泛化更好，应结合测试指标判断。"><EvidencePlot :series="lossSeries" x-label="训练轮次" y-label="训练损失" :x-range="[0,Math.max(1,lossSeries[0]?.points.at(-1)?.x ?? 1)]" integer-x/></ChartCard>
      <ChartCard v-if="v.ensemble_curve?.length" title="集成规模与测试表现" description="复用已训练模型，逐步加入前 N 棵树得到测试准确率；不能据此反复挑选测试集最优参数。"><EvidencePlot :series="ensembleSeries" x-label="参与预测的树 / 轮数" y-label="Accuracy" :x-range="[1,Math.max(2,ensembleSeries[0]?.points.at(-1)?.x ?? 2)]" integer-x :y-range="[0,1]"/></ChartCard>
      <ChartCard v-if="v.feature_contributions" title="特征贡献" :description="`${contributionLabels[v.feature_contributions.label] ?? v.feature_contributions.label}；展示前 ${v.feature_contributions.items.length} 项。系数与分裂次数不能解释成因果关系。`"><svg class="feature-svg" :viewBox="`0 0 640 ${Math.max(180,v.feature_contributions.items.length*30+30)}`" role="img" aria-label="特征贡献"><g v-for="(item,i) in v.feature_contributions.items" :key="item.name"><text x="185" :y="30+Number(i)*30" text-anchor="end" fill="#525d4c" font-size="12">{{ item.name.length>24 ? item.name.slice(0,23)+'…' : item.name }}<title>{{ item.name }}</title></text><rect x="195" :y="17+Number(i)*30" :width="360*item.value/Math.max(...v.feature_contributions.items.map((x:any)=>x.value))" height="18" rx="3" fill="#ba674b"/><text x="625" :y="30+Number(i)*30" text-anchor="end" fill="#525d4c" font-size="12">{{ round(item.value) }}</text></g></svg></ChartCard>
      <ChartCard v-if="probabilities" title="单样本类别概率" description="预测概率来自真实模型输出，未必经过校准；只保存前 40 个测试样本。"><div class="chart-controls"><label>测试样本 <select v-model.number="probabilitiesIndex"><option v-for="row in probabilities.rows" :key="row.sample" :value="row.sample">#{{ row.sample+1 }} · 实际 {{ name(row.actual) }}</option></select></label></div><EvidencePlot v-if="probabilityRow" :series="[{name:'预测概率',color:palette[0],kind:'bar',points:probabilityRow.values.map((y:number,x:number)=>({x,y,label:name(x)}))}]" :x-ticks="probabilities.classes.map((value:number,i:number)=>({value:i,label:name(value)}))" x-label="类别" y-label="模型概率" :y-range="[0,1]" percent-y/></ChartCard>
      <ChartCard v-if="gaussian" title="朴素贝叶斯特征分布" description="使用模型训练出的均值和方差绘制高斯密度；不把分布假设当成原始数据分布。"><div class="chart-controls"><label>查看特征 <select v-model.number="gaussianIndex"><option v-for="(item,i) in v.gaussian_features" :key="item.name" :value="i">{{ item.name }}</option></select></label></div><EvidencePlot :series="gaussianSeries" :x-label="`${gaussian.name}（预处理后）`" y-label="高斯概率密度"/></ChartCard>
      <ChartCard v-if="knn" title="KNN 邻居投票实验" description="选择测试点并改变演示 K；距离在全部预处理特征上计算，二维线段只辅助展示。演示不会修改顶部正式实验结果。" wide>
        <div class="chart-controls"><label>测试样本 <select v-model.number="neighborIndex"><option v-for="(row,i) in knn.examples" :key="row.sample" :value="i">#{{ row.sample+1 }} · 实际 {{ name(row.actual) }}</option></select></label><label>演示 K={{ neighborK }}<input aria-label="演示近邻数量 K" v-model.number="neighborK" type="range" min="1" :max="example?.neighbors?.length ?? 15"/></label><span>{{ knn.distance }} · {{ knn.weights === 'uniform' ? '等权投票' : '距离加权' }}</span></div>
        <div class="diagnostic-split"><EvidencePlot :series="neighborSeries" x-label="训练 PCA 第一主成分" y-label="训练 PCA 第二主成分" :legend="false"/><div><p class="vote-result" v-if="voteRows.length">演示预测：<b>{{ name(voteRows[0][0]) }}</b></p><div v-for="[label,votes] in voteRows" :key="label" class="vote-row"><i :style="{background:color(label)}"></i>{{ name(label) }}<b>{{ knn.weights === 'uniform' ? `${votes} 票` : votes.toFixed(3) }}</b></div><ol class="neighbor-list"><li v-for="(item,i) in activeNeighbors" :key="item.index">邻居 {{ i+1 }} · {{ name(item.label) }}<span>d={{ item.distance.toFixed(3) }}</span></li></ol></div></div>
      </ChartCard>
      <ChartCard v-if="boundary" title="SVM 真实决策区域与支持向量" :description="`${boundary.kernel.toUpperCase()} 核 · 在原始二维特征经预处理后绘图。底色来自真实模型预测，黑线由 decision_function 的零水平集插值得到，圆环为支持向量。`" wide>
        <div class="chart-controls"><label><input type="checkbox" v-model="svmTraining"/>显示训练样本</label><span>{{ boundary.supports.length }} 个去重支持向量 · 测试点为实心</span></div>
        <svg class="evidence-plot boundary-plot" viewBox="0 0 640 330" role="img" aria-label="SVM 决策区域和真实零边界" style="stroke:none;fill:none;background:#fbfbf9">
          <rect v-for="(cell,i) in svmCells" :key="i" :x="66+cell.x*540/(boundary.xs.length-1)" :y="266-(cell.y+1)*236/(boundary.ys.length-1)" :width="540/(boundary.xs.length-1)+.3" :height="236/(boundary.ys.length-1)+.3" :fill="color(cell.label)" opacity=".13"/>
          <line v-for="(segment,i) in boundarySegments" :key="`line-${i}`" v-bind="segment" stroke="#1b2922" stroke-width="1.7"/>
          <g v-if="svmTraining"><circle v-for="(point,i) in boundary.training" :key="i" :cx="sx(point.x)" :cy="sy(point.y)" r="2.5" :fill="color(point.label)" opacity=".55"/></g>
          <circle v-for="(point,i) in boundary.test" :key="`test-${i}`" :cx="sx(point.x)" :cy="sy(point.y)" r="3.5" :fill="color(point.label)" stroke="#fff" stroke-width=".7"><title>测试样本 · {{ name(point.label) }}</title></circle>
          <circle v-for="(point,i) in boundary.supports" :key="`support-${i}`" :cx="sx(point.x)" :cy="sy(point.y)" r="5" stroke="#354c3e" stroke-width="1" fill="none"/>
          <path d="M66 30V266H606" stroke="#9da894"/>
          <g v-for="i in 5" :key="`tick-${i}`"><text :x="66+(i-1)*135" y="289" text-anchor="middle" fill="#586250" font-size="12">{{ round(boundary.xs[0]+(i-1)*(boundary.xs.at(-1)-boundary.xs[0])/4) }}</text><text x="56" :y="270-(i-1)*59" text-anchor="end" fill="#586250" font-size="12">{{ round(boundary.ys[0]+(i-1)*(boundary.ys.at(-1)-boundary.ys[0])/4) }}</text></g>
          <text x="336" y="317" text-anchor="middle" fill="#454a41" font-size="13">{{ boundary.feature_names[0] }}（预处理后）</text><text transform="translate(16 148) rotate(-90)" text-anchor="middle" fill="#454a41" font-size="13">{{ boundary.feature_names[1] }}（预处理后）</text>
        </svg><div class="plot-legend"><span v-for="(label,i) in labels" :key="label"><i :style="{background:color(Number(i))}"></i>{{ label }}</span><span>○ 支持向量</span><span>— f(x)=0 决策边界</span></div>
      </ChartCard>
      <p v-if="v.svm_note" class="evidence-provenance">{{ v.svm_note }}</p>
      <ChartCard v-if="v.tree_structure" title="决策树规则" description="左分支满足条件，右分支不满足；阈值基于所选预处理后的特征。悬停查看完整节点文字，可增加层数并横向滚动。" wide>
        <div class="chart-controls"><label>显示层数 <select v-model.number="treeDepth"><option v-for="n in [3,4,5,6]" :key="n" :value="n">前 {{ n }} 层</option></select></label></div>
        <div class="tree-scroll" tabindex="0"><svg :viewBox="`0 0 ${treeWidth} ${treeDepth*90+10}`" :style="{minWidth:`${treeWidth}px`}" class="tree-svg" role="img" aria-label="决策树结构">
          <line v-for="node in treeNodes.filter((p)=>p.px!=null)" :key="`e-${node.id}`" :x1="node.px" :y1="node.py+22" :x2="node.x" :y2="node.y-22" stroke="#a6b9a8"/>
          <g v-for="node in treeNodes" :key="node.id" :transform="`translate(${node.x},${node.y})`"><rect x="-65" y="-24" width="130" height="50" rx="7" :fill="node.leaf ? '#e7efe8' : '#fffaf1'" stroke="#7f9d85"/><text y="-5" text-anchor="middle" fill="#233d29" font-size="10.5">{{ node.label }}</text><text y="10" text-anchor="middle" fill="#616e5c" font-size="9">{{ node.detail }}</text><text v-if="node.truncated" y="40" text-anchor="middle" fill="#b45a39" font-size="10">↓ 仍有子节点</text><title>{{ node.title }} · {{ node.detail }}{{ node.truncated ? ' · 节点已截断，请增加显示层数' : '' }}</title></g>
        </svg></div>
      </ChartCard>
    </div>
  </section>
</template>
