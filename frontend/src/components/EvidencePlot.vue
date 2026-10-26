<script setup lang="ts">
import { computed } from 'vue'
export interface PlotPoint { x: number, y: number, label?: string, hollow?: boolean }
export interface PlotSeries { name: string, color: string, points: PlotPoint[], kind?: 'line' | 'scatter' | 'bar', dashed?: boolean, barWidth?: number }
const props = defineProps<{ series: PlotSeries[], xLabel: string, yLabel: string, xRange?: number[], yRange?: number[], zeroLine?: boolean, diagonal?: boolean, percentY?: boolean, integerX?: boolean, integerY?: boolean, xTicks?: Array<{ value: number, label: string }>, legend?: boolean }>()
const emit = defineEmits<{ point: [point: PlotPoint, index: number, series: number] }>()
function range(axis: 'x' | 'y', explicit?: number[]) {
  if (explicit?.length === 2) return explicit[0]===explicit[1] ? [explicit[0]-.5,explicit[1]+.5] : explicit
  const values = props.series.flatMap((series) => series.points.map((point) => point[axis])).filter(Number.isFinite)
  let lo = Math.min(...values), hi = Math.max(...values)
  if (!values.length) return [0, 1]
  if (lo === hi) { lo -= .5; hi += .5 }
  const padding = (hi - lo) * .07
  return [lo - padding, hi + padding]
}
const xr = computed(() => range('x', props.xRange))
const yr = computed(() => range('y', props.yRange))
const px = (x: number) => 66 + (x - xr.value[0]!) / (xr.value[1]! - xr.value[0]!) * 540
const py = (y: number) => 266 - (y - yr.value[0]!) / (yr.value[1]! - yr.value[0]!) * 236
const ticks = (r: number[], integer=false) => [...new Set(Array.from({ length: 5 }, (_, i) => {const value=r[0]! + i * (r[1]! - r[0]!) / 4;return integer ? Math.round(value) : value}))]
function number(value: number) {
  if (value === 0) return '0'
  if (Math.abs(value) >= 1e4 || Math.abs(value) < .001) return value.toExponential(1)
  return Number(value.toPrecision(3)).toString()
}
const xt = computed(() => props.xTicks ?? ticks(xr.value,props.integerX).map((value) => ({ value, label: number(value) })))
const line = (points: PlotPoint[]) => points.map((point) => `${px(point.x)},${py(point.y)}`).join(' ')
const barWidth = (series:PlotSeries) => series.barWidth ? Math.abs(px(series.barWidth)-px(0)) : Math.max(4,460/series.points.length)
</script>

<template>
  <svg class="evidence-plot" viewBox="0 0 640 330" role="img" :aria-label="`${xLabel}与${yLabel}`" style="background: #fbfbf9; stroke: none; fill: none">
    <g v-for="tick in ticks(yr,integerY)" :key="tick"><line x1="66" x2="606" :y1="py(tick)" :y2="py(tick)" stroke="#e1e2dc" stroke-width="1"/><text x="56" :y="py(tick) + 4" text-anchor="end" fill="#62665f" font-size="12">{{ percentY ? `${Math.round(tick * 100)}%` : number(tick) }}</text></g>
    <g v-for="tick in xt" :key="tick.value"><line :x1="px(tick.value)" :x2="px(tick.value)" y1="266" y2="271" stroke="#b6b9af"/><text :x="px(tick.value)" y="289" text-anchor="middle" fill="#62665f" font-size="12">{{ tick.label }}</text></g>
    <path d="M66 30V266H606" stroke="#a5ab9f" stroke-width="1.2"/>
    <line v-if="zeroLine && yr[0]! <= 0 && yr[1]! >= 0" x1="66" x2="606" :y1="py(0)" :y2="py(0)" stroke="#c8462c" stroke-dasharray="5 4"/>
    <line v-if="diagonal" :x1="px(Math.max(xr[0]!, yr[0]!))" :y1="py(Math.max(xr[0]!, yr[0]!))" :x2="px(Math.min(xr[1]!, yr[1]!))" :y2="py(Math.min(xr[1]!, yr[1]!))" stroke="#c8462c" stroke-dasharray="5 4"/>
    <g v-for="(item, seriesIndex) in series" :key="item.name">
      <polyline v-if="!item.kind || item.kind === 'line'" :points="line(item.points)" :stroke="item.color" stroke-width="2.4" :stroke-dasharray="item.dashed ? '5 4' : undefined" fill="none"/>
      <g v-for="(point, index) in item.points" :key="index">
        <rect v-if="item.kind === 'bar'" :x="px(point.x) - barWidth(item)/2" :y="py(point.y)" :width="barWidth(item)" :height="Math.max(0, py(Math.max(0, yr[0]!)) - py(point.y))" :fill="item.color" opacity=".8"><title>{{ point.label ?? item.name }}: {{ number(point.y) }}</title></rect>
        <circle v-else-if="item.kind === 'scatter'" :cx="px(point.x)" :cy="py(point.y)" :r="point.hollow ? 4.6 : 3.6" :fill="point.hollow ? '#fff' : item.color" :stroke="item.color" stroke-width="1.5" opacity=".8" @click="emit('point', point, index, seriesIndex)"><title>{{ point.label ?? item.name }} · x={{ number(point.x) }}, y={{ number(point.y) }}</title></circle>
      </g>
    </g>
    <text x="336" y="317" text-anchor="middle" fill="#454a41" font-size="13">{{ xLabel }}</text>
    <text transform="translate(16 148) rotate(-90)" text-anchor="middle" fill="#454a41" font-size="13">{{ yLabel }}</text>
  </svg>
  <div v-if="legend !== false" class="plot-legend"><span v-for="item in series" :key="item.name"><i :style="{ background: item.color }"></i>{{ item.name }}</span></div>
</template>
