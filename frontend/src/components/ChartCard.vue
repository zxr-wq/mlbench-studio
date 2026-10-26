<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, watch } from 'vue'
const props=defineProps<{ title: string, description?: string, wide?: boolean }>()
const expanded = ref(false)
const card = ref<HTMLElement>()
const notice = ref('')
let previousFocus:HTMLElement|null=null
function close(event: KeyboardEvent) {
  if (event.key === 'Escape') expanded.value = false
  if(expanded.value && event.key==='Tab') {
    const items=Array.from(card.value?.querySelectorAll<HTMLElement>('button,select,input,[tabindex="0"]') ?? []).filter((item)=>!(item as HTMLButtonElement).disabled)
    const first=items[0],last=items.at(-1)
    if(event.shiftKey && document.activeElement===first) {event.preventDefault();last?.focus()}
    else if(!event.shiftKey && document.activeElement===last) {event.preventDefault();first?.focus()}
  }
}
watch(expanded, async (open)=>{
  if(open) { previousFocus=document.activeElement as HTMLElement; await nextTick(); card.value?.querySelector('button')?.focus() }
  else { await nextTick(); previousFocus?.focus() }
})
window.addEventListener('keydown', close)
onBeforeUnmount(() => window.removeEventListener('keydown', close))
async function exportImage() {
  const source = card.value?.querySelector('svg')
  if (!source) { notice.value = '此图为交互表格；可下载实验 JSON 或指标 CSV。'; return }
  const clone = source.cloneNode(true) as SVGSVGElement
  const originals = [source, ...source.querySelectorAll('*')]
  const copies = [clone, ...clone.querySelectorAll('*')]
  originals.forEach((element, index) => {
    const style = getComputedStyle(element)
    for (const property of ['fill', 'stroke', 'stroke-width', 'font-family', 'font-size', 'font-weight', 'text-anchor', 'opacity', 'stroke-dasharray']) {
      (copies[index] as SVGElement).style.setProperty(property, style.getPropertyValue(property))
    }
  })
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  const box=source.viewBox.baseVal
  clone.setAttribute('viewBox',`${box.x} ${box.y-44} ${box.width} ${box.height+44}`)
  const heading=document.createElementNS('http://www.w3.org/2000/svg','text')
  heading.setAttribute('x',String(box.x+box.width/2)); heading.setAttribute('y',String(box.y-16))
  heading.setAttribute('text-anchor','middle'); heading.setAttribute('fill','#263c29'); heading.setAttribute('stroke','none'); heading.setAttribute('font-size','18'); heading.setAttribute('font-family','Microsoft YaHei, sans-serif'); heading.textContent=props.title
  clone.prepend(heading)
  clone.setAttribute('width', '1280')
  clone.setAttribute('height', String(1280 * (box.height+44) / box.width))
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)], { type: 'image/svg+xml' }))
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = `mlbench-chart-${props.title.replace(/[\\/:*?"<>|]/g,'-')}.svg`; anchor.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
</script>

<template>
  <Teleport to="body" :disabled="!expanded">
    <div v-if="expanded" class="chart-modal-backdrop" @click="expanded = false"></div>
    <article ref="card" class="chart-card evidence-chart" :class="{ 'chart-expanded': expanded, 'chart-wide': wide }" :role="expanded ? 'dialog' : undefined" :aria-modal="expanded || undefined" :aria-label="title">
      <header class="chart-heading"><div><h3>{{ title }}</h3><p v-if="description">{{ description }}</p></div><div class="chart-tools"><button type="button" @click="exportImage" title="下载可缩放 SVG 图片" aria-label="下载图表">↓</button><button type="button" @click="expanded = !expanded" :title="expanded ? '关闭放大' : '放大查看'" :aria-label="expanded ? '关闭放大' : '放大图表'">{{ expanded ? '×' : '⛶' }}</button></div></header>
      <p v-if="notice" class="inline-notice" role="status">{{ notice }}</p>
      <slot />
    </article>
  </Teleport>
</template>
