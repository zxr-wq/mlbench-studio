<script setup lang="ts">
defineProps<{ modelValue:number, items:string[], label:string }>()
const emit=defineEmits<{ 'update:modelValue':[value:number] }>()
function jump(event:Event, count:number, current:number) {
  const input=event.target as HTMLInputElement
  const value=Number(input.value)
  if(input.value.trim() && Number.isInteger(value) && value>=1 && value<=count) emit('update:modelValue',value-1)
  else input.value=String(current+1)
}
</script>

<template>
  <div class="sample-navigator" role="group" :aria-label="label">
    <button type="button" :disabled="modelValue<=0" @click="emit('update:modelValue',modelValue-1)" :aria-label="`${label}：上一个样本`">← 上一个</button>
    <label>样本 <input type="number" :aria-label="`${label}编号`" :value="modelValue+1" min="1" :max="items.length" step="1" @change="jump($event,items.length,modelValue)" @keydown.enter.prevent="jump($event,items.length,modelValue)"/> / {{ items.length }}</label>
    <button type="button" :disabled="modelValue>=items.length-1" @click="emit('update:modelValue',modelValue+1)" :aria-label="`${label}：下一个样本`">下一个 →</button>
    <span class="sample-label" aria-live="polite">{{ items[modelValue] }}</span>
  </div>
</template>
