<script setup lang="ts">
defineProps<{ name: string; size?: number }>()
</script>

<template>
  <svg :width="size || 18" :height="size || 18" viewBox="0 0 24 24" aria-hidden="true">
    <template v-if="name === 'overview'">
      <rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>
    </template>
    <template v-else-if="name === 'flask'">
      <path d="M9 3h6M10 3v6l-5.3 8.4A2.4 2.4 0 0 0 6.8 21h10.4a2.4 2.4 0 0 0 2.1-3.6L14 9V3M7.2 14h9.6"/>
    </template>
    <template v-else-if="name === 'runs'">
      <path d="M4 5h16M4 12h16M4 19h16"/><circle cx="7" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="17" cy="19" r="1"/>
    </template>
    <template v-else-if="name === 'algorithm'">
      <circle cx="5" cy="12" r="2.2"/><circle cx="19" cy="6" r="2.2"/><circle cx="19" cy="18" r="2.2"/><path d="m7.1 11.1 9.8-4.2M7.1 12.9l9.8 4.2"/>
    </template>
    <template v-else-if="name === 'dataset'">
      <ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/>
    </template>
    <template v-else-if="name === 'search'">
      <circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.2 4.2"/>
    </template>
    <template v-else-if="name === 'plus'">
      <path d="M12 5v14M5 12h14"/>
    </template>
    <template v-else-if="name === 'arrow'">
      <path d="M5 12h14M14 7l5 5-5 5"/>
    </template>
    <template v-else-if="name === 'collapse'">
      <path d="m14 7-5 5 5 5"/>
    </template>
    <template v-else-if="name === 'expand'">
      <path d="m10 7 5 5-5 5"/>
    </template>
    <template v-else-if="name === 'check'">
      <path d="m5 12 4 4L19 7"/>
    </template>
    <template v-else-if="name === 'clock'">
      <circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>
    </template>
    <template v-else-if="name === 'filter'">
      <path d="M4 6h16M7 12h10M10 18h4"/>
    </template>
    <template v-else-if="name === 'code'">
      <path d="m8 8-4 4 4 4M16 8l4 4-4 4M14 5l-4 14"/>
    </template>
    <template v-else>
      <circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2"/>
    </template>
  </svg>
</template>
