<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import ChatPanel from './components/ChatPanel.vue'
import EntriesPanel from './components/EntriesPanel.vue'
import { getHealth } from './api/client'
import { useEntriesStore } from './stores/entries'
import { timezone } from './utils/time'

const mocks = import.meta.env.VITE_USE_MOCKS === 'true'
const HEALTH_INTERVAL_MS = 30_000

const health = ref<'unknown' | 'ok' | 'down'>('unknown')
const healthTitle = ref('Проверяю backend…')
let timer: number | undefined

async function checkHealth() {
  try {
    const h = await getHealth()
    health.value = 'ok'
    healthTitle.value = `Backend готов: ${h.model} на ${h.device}`
  } catch {
    health.value = 'down'
    healthTitle.value = 'Backend недоступен'
  }
}

const entries = useEntriesStore()
function onVisible() {
  if (document.visibilityState === 'visible') {
    void entries.refresh()
    void checkHealth()
  }
}

onMounted(() => {
  void checkHealth()
  timer = window.setInterval(checkHealth, HEALTH_INTERVAL_MS)
  document.addEventListener('visibilitychange', onVisible)
})
onBeforeUnmount(() => {
  window.clearInterval(timer)
  document.removeEventListener('visibilitychange', onVisible)
})
</script>

<template>
  <div class="layout">
    <header class="topbar">
      <h1>Голосовой ассистент</h1>
      <span class="meta">
        <span v-if="mocks" class="badge">моки</span>
        <span class="health" :class="health" :title="healthTitle">{{ health === 'down' ? 'нет связи' : 'backend' }}</span>
        <span>{{ timezone }}</span>
      </span>
    </header>
    <main class="main">
      <ChatPanel />
      <EntriesPanel />
    </main>
  </div>
</template>
