<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { storeToRefs } from 'pinia'
import ChatPanel from './components/ChatPanel.vue'
import EntriesPanel from './components/EntriesPanel.vue'
import HelpTip from './components/HelpTip.vue'
import AccessGate from './components/AccessGate.vue'
import { downloadCalendar, getHealth } from './api/client'
import { useEntriesStore } from './stores/entries'
import { useSessionStore } from './stores/session'
import { useSpeech } from './composables/useSpeech'
import { timezone } from './utils/time'

const mocks = import.meta.env.VITE_USE_MOCKS === 'true'
const HEALTH_INTERVAL_MS = 30_000

const session = useSessionStore()
const { loaded, unlocked } = storeToRefs(session)
const entries = useEntriesStore()
const { supported: ttsSupported, enabled: ttsEnabled } = useSpeech()

const health = ref<'unknown' | 'ok' | 'down'>('unknown')
const healthTitle = ref('Проверяю сервер…')
const toast = ref<string | null>(null)
const exporting = ref(false)
let timer: number | undefined
let toastTimer: number | undefined

function notify(text: string) {
  toast.value = text
  window.clearTimeout(toastTimer)
  toastTimer = window.setTimeout(() => (toast.value = null), 5000)
}

async function checkHealth() {
  try {
    const h = await getHealth()
    health.value = 'ok'
    healthTitle.value = `Сервер работает: ${h.model} на ${h.device}`
  } catch {
    health.value = 'down'
    healthTitle.value = 'Нет связи с сервером'
  }
}

async function exportCalendar() {
  if (exporting.value) return
  exporting.value = true
  try {
    const blob = await downloadCalendar()
    const href = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = href
    a.download = 'assistant.ics'
    a.click()
    URL.revokeObjectURL(href)
    notify('Файл assistant.ics скачан — откройте его, чтобы добавить записи в календарь')
  } catch (e) {
    notify(e instanceof Error ? e.message : 'Не удалось выгрузить календарь')
  } finally {
    exporting.value = false
  }
}

function onVisible() {
  if (document.visibilityState === 'visible') {
    void entries.refresh(true)
    void checkHealth()
  }
}

onMounted(() => {
  void session.load()
  void checkHealth()
  timer = window.setInterval(checkHealth, HEALTH_INTERVAL_MS)
  document.addEventListener('visibilitychange', onVisible)
})
onBeforeUnmount(() => {
  window.clearInterval(timer)
  window.clearTimeout(toastTimer)
  document.removeEventListener('visibilitychange', onVisible)
})
</script>

<template>
  <div class="layout">
    <header class="topbar">
      <div class="brand">
        <h1>Голосовой ассистент</h1>
        <HelpTip />
      </div>
      <div class="meta">
        <span v-if="mocks" class="badge">моки</span>
        <button
          v-if="ttsSupported"
          type="button"
          class="icon-action"
          :class="{ active: ttsEnabled }"
          :aria-pressed="ttsEnabled"
          :title="ttsEnabled ? 'Озвучка ответов включена' : 'Озвучка ответов выключена'"
          :aria-label="ttsEnabled ? 'Выключить озвучку ответов' : 'Включить озвучку ответов'"
          @click="ttsEnabled = !ttsEnabled"
        >
          <svg v-if="ttsEnabled" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path fill="currentColor" d="M3 9v6h4l5 5V4L7 9H3Zm13.5 3A4.5 4.5 0 0 0 14 8v8a4.47 4.47 0 0 0 2.5-4ZM14 3.23v2.06a7 7 0 0 1 0 13.42v2.06A9 9 0 0 0 14 3.23Z" />
          </svg>
          <svg v-else viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path fill="currentColor" d="M16.5 12A4.5 4.5 0 0 0 14 8v2.18l2.45 2.45c.03-.2.05-.41.05-.63ZM19 12c0 .94-.2 1.82-.54 2.64l1.51 1.51A8.8 8.8 0 0 0 21 12a9 9 0 0 0-7-8.77v2.06A7 7 0 0 1 19 12ZM4.27 3 3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06a9 9 0 0 0 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3ZM12 4 9.91 6.09 12 8.18V4Z" />
          </svg>
        </button>
        <button
          v-if="unlocked && loaded"
          type="button"
          class="icon-action"
          :disabled="exporting"
          title="Скачать записи для календаря (.ics)"
          aria-label="Скачать записи для календаря"
          @click="exportCalendar"
        >
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path fill="currentColor" d="M19 4h-1V2h-2v2H8V2H6v2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2Zm0 16H5V9h14v11ZM7 11h5v5H7v-5Z" />
          </svg>
        </button>
        <span class="status" :class="health" :title="healthTitle">
          <span class="status-dot" aria-hidden="true" />
          <span class="status-label">{{ health === 'down' ? 'нет связи' : 'онлайн' }}</span>
        </span>
        <span class="tz" title="Часовой пояс">{{ timezone }}</span>
      </div>
    </header>
    <main v-if="!loaded" class="main-loading">Загрузка…</main>
    <main v-else-if="!unlocked" class="main-gate">
      <AccessGate />
    </main>
    <main v-else class="main">
      <ChatPanel />
      <EntriesPanel />
    </main>
    <transition name="toast">
      <div v-if="toast" class="toast" role="status">{{ toast }}</div>
    </transition>
  </div>
</template>
