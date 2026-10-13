<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import ChatPanel from './components/ChatPanel.vue'
import EntriesPanel from './components/EntriesPanel.vue'
import HelpTip from './components/HelpTip.vue'
import AccessGate from './components/AccessGate.vue'
import VoicePicker from './components/VoicePicker.vue'
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

const { notice: speechNotice } = useSpeech()
watch(speechNotice, (text) => {
  if (!text) return
  notify(text)
  speechNotice.value = null
})

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
        <VoicePicker />
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
