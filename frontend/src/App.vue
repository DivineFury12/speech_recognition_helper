<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { storeToRefs } from 'pinia'
import ChatPanel from './components/ChatPanel.vue'
import EntriesPanel from './components/EntriesPanel.vue'
import HelpTip from './components/HelpTip.vue'
import AccessGate from './components/AccessGate.vue'
import { getCalendarLink, getHealth } from './api/client'
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
const healthTitle = ref('Проверяю backend…')
const calendarUrl = ref<string | null>(null)
const calendarNote = ref<string | null>(null)
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

async function openCalendar() {
  calendarNote.value = null
  try {
    const { url } = await getCalendarLink()
    calendarUrl.value = url
    try {
      await navigator.clipboard.writeText(url)
      calendarNote.value = 'Ссылка скопирована — добавьте её в календарь как подписку'
    } catch {
      calendarNote.value = 'Скопируйте ссылку и добавьте её в календарь как подписку'
    }
  } catch (e) {
    calendarNote.value = e instanceof Error ? e.message : 'Не удалось получить ссылку'
  }
}

function onVisible() {
  if (document.visibilityState === 'visible') {
    void entries.refresh()
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
        <label v-if="ttsSupported" class="toggle" title="Читать ответы вслух">
          <input v-model="ttsEnabled" type="checkbox" />
          Озвучка
        </label>
        <button v-if="unlocked" type="button" class="link" @click="openCalendar">В календарь</button>
        <span class="health" :class="health" :title="healthTitle">{{ health === 'down' ? 'нет связи' : 'backend' }}</span>
        <span class="tz">{{ timezone }}</span>
      </div>
    </header>
    <div v-if="calendarNote" class="banner info calendar-note">
      {{ calendarNote }}
      <template v-if="calendarUrl">
        <code class="calendar-url">{{ calendarUrl }}</code>
        <a :href="calendarUrl" class="link" download="assistant.ics">Скачать .ics</a>
      </template>
      <button type="button" class="link muted" @click="calendarNote = null">Скрыть</button>
    </div>
    <main v-if="!loaded" class="main-loading">Загрузка…</main>
    <main v-else-if="!unlocked" class="main-gate">
      <AccessGate />
    </main>
    <main v-else class="main">
      <ChatPanel />
      <EntriesPanel />
    </main>
  </div>
</template>
