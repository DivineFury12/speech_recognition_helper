<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import ChatPanel from './components/ChatPanel.vue'
import EntriesPanel from './components/EntriesPanel.vue'
import HelpTip from './components/HelpTip.vue'
import AccessGate from './components/AccessGate.vue'
import VoicePicker from './components/VoicePicker.vue'
import CalendarMenu from './components/CalendarMenu.vue'
import { getHealth, getServicesStatus } from './api/client'
import type { ServiceState } from './api/types'
import { useChatStore } from './stores/chat'
import { useEntriesStore } from './stores/entries'
import { useSessionStore } from './stores/session'
import { useSpeech } from './composables/useSpeech'
import { timezone } from './utils/time'

const mocks = import.meta.env.VITE_USE_MOCKS === 'true'
const HEALTH_INTERVAL_MS = 30_000

const session = useSessionStore()
const { loaded, unlocked, voiceEnabled } = storeToRefs(session)
const entries = useEntriesStore()

const health = ref<'unknown' | 'ok' | 'warn' | 'down'>('unknown')
const services = ref<ServiceState[]>([])
const statusOpen = ref(false)
const statusEl = ref<HTMLElement | null>(null)
const toast = ref<string | null>(null)
const chat = useChatStore()
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
    await getHealth()
  } catch {
    health.value = 'down'
    services.value = []
    return
  }
  try {
    services.value = (await getServicesStatus()).services
    health.value = services.value.some((s) => s.state === 'error') ? 'warn' : 'ok'
  } catch {
    health.value = 'ok'
  }
}

const statusLabel = () => (health.value === 'down' ? 'нет связи' : health.value === 'warn' ? 'есть проблемы' : 'онлайн')

function onDocumentClick(e: MouseEvent) {
  if (statusOpen.value && statusEl.value && !statusEl.value.contains(e.target as Node)) statusOpen.value = false
}

function readGoogleResult() {
  const params = new URLSearchParams(window.location.search)
  const result = params.get('google')
  if (!result) return
  const text: Record<string, string> = {
    connected: 'Google Календарь подключён — записи появятся в календаре «Ассистент»',
    denied: 'Подключение Google Календаря отменено',
    error: 'Не удалось подключить Google Календарь, попробуйте ещё раз',
  }
  notify(text[result] ?? 'Google Календарь: неизвестный ответ')
  params.delete('google')
  const query = params.toString()
  window.history.replaceState(null, '', window.location.pathname + (query ? `?${query}` : ''))
}

watch([loaded, unlocked], ([isLoaded, isUnlocked]) => {
  if (isLoaded && isUnlocked) {
    void chat.showDailySummary()
    void checkHealth()
  }
})

function onVisible() {
  if (document.visibilityState === 'visible') {
    void entries.refresh(true)
    void checkHealth()
  }
}

onMounted(() => {
  void session.load()
  void checkHealth()
  readGoogleResult()
  timer = window.setInterval(checkHealth, HEALTH_INTERVAL_MS)
  document.addEventListener('visibilitychange', onVisible)
  document.addEventListener('click', onDocumentClick)
})
onBeforeUnmount(() => {
  window.clearInterval(timer)
  window.clearTimeout(toastTimer)
  document.removeEventListener('visibilitychange', onVisible)
  document.removeEventListener('click', onDocumentClick)
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
        <VoicePicker v-if="voiceEnabled" />
        <CalendarMenu v-if="unlocked && loaded" @notify="notify" />
        <div ref="statusEl" class="status-wrap">
          <button
            type="button"
            class="status"
            :class="health"
            :aria-expanded="statusOpen"
            title="Состояние сервисов"
            @click.stop="statusOpen = !statusOpen; statusOpen && checkHealth()"
          >
            <span class="status-dot" aria-hidden="true" />
            <span class="status-label">{{ statusLabel() }}</span>
          </button>
          <div v-show="statusOpen" class="menu status-menu" role="dialog" aria-label="Состояние сервисов">
            <div class="menu-title">Состояние сервисов</div>
            <p v-if="health === 'down'" class="service-row">
              <span class="service-dot error" />
              <span><b>Сервер</b><br /><small>нет связи — проверьте, что backend запущен</small></span>
            </p>
            <p v-for="s in services" :key="s.name" class="service-row">
              <span class="service-dot" :class="s.state" />
              <span><b>{{ s.name }}</b><br /><small>{{ s.detail }}</small></span>
            </p>
          </div>
        </div>
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
