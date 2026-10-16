<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import ChatPanel from './components/ChatPanel.vue'
import AgendaPanel from './components/AgendaPanel.vue'
import EntryEditor from './components/EntryEditor.vue'
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
import { useReminders } from './composables/useReminders'
import { useTheme } from './composables/useTheme'
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
const theme = useTheme()
const mobileTab = ref<'chat' | 'agenda'>('chat')
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

const reminders = useReminders()
watch(reminders.lastShown, (text) => {
  if (text) notify(text)
})

async function toggleReminders() {
  notify(await reminders.toggle())
}

watch([loaded, unlocked], ([isLoaded, isUnlocked]) => {
  if (isLoaded && isUnlocked) {
    reminders.startTimer()
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
        <h1><span class="brand-long">Голосовой </span>ассистент</h1>
        <HelpTip />
      </div>
      <div class="meta">
        <span v-if="mocks" class="badge">моки</span>
        <VoicePicker v-if="voiceEnabled" />
        <button
          v-if="unlocked && loaded && reminders.supported"
          type="button"
          class="icon-action"
          :class="{ active: reminders.enabled.value }"
          :title="reminders.enabled.value ? 'Уведомления о напоминаниях включены' : 'Включить уведомления о напоминаниях'"
          :aria-pressed="reminders.enabled.value"
          aria-label="Уведомления о напоминаниях"
          @click="toggleReminders"
        >
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path fill="currentColor" d="M12 22a2 2 0 0 0 2-2h-4a2 2 0 0 0 2 2Zm6-6V11a6 6 0 0 0-5-5.91V4a1 1 0 0 0-2 0v1.09A6 6 0 0 0 6 11v5l-2 2v1h16v-1l-2-2Z" />
          </svg>
        </button>
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
        <button
          type="button"
          class="icon-action"
          :title="theme.label.value"
          :aria-label="theme.label.value"
          @click="theme.cycle()"
        >
          <svg v-if="theme.mode.value === 'dark'" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path fill="currentColor" d="M12.3 22A10 10 0 0 1 9.6 2.4a8 8 0 0 0 12 10A10 10 0 0 1 12.3 22Z" />
          </svg>
          <svg v-else-if="theme.mode.value === 'light'" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path fill="currentColor" d="M12 17a5 5 0 1 1 0-10 5 5 0 0 1 0 10Zm0-15 1 3h-2l1-3Zm0 20-1-3h2l-1 3ZM2 12l3-1v2l-3-1Zm20 0-3 1v-2l3 1ZM4.9 4.9l2.8 1.4-1.4 1.4-1.4-2.8Zm14.2 14.2-2.8-1.4 1.4-1.4 1.4 2.8Zm0-14.2-1.4 2.8-1.4-1.4 2.8-1.4ZM4.9 19.1l1.4-2.8 1.4 1.4-2.8 1.4Z" />
          </svg>
          <svg v-else viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path fill="currentColor" d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 18V4a8 8 0 0 1 0 16Z" />
          </svg>
        </button>
        <span class="tz" title="Часовой пояс">{{ timezone }}</span>
      </div>
    </header>
    <main v-if="!loaded" class="main-loading">Загрузка…</main>
    <main v-else-if="!unlocked" class="main-gate">
      <AccessGate />
    </main>
    <main v-else class="main" :class="`show-${mobileTab}`">
      <ChatPanel />
      <AgendaPanel />
    </main>
    <nav v-if="loaded && unlocked" class="tabbar" aria-label="Разделы">
      <button type="button" :class="{ active: mobileTab === 'chat' }" :aria-pressed="mobileTab === 'chat'" @click="mobileTab = 'chat'">
        <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path fill="currentColor" d="M4 4h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8l-4 4V6a2 2 0 0 1 2-2Z" /></svg>
        Чат
      </button>
      <button type="button" :class="{ active: mobileTab === 'agenda' }" :aria-pressed="mobileTab === 'agenda'" @click="mobileTab = 'agenda'">
        <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path fill="currentColor" d="M19 4h-1V2h-2v2H8V2H6v2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2Zm0 16H5V9h14v11ZM7 11h5v5H7v-5Z" /></svg>
        Записи
      </button>
    </nav>
    <EntryEditor />
    <transition name="toast">
      <div v-if="toast" class="toast" role="status">{{ toast }}</div>
    </transition>
  </div>
</template>
