<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useChatStore } from '@/stores/chat'
import { useSessionStore } from '@/stores/session'
import { MAX_RECORD_SECONDS, useRecorder } from '@/composables/useRecorder'
import { stopSpeech } from '@/composables/useSpeech'
import EntryItem from './EntryItem.vue'

const chat = useChatStore()
const { messages, sending, awaiting, retryLast } = storeToRefs(chat)
const { voiceEnabled } = storeToRefs(useSessionStore())

const draft = ref('')
const listEl = ref<HTMLElement | null>(null)
const inputEl = ref<HTMLTextAreaElement | null>(null)

const recorder = useRecorder(({ blob, filename }) => chat.sendAudio(blob, filename))
const { isRecording, elapsed, error: recError } = recorder

const hasText = computed(() => draft.value.trim().length > 0)
const mode = computed<'send' | 'mic'>(() => (isRecording.value || hasText.value || !voiceEnabled.value ? 'send' : 'mic'))
const timer = computed(() => `${Math.floor(elapsed.value / 60)}:${String(elapsed.value % 60).padStart(2, '0')}`)
const actionLabel = computed(() => {
  if (sending.value) return 'Отправляю'
  if (isRecording.value) return 'Отправить голосовое'
  return mode.value === 'send' ? 'Отправить' : 'Записать голосовое'
})

function primary() {
  if (sending.value) return
  if (isRecording.value) {
    recorder.stop()
    return
  }
  if (hasText.value) {
    void chat.sendText(draft.value)
    draft.value = ''
    return
  }
  if (!voiceEnabled.value) return
  stopSpeech()
  void recorder.start()
}

function autoresize() {
  const el = inputEl.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 140)}px`
}
watch(draft, () => nextTick(autoresize))

function onGlobalKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && isRecording.value) recorder.cancel()
}
onMounted(() => window.addEventListener('keydown', onGlobalKeydown))
onBeforeUnmount(() => window.removeEventListener('keydown', onGlobalKeydown))

watch(sending, async (v) => {
  if (v) return
  await nextTick()
  inputEl.value?.focus()
})

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    if (hasText.value) primary()
  }
}

watch(
  () => messages.value.length,
  async () => {
    await nextTick()
    listEl.value?.scrollTo({ top: listEl.value.scrollHeight, behavior: 'smooth' })
  },
)
</script>

<template>
  <section class="panel chat" aria-label="Чат с ассистентом">
    <div ref="listEl" class="messages" aria-live="polite">
      <p v-if="!messages.length" class="empty">
        Скажите или напишите, например: «Напомни мне завтра в 10 утра позвонить Ивану» или «Что у меня сегодня?»
      </p>
      <div v-for="m in messages" :key="m.id" class="msg" :class="[m.role, { pending: m.pending }]">
        <div class="bubble">
          <span v-if="m.voice" class="voice-mark">
            <svg viewBox="0 0 24 24" width="12" height="12" aria-hidden="true"><path fill="currentColor" d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2Z" /></svg>
            голосовое
          </span>
          <p class="text">{{ m.text }}</p>
          <span v-if="m.status === 'needs_clarification'" class="tag">ждёт ответа</span>
          <ul v-if="m.entries?.length" class="entries-inline">
            <EntryItem v-for="e in m.entries" :key="`${e.id}:${e.date ?? ''}`" :entry="e" compact />
          </ul>
        </div>
      </div>
      <div v-if="sending" class="msg assistant">
        <div class="bubble typing"><span /><span /><span /></div>
      </div>
    </div>

    <div v-if="retryLast && !sending" class="banner error">
      Не удалось отправить.
      <button type="button" class="link" @click="chat.retry()">Повторить</button>
    </div>
    <div v-if="recError" class="banner error">{{ recError }}</div>
    <div v-if="awaiting" class="banner info">
      Ассистент ждёт ответа на уточнение.
      <button type="button" class="link" @click="chat.reset()">Начать заново</button>
    </div>
    <div v-else-if="messages.length && !sending" class="toolbar">
      <button type="button" class="link muted" @click="chat.reset()">Очистить чат</button>
    </div>

    <form class="composer" @submit.prevent="primary">
      <div class="field-pill" :class="{ recording: isRecording }">
        <template v-if="isRecording">
          <span class="rec-dot" aria-hidden="true" />
          <span class="rec-timer">{{ timer }}</span>
          <span class="rec-limit">из 0:{{ MAX_RECORD_SECONDS }}</span>
          <button type="button" class="rec-cancel" @click="recorder.cancel()">Отмена</button>
        </template>
        <textarea
          v-else
          ref="inputEl"
          v-model="draft"
          rows="1"
          aria-label="Сообщение ассистенту"
          :placeholder="awaiting ? 'Ответьте на уточнение…' : 'Сообщение'"
          :disabled="sending"
          @keydown="onKeydown"
        />
      </div>
      <button
        type="submit"
        class="round-btn"
        :class="{ recording: isRecording, busy: sending }"
        :disabled="sending || (mode === 'mic' && !recorder.supported) || (!voiceEnabled && !hasText)"
        :aria-label="actionLabel"
        :title="mode === 'mic' && !recorder.supported ? 'Браузер не поддерживает запись' : actionLabel"
      >
        <span v-if="sending" class="spinner" aria-hidden="true" />
        <svg v-else-if="mode === 'send'" viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
          <path fill="currentColor" d="M3.4 20.4 21 12 3.4 3.6 3.39 10.1 15 12 3.39 13.9z" />
        </svg>
        <svg v-else viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
          <path fill="currentColor" d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2Z" />
        </svg>
      </button>
    </form>
  </section>
</template>
