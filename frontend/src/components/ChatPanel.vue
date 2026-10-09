<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useChatStore } from '@/stores/chat'
import { MAX_RECORD_SECONDS, useRecorder } from '@/composables/useRecorder'
import EntryItem from './EntryItem.vue'

const chat = useChatStore()
const { messages, sending, conversationId, retryLast } = storeToRefs(chat)

const draft = ref('')
const listEl = ref<HTMLElement | null>(null)
const inputEl = ref<HTMLTextAreaElement | null>(null)

const recorder = useRecorder(({ blob, filename }) => chat.sendAudio(blob, filename))
const { isRecording, elapsed, error: recError } = recorder

function submit() {
  if (sending.value) return
  if (isRecording.value) {
    recorder.stop()
    return
  }
  if (!draft.value.trim()) return
  void chat.sendText(draft.value)
  draft.value = ''
}

function autoresize() {
  const el = inputEl.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${el.scrollHeight}px`
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
    submit()
  }
}

function toggleMic() {
  if (isRecording.value) recorder.stop()
  else void recorder.start()
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
          <span v-if="m.voice" class="voice-mark">голосом</span>
          <p class="text">{{ m.text }}</p>
          <span v-if="m.status === 'needs_clarification'" class="tag">ждёт уточнения</span>
          <ul v-if="m.entries?.length" class="entries-inline">
            <EntryItem v-for="e in m.entries" :key="e.id" :entry="e" compact />
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
    <div v-if="conversationId" class="banner info">
      Ассистент ждёт ответа на уточнение.
      <button type="button" class="link" @click="chat.reset()">Начать заново</button>
    </div>
    <div v-else-if="messages.length && !sending" class="toolbar">
      <button type="button" class="link muted" @click="chat.reset()">Очистить чат</button>
    </div>

    <form class="composer" @submit.prevent="submit">
      <button
        type="button"
        class="mic"
        :class="{ active: isRecording }"
        :disabled="sending || !recorder.supported"
        :aria-pressed="isRecording"
        :aria-label="isRecording ? 'Остановить запись и отправить' : 'Записать голосовое сообщение'"
        :title="isRecording ? 'Остановить и отправить' : `Записать голос (до ${MAX_RECORD_SECONDS} с)`"
        @click="toggleMic"
      >
        <svg v-if="!isRecording" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
          <path fill="currentColor" d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2Z" />
        </svg>
        <svg v-else viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" /></svg>
      </button>

      <div v-if="isRecording" class="recording">
        <div class="recording-row">
          <span class="dot" /> Говорите… {{ elapsed }} / {{ MAX_RECORD_SECONDS }} с
          <progress :value="elapsed" :max="MAX_RECORD_SECONDS" />
          <button type="button" class="cancel" title="Отменить запись (Esc)" @click="recorder.cancel()">Отмена</button>
        </div>
        <span class="hint">Договорите и нажмите «Отправить». Esc — отмена. Через {{ MAX_RECORD_SECONDS }} с запись уйдёт сама.</span>
      </div>
      <textarea
        v-else
        ref="inputEl"
        v-model="draft"
        rows="1"
        aria-label="Сообщение ассистенту"
        :placeholder="conversationId ? 'Ответьте на уточнение…' : 'Сообщение…'"
        :disabled="sending"
        @keydown="onKeydown"
      />

      <button type="submit" class="send" :class="{ rec: isRecording }" :disabled="sending || (!isRecording && !draft.trim())">
        {{ sending ? 'Отправка…' : 'Отправить' }}
      </button>
    </form>
  </section>
</template>
