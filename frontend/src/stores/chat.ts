import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import { getDailySummary, sendAudioMessage, sendTextMessage } from '@/api/client'
import type { AssistantTurn, Entry } from '@/api/types'
import { speak, stopSpeech } from '@/composables/useSpeech'
import { isoDate, timezone } from '@/utils/time'
import { useEntriesStore } from './entries'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'error'
  text: string
  status?: AssistantTurn['status']
  entries?: Entry[]
  pending?: boolean
  voice?: boolean
  suggestions?: string[]
}

const SUMMARY_KEY = 'assistant-summary-day'

const STORAGE_KEY = 'assistant-chat-v2'
const MAX_STORED = 100

interface Persisted {
  messages: ChatMessage[]
  conversationId: string | null
  awaiting: boolean
}

function load(): Persisted {
  try {
    const raw = localStorage.getItem(STORAGE_KEY) ?? sessionStorage.getItem(STORAGE_KEY)
    if (raw) {
      const p = JSON.parse(raw) as Persisted
      if (Array.isArray(p.messages)) {
        return {
          messages: p.messages.map((m) => ({ ...m, pending: false })),
          conversationId: p.conversationId ?? null,
          awaiting: !!p.awaiting,
        }
      }
    }
  } catch {}
  return { messages: [], conversationId: null, awaiting: false }
}

let seq = 0
const localId = () => `local-${Date.now()}-${++seq}`

export const useChatStore = defineStore('chat', () => {
  const initial = load()
  const messages = ref<ChatMessage[]>(initial.messages)
  const sending = ref(false)
  const conversationId = ref<string | null>(initial.conversationId)
  const awaiting = ref(initial.awaiting)
  const retryLast = ref<(() => void) | null>(null)

  watch(
    [messages, conversationId, awaiting],
    () => {
      try {
        const data: Persisted = {
          messages: messages.value.slice(-MAX_STORED),
          conversationId: conversationId.value,
          awaiting: awaiting.value,
        }
        localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
      } catch {}
    },
    { deep: true },
  )

  async function run(
    user: Pick<ChatMessage, 'text' | 'voice'>,
    call: (conversation_id?: string) => Promise<AssistantTurn>,
    previousUserId?: string,
    originalConversationId?: string | null,
  ) {
    const userId = previousUserId ?? localId()
    const requestConversationId = originalConversationId === undefined ? conversationId.value : originalConversationId
    stopSpeech()
    if (!previousUserId) messages.value.push({ id: userId, role: 'user', ...user, pending: !!user.voice })
    else {
      messages.value = messages.value.filter((message) => message.role !== 'error')
      const existing = messages.value.find((message) => message.id === userId)
      if (existing) existing.pending = !!user.voice
    }
    sending.value = true
    retryLast.value = null
    try {
      const turn = await call(requestConversationId ?? undefined)
      const m = messages.value.find((x) => x.id === userId)
      if (m) {
        if (turn.user_text) m.text = turn.user_text
        else if (m.voice) m.text = 'Речь не распознана'
        m.pending = false
      }
      conversationId.value = turn.conversation_id
      awaiting.value = turn.status === 'needs_clarification'
      speak(turn.assistant_text)
      messages.value.push({
        id: turn.turn_id,
        role: 'assistant',
        text: turn.assistant_text,
        status: turn.status,
        entries: turn.entries,
        suggestions: turn.suggestions ?? [],
      })
    } catch (e) {
      const m = messages.value.find((x) => x.id === userId)
      if (m) {
        m.pending = false
        if (m.voice) m.text = 'Голосовое сообщение (не отправлено)'
      }
      messages.value.push({ id: localId(), role: 'error', text: e instanceof Error ? e.message : 'Неизвестная ошибка' })
      retryLast.value = () => {
        void run(user, call, userId, requestConversationId)
      }
    } finally {
      sending.value = false
      void useEntriesStore().refresh(true)
    }
  }

  function sendText(text: string) {
    const t = text.trim()
    if (!t || sending.value) return
    const request_id = crypto.randomUUID()
    return run({ text: t }, (conversation_id) => sendTextMessage({ text: t, timezone, conversation_id, request_id }))
  }

  function sendAudio(file: Blob, filename: string) {
    if (sending.value) return
    const request_id = crypto.randomUUID()
    return run(
      { text: 'Распознаю голосовое…', voice: true },
      (conversation_id) => sendAudioMessage({ file, filename, timezone, conversation_id, request_id }),
    )
  }

  async function showDailySummary() {
    const day = isoDate(new Date())
    try {
      if (localStorage.getItem(SUMMARY_KEY) === day) return
    } catch {}
    try {
      const summary = await getDailySummary(timezone)
      messages.value.push({
        id: localId(),
        role: 'assistant',
        text: summary.assistant_text,
        status: 'completed',
        entries: summary.entries,
      })
      window.setTimeout(() => speak(summary.assistant_text), 1500)
      try {
        localStorage.setItem(SUMMARY_KEY, day)
      } catch {}
    } catch {}
  }

  function retry() {
    const fn = retryLast.value
    retryLast.value = null
    fn?.()
  }

  function reset() {
    messages.value = []
    conversationId.value = null
    awaiting.value = false
    retryLast.value = null
  }

  return { messages, sending, conversationId, awaiting, retryLast, sendText, sendAudio, retry, reset, showDailySummary }
})
