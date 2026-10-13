import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import { sendAudioMessage, sendTextMessage } from '@/api/client'
import type { AssistantTurn, Entry } from '@/api/types'
import { timezone } from '@/utils/time'
import { useEntriesStore } from './entries'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'error'
  text: string
  status?: AssistantTurn['status']
  entries?: Entry[]
  pending?: boolean
  voice?: boolean
}

const STORAGE_KEY = 'assistant-chat-v1'
const MAX_STORED = 100

interface Persisted {
  messages: ChatMessage[]
  conversationId: string | null
}

function load(): Persisted {
  try {
    const raw = localStorage.getItem(STORAGE_KEY) ?? sessionStorage.getItem(STORAGE_KEY)
    if (raw) {
      const p = JSON.parse(raw) as Persisted
      if (Array.isArray(p.messages)) return { messages: p.messages.map((m) => ({ ...m, pending: false })), conversationId: p.conversationId ?? null }
    }
  } catch {}
  return { messages: [], conversationId: null }
}

let seq = 0
const localId = () => `local-${Date.now()}-${++seq}`

export const useChatStore = defineStore('chat', () => {
  const initial = load()
  const messages = ref<ChatMessage[]>(initial.messages)
  const sending = ref(false)
  const conversationId = ref<string | null>(initial.conversationId)
  const awaitingClarification = computed(() => {
    const lastAssistant = [...messages.value].reverse().find((message) => message.role === 'assistant')
    return lastAssistant?.status === 'needs_clarification'
  })
  const retryLast = ref<(() => void) | null>(null)

  watch(
    [messages, conversationId],
    () => {
      try {
        const data: Persisted = { messages: messages.value.slice(-MAX_STORED), conversationId: conversationId.value }
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
    const requestConversationId = originalConversationId === undefined
      ? conversationId.value : originalConversationId
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
      messages.value.push({
        id: turn.turn_id,
        role: 'assistant',
        text: turn.assistant_text,
        status: turn.status,
        entries: turn.entries,
      })
    } catch (e) {
      const m = messages.value.find((x) => x.id === userId)
      if (m) {
        m.pending = false
        if (m.voice) m.text = 'Голосовое сообщение (не отправлено)'
      }
      messages.value.push({ id: localId(), role: 'error', text: e instanceof Error ? e.message : 'Неизвестная ошибка' })
      retryLast.value = () => { void run(user, call, userId, requestConversationId) }
    } finally {
      sending.value = false
      void useEntriesStore().refresh()
    }
  }

  function sendText(text: string) {
    const t = text.trim()
    if (!t || sending.value) return
    const request_id = crypto.randomUUID()
    return run({ text: t }, (conversation_id) => sendTextMessage({
      text: t, timezone, conversation_id, request_id,
    }))
  }

  function sendAudio(file: Blob, filename: string) {
    if (sending.value) return
    const request_id = crypto.randomUUID()
    return run(
      { text: 'Распознаю голосовое…', voice: true },
      (conversation_id) => sendAudioMessage({ file, filename, timezone, conversation_id, request_id }),
    )
  }

  function retry() {
    const fn = retryLast.value
    retryLast.value = null
    fn?.()
  }

  function reset() {
    messages.value = []
    conversationId.value = null
    retryLast.value = null
  }

  return { messages, sending, conversationId, awaitingClarification, retryLast, sendText, sendAudio, retry, reset }
})
