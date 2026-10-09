import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import { sendAudioMessage, sendTextMessage } from '@/api/client'
import type { AssistantTurn, Entry } from '@/api/types'
import { speak } from '@/composables/useSpeech'
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

const STORAGE_KEY = 'assistant-chat-v2'
const MAX_STORED = 100

interface Persisted {
  messages: ChatMessage[]
  conversationId: string | null
  awaiting: boolean
}

function load(): Persisted {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
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
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(data))
      } catch {}
    },
    { deep: true },
  )

  async function run(
    user: Pick<ChatMessage, 'text' | 'voice'>,
    call: (conversation_id?: string) => Promise<AssistantTurn>,
    retry: () => void,
  ) {
    const userId = localId()
    messages.value.push({ id: userId, role: 'user', ...user, pending: !!user.voice })
    sending.value = true
    retryLast.value = null
    try {
      const turn = await call(conversationId.value ?? undefined)
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
      })
    } catch (e) {
      const m = messages.value.find((x) => x.id === userId)
      if (m) {
        m.pending = false
        if (m.voice) m.text = 'Голосовое сообщение (не отправлено)'
      }
      messages.value.push({ id: localId(), role: 'error', text: e instanceof Error ? e.message : 'Неизвестная ошибка' })
      retryLast.value = retry
    } finally {
      sending.value = false
      void useEntriesStore().refresh()
    }
  }

  function sendText(text: string) {
    const t = text.trim()
    if (!t || sending.value) return
    return run({ text: t }, (conversation_id) => sendTextMessage({ text: t, timezone, conversation_id }), () => sendText(t))
  }

  function sendAudio(file: Blob, filename: string) {
    if (sending.value) return
    return run(
      { text: 'Распознаю голосовое…', voice: true },
      (conversation_id) => sendAudioMessage({ file, filename, timezone, conversation_id }),
      () => sendAudio(file, filename),
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
    awaiting.value = false
    retryLast.value = null
  }

  return { messages, sending, conversationId, awaiting, retryLast, sendText, sendAudio, retry, reset }
})
