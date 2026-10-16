import { computed, ref, watch } from 'vue'
import { ApiError, listVoices, synthesize } from '@/api/client'
import type { VoiceInfo } from '@/api/types'

const STORAGE_KEY = 'assistant-voice'
const LEGACY_KEY = 'assistant-tts'
export const BROWSER_VOICE = 'browser'
export const VOICE_OFF = 'off'

const browserSupported = typeof window !== 'undefined' && 'speechSynthesis' in window

function readVoice(): string {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'alina' || saved === 'evgeny') return 'ksenia'
    if (saved) return saved
    return localStorage.getItem(LEGACY_KEY) === '1' ? BROWSER_VOICE : VOICE_OFF
  } catch {
    return VOICE_OFF
  }
}

function readVolume(): number {
  try {
    const saved = Number(localStorage.getItem('speech-volume'))
    return localStorage.getItem('speech-volume') !== null && saved >= 0 && saved <= 1 ? saved : 1
  } catch {
    return 1
  }
}

const voice = ref(readVoice())
const volume = ref(readVolume())
const serverVoices = ref<VoiceInfo[]>([])
const serverAvailable = ref(false)
const serverLoading = ref(false)
const speaking = ref(false)
const previewing = ref<string | null>(null)
const notice = ref<string | null>(null)
let audio: HTMLAudioElement | null = null
let audioUrl: string | null = null
let pollTimer: number | undefined
let requestId = 0

watch(voice, (v) => {
  try {
    localStorage.setItem(STORAGE_KEY, v)
  } catch {}
  if (v === VOICE_OFF) stop()
})

watch(volume, (v) => {
  try {
    localStorage.setItem('speech-volume', String(v))
  } catch {}
  if (audio) audio.volume = v
})

const enabled = computed(() => voice.value !== VOICE_OFF)
const currentName = computed(() => {
  if (voice.value === VOICE_OFF) return 'Без озвучки'
  if (voice.value === BROWSER_VOICE) return 'Робот'
  return serverVoices.value.find((v) => v.id === voice.value)?.name ?? 'Голос'
})

async function loadVoices() {
  window.clearTimeout(pollTimer)
  try {
    const list = await listVoices()
    serverVoices.value = list.voices
    serverAvailable.value = list.available
    serverLoading.value = list.loading
    if (list.loading) pollTimer = window.setTimeout(loadVoices, 5000)
  } catch {
    serverAvailable.value = false
    serverLoading.value = false
  }
}

function cleanText(text: string): string {
  return text.replace(/[•✓«»]/g, '').replace(/\n+/g, '. ')
}

function stop() {
  requestId++
  if (browserSupported) window.speechSynthesis.cancel()
  if (audio) {
    audio.pause()
    audio = null
  }
  if (audioUrl) {
    URL.revokeObjectURL(audioUrl)
    audioUrl = null
  }
  speaking.value = false
  previewing.value = null
}

function speakBrowser(text: string) {
  if (!browserSupported) return
  const utterance = new SpeechSynthesisUtterance(cleanText(text))
  utterance.lang = 'ru-RU'
  utterance.volume = volume.value
  const ru = window.speechSynthesis.getVoices().find((v) => v.lang.toLowerCase().startsWith('ru'))
  if (ru) utterance.voice = ru
  utterance.onend = () => (speaking.value = false)
  speaking.value = true
  window.speechSynthesis.speak(utterance)
}

function splitText(text: string, limit = 220): string[] {
  const sentences = text
    .split(/\n+|(?<=[.!?…])\s+/)
    .map((x) => x.trim())
    .filter(Boolean)
  const parts: string[] = []
  for (const sentence of sentences) {
    const last = parts[parts.length - 1]
    if (last && last.length + 1 + sentence.length <= limit) parts[parts.length - 1] = `${last} ${sentence}`
    else parts.push(sentence)
  }
  return parts
}

function playBlob(blob: Blob, id: number): Promise<void> {
  return new Promise((resolve, reject) => {
    if (id !== requestId) return resolve()
    audioUrl = URL.createObjectURL(blob)
    audio = new Audio(audioUrl)
    audio.volume = volume.value
    const done = () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl)
      audioUrl = null
      resolve()
    }
    audio.onended = done
    audio.onerror = () => reject(new Error('Не удалось воспроизвести звук'))
    audio.play().catch(reject)
  })
}

function voiceName(voiceId: string): string {
  return serverVoices.value.find((v) => v.id === voiceId)?.name ?? voiceId
}

async function speakWith(voiceId: string, text: string) {
  stop()
  const id = ++requestId
  if (!text) return
  if (voiceId === BROWSER_VOICE) {
    speakBrowser(text)
    return
  }
  if (!serverAvailable.value) {
    notice.value = serverLoading.value
      ? `Голос «${voiceName(voiceId)}» ещё загружается — этот ответ прочитал голос браузера`
      : `Голос «${voiceName(voiceId)}» недоступен на сервере — ответ прочитал голос браузера`
    void loadVoices()
    speakBrowser(text)
    return
  }
  speaking.value = true
  const parts = splitText(text)
  try {
    let next: Promise<Blob> | null = synthesize(parts[0], voiceId)
    for (let i = 0; i < parts.length && next; i++) {
      const blob: Blob = await next
      if (id !== requestId) return
      next = i + 1 < parts.length ? synthesize(parts[i + 1], voiceId) : null
      await playBlob(blob, id)
      if (id !== requestId) return
    }
  } catch (e) {
    if (id !== requestId) return
    if (e instanceof ApiError && e.status === 503) {
      serverAvailable.value = false
      void loadVoices()
    }
    notice.value =
      e instanceof DOMException && e.name === 'NotAllowedError'
        ? 'Браузер заблокировал звук — кликните по странице, и озвучка заработает'
        : `Не удалось озвучить ответ голосом «${voiceName(voiceId)}»`
  } finally {
    if (id === requestId) {
      speaking.value = false
      previewing.value = null
    }
  }
}

export function speak(text: string) {
  if (voice.value === VOICE_OFF) return
  void speakWith(voice.value, text)
}

export function stopSpeech() {
  stop()
}

function preview(voiceId: string) {
  const name = voiceId === BROWSER_VOICE ? 'голос браузера' : serverVoices.value.find((v) => v.id === voiceId)?.name
  void speakWith(voiceId, `Привет! Я ${name}. Буду читать вам ответы ассистента.`)
  previewing.value = voiceId
}

export function useSpeech() {
  return {
    voice,
    volume,
    enabled,
    currentName,
    serverVoices,
    serverAvailable,
    serverLoading,
    speaking,
    previewing,
    browserSupported,
    notice,
    loadVoices,
    preview,
    stop,
  }
}
