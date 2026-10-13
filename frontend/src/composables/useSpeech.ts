import { computed, ref, watch } from 'vue'
import { listVoices, synthesize } from '@/api/client'
import type { VoiceInfo } from '@/api/types'

const STORAGE_KEY = 'assistant-voice'
const LEGACY_KEY = 'assistant-tts'
export const BROWSER_VOICE = 'browser'
export const VOICE_OFF = 'off'

const browserSupported = typeof window !== 'undefined' && 'speechSynthesis' in window

function readVoice(): string {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) return saved
    return localStorage.getItem(LEGACY_KEY) === '1' ? BROWSER_VOICE : VOICE_OFF
  } catch {
    return VOICE_OFF
  }
}

const voice = ref(readVoice())
const serverVoices = ref<VoiceInfo[]>([])
const serverAvailable = ref(false)
const serverLoading = ref(false)
const speaking = ref(false)
const previewing = ref<string | null>(null)
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

const enabled = computed(() => voice.value !== VOICE_OFF)
const currentName = computed(() => {
  if (voice.value === VOICE_OFF) return 'Без озвучки'
  if (voice.value === BROWSER_VOICE) return 'Браузер'
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
  const ru = window.speechSynthesis.getVoices().find((v) => v.lang.toLowerCase().startsWith('ru'))
  if (ru) utterance.voice = ru
  utterance.onend = () => (speaking.value = false)
  speaking.value = true
  window.speechSynthesis.speak(utterance)
}

async function speakWith(voiceId: string, text: string) {
  stop()
  const id = ++requestId
  if (!text) return
  if (voiceId === BROWSER_VOICE || !serverAvailable.value) {
    speakBrowser(text)
    return
  }
  speaking.value = true
  try {
    const blob = await synthesize(text, voiceId)
    if (id !== requestId) return
    audioUrl = URL.createObjectURL(blob)
    audio = new Audio(audioUrl)
    audio.onended = () => {
      speaking.value = false
      previewing.value = null
    }
    await audio.play()
  } catch {
    if (id === requestId) {
      speaking.value = false
      speakBrowser(text)
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
    enabled,
    currentName,
    serverVoices,
    serverAvailable,
    serverLoading,
    speaking,
    previewing,
    browserSupported,
    loadVoices,
    preview,
    stop,
  }
}
