import { ref, watch } from 'vue'

const STORAGE_KEY = 'assistant-tts'
const supported = typeof window !== 'undefined' && 'speechSynthesis' in window

function readFlag(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

const enabled = ref(supported && readFlag())

watch(enabled, (v) => {
  try {
    localStorage.setItem(STORAGE_KEY, v ? '1' : '0')
  } catch {}
  if (!v && supported) window.speechSynthesis.cancel()
})

function pickVoice(): SpeechSynthesisVoice | undefined {
  return window.speechSynthesis.getVoices().find((v) => v.lang.toLowerCase().startsWith('ru'))
}

export function speak(text: string) {
  if (!supported || !enabled.value || !text) return
  const synth = window.speechSynthesis
  synth.cancel()
  const utterance = new SpeechSynthesisUtterance(text.replace(/•/g, '').replace(/\n+/g, '. '))
  utterance.lang = 'ru-RU'
  const voice = pickVoice()
  if (voice) utterance.voice = voice
  synth.speak(utterance)
}

export function useSpeech() {
  return { supported, enabled }
}
