import { ref, watch } from 'vue'
import { stopSpeech, useSpeech } from './useSpeech'

const STORAGE_KEY = 'barge-in'
const THRESHOLD = 0.06
const HOLD_MS = 250

function readEnabled(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

const enabled = ref(readEnabled())
const listeners = new Set<() => void>()
let stream: MediaStream | null = null
let context: AudioContext | null = null
let frame = 0

watch(enabled, (v) => {
  try {
    localStorage.setItem(STORAGE_KEY, v ? '1' : '0')
  } catch {}
  if (!v) stopListening()
})

function stopListening() {
  cancelAnimationFrame(frame)
  stream?.getTracks().forEach((t) => t.stop())
  void context?.close().catch(() => undefined)
  stream = null
  context = null
}

async function startListening() {
  if (stream || !navigator.mediaDevices?.getUserMedia) return
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    })
  } catch {
    enabled.value = false
    return
  }
  context = new AudioContext()
  const analyser = context.createAnalyser()
  analyser.fftSize = 1024
  context.createMediaStreamSource(stream).connect(analyser)
  const samples = new Float32Array(analyser.fftSize)
  let loudSince = 0
  const tick = () => {
    analyser.getFloatTimeDomainData(samples)
    let sum = 0
    for (const s of samples) sum += s * s
    const level = Math.sqrt(sum / samples.length)
    const now = performance.now()
    if (level > THRESHOLD) {
      loudSince = loudSince || now
      if (now - loudSince >= HOLD_MS) {
        stopListening()
        stopSpeech()
        listeners.forEach((l) => l())
        return
      }
    } else {
      loudSince = 0
    }
    frame = requestAnimationFrame(tick)
  }
  frame = requestAnimationFrame(tick)
}

const { speaking } = useSpeech()
watch(speaking, (isSpeaking) => {
  if (isSpeaking && enabled.value) void startListening()
  else stopListening()
})

export function useBargeIn() {
  function onBargeIn(listener: () => void) {
    listeners.add(listener)
    return () => listeners.delete(listener)
  }
  return { enabled, onBargeIn }
}
