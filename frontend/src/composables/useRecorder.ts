import { onBeforeUnmount, ref } from 'vue'

export const MAX_RECORD_SECONDS = 25
const STOP_AT_MS = MAX_RECORD_SECONDS * 1000 - 300
const MIN_RECORD_MS = 600

const MIME_CANDIDATES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4']

export interface Recording {
  blob: Blob
  filename: string
}

export function useRecorder(onDone: (rec: Recording) => void) {
  const isRecording = ref(false)
  const elapsed = ref(0)
  const error = ref<string | null>(null)

  let recorder: MediaRecorder | null = null
  let stream: MediaStream | null = null
  let chunks: Blob[] = []
  let timer: number | undefined
  let startedAt = 0
  let discard = false

  const supported = typeof window !== 'undefined' && 'MediaRecorder' in window && !!navigator.mediaDevices?.getUserMedia

  function cleanup() {
    window.clearInterval(timer)
    timer = undefined
    stream?.getTracks().forEach((t) => t.stop())
    stream = null
    recorder = null
    isRecording.value = false
  }

  async function start() {
    error.value = null
    if (!supported) {
      error.value = 'Браузер не поддерживает запись звука (нужен HTTPS или localhost)'
      return
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      error.value = 'Нет доступа к микрофону'
      return
    }
    const mimeType = MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m))
    const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    recorder = rec
    chunks = []
    rec.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data)
    }
    rec.onstop = () => {
      const type = rec.mimeType || mimeType || 'audio/webm'
      const blob = new Blob(chunks, { type })
      const tooShort = Date.now() - startedAt < MIN_RECORD_MS
      cleanup()
      if (discard) return
      if (tooShort || blob.size === 0) {
        error.value = 'Запись слишком короткая — скажите команду и потом нажмите «Отправить»'
        return
      }
      onDone({ blob, filename: `voice.${extFor(type)}` })
    }
    discard = false
    rec.start()
    isRecording.value = true
    startedAt = Date.now()
    elapsed.value = 0
    timer = window.setInterval(() => {
      elapsed.value = Math.min(MAX_RECORD_SECONDS, Math.floor((Date.now() - startedAt) / 1000))
      if (Date.now() - startedAt >= STOP_AT_MS) stop()
    }, 200)
  }

  function stop() {
    if (recorder && recorder.state !== 'inactive') recorder.stop()
  }

  function cancel() {
    discard = true
    stop()
  }

  onBeforeUnmount(() => {
    if (recorder) recorder.onstop = null
    stop()
    cleanup()
  })

  return { isRecording, elapsed, error, supported, start, stop, cancel }
}

function extFor(type: string): string {
  if (type.includes('ogg')) return 'ogg'
  if (type.includes('mp4')) return 'm4a'
  return 'webm'
}
