import type {
  AssistantTurn,
  DailySummary,
  Entry,
  EntryList,
  EntryUpdate,
  ErrorResponse,
  GoogleStatus,
  HealthResponse,
  ListEntriesQuery,
  ServicesStatus,
  SessionInfo,
  TextMessageRequest,
  VoiceList,
} from './types'

const BASE_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '')

const ASSISTANT_TIMEOUT_MS = 60_000
const DEFAULT_TIMEOUT_MS = 15_000

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
  ) {
    super(message)
  }
}

type Listener = (status: number) => void
const authListeners = new Set<Listener>()

export function onAuthError(listener: Listener): () => void {
  authListeners.add(listener)
  return () => authListeners.delete(listener)
}

async function send(path: string, init: RequestInit = {}, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<Response> {
  let res: Response
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      ...init,
      credentials: 'include',
      headers: { 'X-Requested-With': 'assistant', ...(init.headers ?? {}) },
      signal: AbortSignal.timeout(timeoutMs),
    })
  } catch (e) {
    if (e instanceof DOMException && e.name === 'TimeoutError') {
      throw new ApiError('Сервер не ответил вовремя. Попробуйте ещё раз.', null)
    }
    throw new ApiError('Нет связи с сервером. Проверьте сеть и что backend запущен.', null)
  }
  if (!res.ok) {
    let detail = `Ошибка сервера (${res.status})`
    try {
      const body = (await res.json()) as Partial<ErrorResponse>
      if (typeof body.detail === 'string') detail = body.detail
    } catch {}
    if (res.status === 401) authListeners.forEach((l) => l(res.status))
    throw new ApiError(detail, res.status)
  }
  return res
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const res = await send(path, init, timeoutMs)
  return (await res.json()) as T
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export function sendTextMessage(body: TextMessageRequest): Promise<AssistantTurn> {
  return request('/v1/assistant/messages', json('POST', body), ASSISTANT_TIMEOUT_MS)
}

export function sendAudioMessage(params: {
  file: Blob
  filename: string
  timezone: string
  conversation_id?: string
}): Promise<AssistantTurn> {
  const form = new FormData()
  form.append('file', params.file, params.filename)
  form.append('timezone', params.timezone)
  if (params.conversation_id) form.append('conversation_id', params.conversation_id)
  return request('/v1/assistant/audio', { method: 'POST', body: form }, ASSISTANT_TIMEOUT_MS)
}

export function listEntries(query: ListEntriesQuery): Promise<EntryList> {
  const qs = new URLSearchParams()
  for (const [k, v] of Object.entries(query)) if (v != null) qs.set(k, String(v))
  return request(`/v1/entries?${qs}`)
}

export function listUndatedEntries(): Promise<EntryList> {
  return request('/v1/entries/undated')
}

export function updateEntry(id: string, changes: EntryUpdate): Promise<Entry> {
  return request(`/v1/entries/${id}`, json('PATCH', changes))
}

export async function deleteEntry(id: string): Promise<void> {
  await send(`/v1/entries/${id}`, { method: 'DELETE' })
}

export function getSession(): Promise<SessionInfo> {
  return request('/v1/session')
}

export async function unlockSession(code: string): Promise<void> {
  await send('/v1/session/unlock', json('POST', { code }))
}

export function getCalendarLink(): Promise<{ url: string }> {
  return request('/v1/calendar')
}

export async function downloadCalendar(): Promise<Blob> {
  const { url } = await getCalendarLink()
  const res = await send(new URL(url).pathname)
  return res.blob()
}

export function listVoices(): Promise<VoiceList> {
  return request('/v1/speech/voices', {}, 5_000)
}

export async function synthesize(text: string, voice: string): Promise<Blob> {
  const res = await send('/v1/speech/synthesize', json('POST', { text, voice }), 90_000)
  return res.blob()
}

export function getDailySummary(timezone: string): Promise<DailySummary> {
  return request(`/v1/assistant/summary?${new URLSearchParams({ timezone })}`)
}

export function getServicesStatus(): Promise<ServicesStatus> {
  return request('/v1/status', {}, 5_000)
}

export function rotateCalendarLink(): Promise<{ url: string }> {
  return request('/v1/calendar/rotate', { method: 'POST' })
}

export function getGoogleStatus(): Promise<GoogleStatus> {
  return request('/v1/google/status')
}

export function googleConnectUrl(): string {
  return `${BASE_URL}/v1/google/connect?${new URLSearchParams({ origin: window.location.origin })}`
}

export async function disconnectGoogle(): Promise<void> {
  await send('/v1/google/disconnect', { method: 'POST' })
}

export function getHealth(): Promise<HealthResponse> {
  return request('/health', {}, 5_000)
}
