import type { AssistantTurn, EntryList, ErrorResponse, HealthResponse, ListEntriesQuery, TextMessageRequest } from './types'

const BASE_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '')

const ASSISTANT_TIMEOUT_MS = 120_000
const DEFAULT_TIMEOUT_MS = 15_000

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
  ) {
    super(message)
  }
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE_URL}${path}`, { ...init, signal: AbortSignal.timeout(timeoutMs) })
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
    throw new ApiError(detail, res.status)
  }
  return (await res.json()) as T
}

export function sendTextMessage(body: TextMessageRequest): Promise<AssistantTurn> {
  return request(
    '/v1/assistant/messages',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    ASSISTANT_TIMEOUT_MS,
  )
}

export function sendAudioMessage(params: {
  file: Blob
  filename: string
  timezone: string
  conversation_id?: string
  request_id?: string
}): Promise<AssistantTurn> {
  const form = new FormData()
  form.append('file', params.file, params.filename)
  form.append('timezone', params.timezone)
  if (params.conversation_id) form.append('conversation_id', params.conversation_id)
  if (params.request_id) form.append('request_id', params.request_id)
  return request('/v1/assistant/audio', { method: 'POST', body: form }, ASSISTANT_TIMEOUT_MS)
}

export function listEntries(query: ListEntriesQuery): Promise<EntryList> {
  const qs = new URLSearchParams()
  for (const [k, v] of Object.entries(query)) if (v != null) qs.set(k, String(v))
  return request(`/v1/entries?${qs}`)
}

export function getHealth(): Promise<HealthResponse> {
  return request('/health', {}, 5_000)
}
