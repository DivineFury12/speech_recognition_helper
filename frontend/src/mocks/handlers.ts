import { delay, http, HttpResponse } from 'msw'
import type { AssistantTurn, Entry, EntryList, ErrorResponse, HealthResponse, Schemas } from '@/api/types'
import { addDays, isoDate } from '@/utils/time'

const API = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '')

type Kind = Schemas['EntryKind']

const tzDefault = Intl.DateTimeFormat().resolvedOptions().timeZone
const today = new Date()

const entries: Entry[] = [
  mkEntry('meeting', 'Планёрка с командой', isoDate(today), '11:00', 30),
  mkEntry('task', 'Подготовить презентацию', isoDate(addDays(today, 2)), null),
  mkEntry('reminder', 'Оплатить интернет', isoDate(addDays(today, 4)), '09:00'),
]

const drafts = new Map<string, { kind: Kind; title: string; date: string; tz: string; duration: number | null }>()

function mkEntry(kind: Kind, title: string, date: string | null, time: string | null, duration: number | null = null, tz = tzDefault): Entry {
  return {
    id: crypto.randomUUID(),
    kind,
    title,
    date,
    time,
    timezone: tz,
    duration_minutes: kind === 'meeting' ? (duration ?? 60) : null,
    status: 'pending',
    created_at: new Date().toISOString(),
  }
}

function parseTime(text: string): string | null {
  const m = text.match(/(?:^|\D)([01]?\d|2[0-3])(?::|\.|\s?ч\s?)?([0-5]\d)?(?:\s*(утра|дня|вечера))?(?=\D|$)/i)
  if (!m || !/(\bв\s*\d|\d:\d|утра|дня|вечера|^\s*\d)/i.test(text)) return null
  let h = Number(m[1])
  if (/вечера|дня/i.test(m[3] ?? '') && h < 12) h += 12
  return `${String(h).padStart(2, '0')}:${m[2] ?? '00'}`
}

function parseDate(text: string, tz: string): string {
  const now = new Date()
  if (/послезавтра/i.test(text)) return isoDate(addDays(now, 2), tz)
  if (/завтра/i.test(text)) return isoDate(addDays(now, 1), tz)
  return isoDate(now, tz)
}

function parseDuration(text: string): number | null {
  if (/на\s+полчаса/i.test(text)) return 30
  const match = text.match(/на\s+(\d+)\s*(минут(?:у|ы)?|час(?:а|ов)?)/i)
  if (!match) return null
  return Number(match[1]) * (match[2].startsWith('час') ? 60 : 1)
}

function cleanTitle(text: string): string {
  const t = text
    .replace(/^(напомни( мне)?|поставь|создай|добавь|запланируй)\s*/i, '')
    .replace(/(встречу|задачу|напоминание)\s*/i, '')
    .replace(/(послезавтра|завтра|сегодня)/gi, '')
    .replace(/в\s*\d{1,2}([:.]\d{2})?(\s*(утра|дня|вечера))?/gi, '')
    .replace(/на\s+(?:\d+\s*(?:минут(?:у|ы)?|час(?:а|ов)?)|полчаса)/gi, '')
    .replace(/\s+/g, ' ')
    .trim()
  return t ? t[0].toUpperCase() + t.slice(1) : 'Без названия'
}

function turn(p: Omit<AssistantTurn, 'turn_id'>): AssistantTurn {
  return { ...p, turn_id: crypto.randomUUID() }
}

function describe(e: Entry): string {
  return [e.title, e.date, e.time].filter(Boolean).join(', ')
}

function handle(text: string, tz: string, conversationId: string | undefined): AssistantTurn {
  const cid = conversationId ?? crypto.randomUUID()
  const base = { conversation_id: cid, user_text: text }

  const draft = conversationId ? drafts.get(conversationId) : undefined
  if (draft) {
    const time = parseTime(text)
    if (!time) {
      return turn({ ...base, assistant_text: 'Не понял время. Назовите, например, «в 15:00».', status: 'needs_clarification', intent: 'create_entry', entries: [] })
    }
    drafts.delete(cid)
    const e = mkEntry(draft.kind, draft.title, draft.date, time, draft.duration, draft.tz)
    entries.push(e)
    return turn({ ...base, assistant_text: `Готово: ${describe(e)}.`, status: 'completed', intent: 'create_entry', entries: [e] })
  }

  if (/что у меня|какие (дела|задачи|встречи)|покажи|список/i.test(text)) {
    const from = isoDate(new Date(), tz)
    const to = /недел/i.test(text) ? isoDate(addDays(new Date(), 6), tz) : /завтра/i.test(text) ? isoDate(addDays(new Date(), 1), tz) : from
    const found = entries.filter((e) => e.date && e.date >= from && e.date <= to)
    return turn({
      ...base,
      assistant_text: found.length ? `Нашёл ${found.length}:\n${found.map((e) => '• ' + describe(e)).join('\n')}` : 'На этот период ничего нет.',
      status: 'completed',
      intent: 'list_entries',
      entries: found,
    })
  }

  if (/найди|поищи|где/i.test(text)) {
    const words = text.toLowerCase().replace(/найди|поищи|где|задачу|встречу|про/gi, '').split(/\s+/).filter((w) => w.length > 3)
    const found = entries.filter((e) => words.some((w) => e.title.toLowerCase().includes(w.slice(0, -2))))
    return turn({
      ...base,
      assistant_text: found.length ? `Нашёл: ${found.map(describe).join('; ')}` : 'Ничего похожего не нашёл.',
      status: 'completed',
      intent: 'search_entries',
      entries: found,
    })
  }

  if (/напомни|поставь|создай|добавь|запланируй|встреч|задач/i.test(text)) {
    const kind: Kind = /встреч/i.test(text) ? 'meeting' : /напомни|напоминан/i.test(text) ? 'reminder' : 'task'
    const title = cleanTitle(text)
    const date = parseDate(text, tz)
    const time = parseTime(text)
    const duration = kind === 'meeting' ? parseDuration(text) : null
    if (kind !== 'task' && !time) {
      drafts.set(cid, { kind, title, date, tz, duration })
      return turn({ ...base, assistant_text: kind === 'meeting' ? 'На какое время поставить встречу?' : 'Во сколько напомнить?', status: 'needs_clarification', intent: 'create_entry', entries: [] })
    }
    const e = mkEntry(kind, title, date, time, duration, tz)
    entries.push(e)
    return turn({ ...base, assistant_text: `Создал: ${describe(e)}.`, status: 'completed', intent: 'create_entry', entries: [e] })
  }

  return turn({ ...base, assistant_text: 'Я умею создавать задачи, напоминания и встречи, а также показывать и искать их. <b>HTML</b> выводится как текст.', status: 'no_action', intent: 'unknown', entries: [] })
}

function err(status: number, detail: string) {
  return HttpResponse.json<ErrorResponse>({ detail }, { status })
}

export const handlers = [
  http.get(`${API}/health`, () => HttpResponse.json<HealthResponse>({ status: 'ok', model: 'mock', device: 'browser' })),

  http.post<never, never, AssistantTurn | ErrorResponse>(`${API}/v1/assistant/messages`, async ({ request }) => {
    await delay(500)
    const body = (await request.json()) as { text?: string; timezone?: string; conversation_id?: string }
    if (!body.text || !body.timezone) return err(422, 'Поля text и timezone обязательны')
    if (/ошибка/i.test(body.text)) return err(500, 'Внутренняя ошибка (мок)')
    if (/\b422\b/.test(body.text)) return err(422, 'Некорректный запрос (мок)')
    return HttpResponse.json<AssistantTurn>(handle(body.text, body.timezone, body.conversation_id))
  }),

  http.post<never, never, AssistantTurn | ErrorResponse>(`${API}/v1/assistant/audio`, async ({ request }) => {
    await delay(900)
    const form = await request.formData()
    const file = form.get('file')
    const tz = form.get('timezone')
    if (!(file instanceof Blob) || file.size === 0) return err(400, 'Пустой аудиофайл')
    if (typeof tz !== 'string' || !tz) return err(422, 'Поле timezone обязательно')
    const cid = form.get('conversation_id')
    if (file.size < 3000) {
      return HttpResponse.json<AssistantTurn>(
        turn({ conversation_id: typeof cid === 'string' ? cid : crypto.randomUUID(), user_text: '', assistant_text: 'Не расслышал, повторите, пожалуйста.', status: 'no_action', intent: 'unknown', entries: [] }),
      )
    }
    const transcript = typeof cid === 'string' && drafts.has(cid) ? 'В 15:00' : 'Поставь встречу с Иваном завтра'
    return HttpResponse.json<AssistantTurn>(handle(transcript, tz, typeof cid === 'string' ? cid : undefined))
  }),

  http.get<never, never, EntryList | ErrorResponse>(`${API}/v1/entries`, async ({ request }) => {
    await delay(250)
    const url = new URL(request.url)
    const from = url.searchParams.get('date_from')
    const to = url.searchParams.get('date_to')
    if (!from || !to) return err(422, 'date_from и date_to обязательны')
    const kind = url.searchParams.get('kind')
    const status = url.searchParams.get('status')
    const items = entries.filter(
      (e) => e.date && e.date >= from && e.date <= to && (!kind || e.kind === kind) && (!status || e.status === status),
    )
    return HttpResponse.json<EntryList>({ items })
  }),
]
