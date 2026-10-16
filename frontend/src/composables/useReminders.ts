import { ref } from 'vue'
import { listEntries } from '@/api/client'
import type { Entry } from '@/api/types'
import { addDays, isoDate } from '@/utils/time'
import { speak } from './useSpeech'

const STORAGE_KEY = 'reminders-enabled'
const FIRED_KEY = 'reminders-fired'
const CHECK_MS = 30_000
const LATE_MS = 5 * 60_000

const supported = typeof window !== 'undefined' && 'Notification' in window
const permission = ref<NotificationPermission>(supported ? Notification.permission : 'denied')
const enabled = ref(readEnabled())
const lastShown = ref<string | null>(null)
let timer: number | undefined

function readEnabled(): boolean {
  try {
    return supported && Notification.permission === 'granted' && localStorage.getItem(STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

function readFired(): Record<string, number> {
  try {
    return JSON.parse(localStorage.getItem(FIRED_KEY) ?? '{}') as Record<string, number>
  } catch {
    return {}
  }
}

function saveFired(fired: Record<string, number>) {
  const dayAgo = Date.now() - 86_400_000
  const kept = Object.fromEntries(Object.entries(fired).filter(([, at]) => at > dayAgo))
  try {
    localStorage.setItem(FIRED_KEY, JSON.stringify(kept))
  } catch {}
}

export function leadMinutes(entry: Entry): number | null {
  if (!entry.time || !entry.date) return null
  if (entry.remind_minutes != null) return entry.remind_minutes
  if (entry.kind === 'meeting') return 10
  if (entry.kind === 'reminder') return 0
  return null
}

function lead(minutes: number): string {
  if (minutes === 0) return 'сейчас'
  if (minutes % 60 === 0) return minutes === 60 ? 'через час' : `через ${minutes / 60} ч`
  return `через ${minutes} мин`
}

async function check() {
  if (!enabled.value) return
  const now = new Date()
  let items: Entry[]
  try {
    items = (await listEntries({ date_from: isoDate(now), date_to: isoDate(addDays(now, 1)) })).items
  } catch {
    return
  }
  const fired = readFired()
  for (const entry of items) {
    const minutes = leadMinutes(entry)
    if (minutes == null || entry.status === 'completed') continue
    const start = new Date(`${entry.date}T${entry.time}:00`)
    const at = start.getTime() - minutes * 60_000
    const key = `${entry.id}|${entry.date}|${entry.time}|${minutes}`
    if (fired[key] || Date.now() < at || Date.now() - at > LATE_MS) continue
    fired[key] = Date.now()
    const body = minutes === 0 ? `Сейчас, ${entry.time}` : `Начало ${lead(minutes)}, в ${entry.time}`
    try {
      new Notification(entry.title, { body, tag: key })
    } catch {}
    lastShown.value = `${entry.title}: ${body}`
    speak(minutes === 0 ? `Напоминаю: ${entry.title}.` : `Напоминаю: ${entry.title} ${lead(minutes)}.`)
  }
  saveFired(fired)
}

function startTimer() {
  window.clearInterval(timer)
  if (!enabled.value) return
  void check()
  timer = window.setInterval(check, CHECK_MS)
}

async function toggle(): Promise<string> {
  if (!supported) return 'Этот браузер не поддерживает уведомления'
  if (enabled.value) {
    enabled.value = false
    try {
      localStorage.setItem(STORAGE_KEY, '0')
    } catch {}
    startTimer()
    return 'Уведомления о напоминаниях выключены'
  }
  permission.value = await Notification.requestPermission()
  if (permission.value !== 'granted') return 'Браузер не разрешил уведомления — включите их в настройках сайта'
  enabled.value = true
  try {
    localStorage.setItem(STORAGE_KEY, '1')
  } catch {}
  startTimer()
  return 'Уведомления включены: напомню о встречах и напоминаниях, пока вкладка открыта'
}

export function useReminders() {
  return { supported, enabled, permission, lastShown, toggle, startTimer }
}
