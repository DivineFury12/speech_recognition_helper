import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { createEntry, deleteEntry, listEntries, listUndatedEntries, updateEntry } from '@/api/client'
import type { Entry, EntryCreate, EntryUpdate } from '@/api/types'
import { addDays, isoDate } from '@/utils/time'

export type KindFilter = 'all' | Entry['kind']
export type Period = 'today' | 'week' | 'month' | 'undated'

export interface AgendaSection {
  id: string
  title: string
  hint?: string
  items: Entry[]
}

const WEEKDAY = new Intl.DateTimeFormat('ru-RU', { weekday: 'long' })
const DAY_MONTH = new Intl.DateTimeFormat('ru-RU', { day: 'numeric', month: 'long' })

function dayTitle(date: string, today: string, tomorrow: string): { title: string; hint: string } {
  const [y, m, d] = date.split('-').map(Number)
  const day = new Date(y, m - 1, d)
  const hint = DAY_MONTH.format(day)
  if (date === today) return { title: 'Сегодня', hint }
  if (date === tomorrow) return { title: 'Завтра', hint }
  const weekday = WEEKDAY.format(day)
  return { title: weekday[0].toUpperCase() + weekday.slice(1), hint }
}

function byTime(a: Entry, b: Entry): number {
  const high = Number(b.priority === 'high') - Number(a.priority === 'high')
  if ((a.time ?? '') === (b.time ?? '')) return high
  if (!a.time) return 1
  if (!b.time) return -1
  return a.time < b.time ? -1 : 1
}
export const useEntriesStore = defineStore('entries', () => {
  const period = ref<Period>('week')
  const upcoming = ref<Entry[]>([])
  const overdue = ref<Entry[]>([])
  const undated = ref<Entry[]>([])
  const loading = ref(false)
  const loaded = ref(false)
  const error = ref<string | null>(null)
  const kind = ref<KindFilter>('all')
  const showDone = ref(false)
  let reqId = 0

  async function refresh(silent = false) {
    const id = ++reqId
    if (!silent) loading.value = true
    error.value = null
    const now = new Date()
    const today = isoDate(now)
    const selectedPeriod = period.value
    try {
      const [ahead, past, noDate] = await Promise.all([
        selectedPeriod === 'undated' ? Promise.resolve({ items: [] as Entry[] }) : listEntries({
          date_from: selectedPeriod === 'month' ? isoDate(new Date(now.getFullYear(), now.getMonth(), 1)) : today,
          date_to: selectedPeriod === 'today' ? today
            : selectedPeriod === 'month' ? isoDate(new Date(now.getFullYear(), now.getMonth() + 1, 0))
              : isoDate(addDays(now, 6)),
        }),
        selectedPeriod === 'today' || selectedPeriod === 'week'
          ? listEntries({ date_from: isoDate(addDays(now, -30)), date_to: isoDate(addDays(now, -1)), status: 'pending' })
          : Promise.resolve({ items: [] as Entry[] }),
        selectedPeriod === 'week' || selectedPeriod === 'undated'
          ? listUndatedEntries()
          : Promise.resolve({ items: [] as Entry[] }),
      ])
      if (id !== reqId) return
      upcoming.value = ahead.items
      overdue.value = past.items.filter((e) => !e.recurrence)
      undated.value = noDate.items
      loaded.value = true
    } catch (e) {
      if (id === reqId) error.value = e instanceof Error ? e.message : 'Не удалось загрузить записи'
    } finally {
      if (id === reqId) loading.value = false
    }
  }

  const visible = (e: Entry) =>
    (kind.value === 'all' || e.kind === kind.value) && (showDone.value || e.status !== 'completed')

  const sections = computed<AgendaSection[]>(() => {
    const now = new Date()
    const today = isoDate(now)
    const tomorrow = isoDate(addDays(now, 1))
    const result: AgendaSection[] = []
    const late = overdue.value.filter(visible).sort((a, b) => (a.date ?? '').localeCompare(b.date ?? ''))
    if (late.length) result.push({ id: 'overdue', title: 'Просрочено', items: late })
    const byDay = new Map<string, Entry[]>()
    for (const e of upcoming.value.filter(visible)) {
      if (!e.date) continue
      byDay.set(e.date, [...(byDay.get(e.date) ?? []), e])
    }
    if ((period.value === 'today' || period.value === 'week') && !byDay.has(today)) byDay.set(today, [])
    for (const date of [...byDay.keys()].sort()) {
      const { title, hint } = dayTitle(date, today, tomorrow)
      result.push({ id: date, title, hint, items: byDay.get(date)!.sort(byTime) })
    }
    const free = undated.value.filter(visible)
    if (free.length || period.value === 'undated') result.push({ id: 'undated', title: 'Без срока', items: free.sort(byTime) })
    return result
  })

  const counts = computed(() => {
    const all = [...upcoming.value, ...undated.value].filter((e) => e.status !== 'completed')
    return { total: all.length, overdue: overdue.value.length }
  })

  async function act(run: () => Promise<unknown>) {
    error.value = null
    try {
      await run()
      return true
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Не удалось сохранить'
      return false
    } finally {
      await refresh(true)
    }
  }

  const toggleDone = (entry: Entry) =>
    act(() => updateEntry(entry.id, { status: entry.status === 'completed' ? 'pending' : 'completed' }))
  const save = (entry: Entry, changes: EntryUpdate) => act(() => updateEntry(entry.id, changes))
  const create = (body: EntryCreate) => act(() => createEntry(body))
  const remove = (entry: Entry) => act(() => deleteEntry(entry.id))

  function setPeriod(next: Period) {
    if (period.value === next) return
    period.value = next
    loaded.value = false
    upcoming.value = []
    overdue.value = []
    undated.value = []
    void refresh()
  }

  return { period, sections, counts, loading, loaded, error, kind, showDone, refresh, setPeriod, toggleDone, save, create, remove }
})
