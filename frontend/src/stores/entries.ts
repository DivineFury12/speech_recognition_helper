import { defineStore } from 'pinia'
import { ref } from 'vue'
import { deleteEntry, listEntries, listUndatedEntries, updateEntry } from '@/api/client'
import type { Entry, EntryUpdate } from '@/api/types'
import { addDays, isoDate } from '@/utils/time'

export type Period = 'today' | 'week' | 'undated'

export const useEntriesStore = defineStore('entries', () => {
  const period = ref<Period>('today')
  const items = ref<Entry[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  let reqId = 0

  async function refresh(silent = false) {
    const id = ++reqId
    if (!silent) loading.value = true
    error.value = null
    try {
      let res
      if (period.value === 'undated') {
        res = await listUndatedEntries()
      } else {
        const now = new Date()
        const date_from = isoDate(now)
        const date_to = period.value === 'today' ? date_from : isoDate(addDays(now, 7))
        res = await listEntries({ date_from, date_to })
      }
      if (id === reqId) items.value = res.items
    } catch (e) {
      if (id === reqId) error.value = e instanceof Error ? e.message : 'Не удалось загрузить записи'
    } finally {
      if (id === reqId) loading.value = false
    }
  }

  function setPeriod(p: Period) {
    period.value = p
    return refresh()
  }

  async function act(run: () => Promise<unknown>) {
    error.value = null
    try {
      await run()
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Не удалось сохранить'
    } finally {
      await refresh(true)
    }
  }

  const toggleDone = (entry: Entry) =>
    act(() => updateEntry(entry.id, { status: entry.status === 'completed' ? 'pending' : 'completed' }))
  const save = (entry: Entry, changes: EntryUpdate) => act(() => updateEntry(entry.id, changes))
  const remove = (entry: Entry) => act(() => deleteEntry(entry.id))

  return { period, items, loading, error, refresh, setPeriod, toggleDone, save, remove }
})
