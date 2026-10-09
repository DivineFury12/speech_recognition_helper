import { defineStore } from 'pinia'
import { ref } from 'vue'
import { listEntries } from '@/api/client'
import type { Entry } from '@/api/types'
import { addDays, isoDate } from '@/utils/time'

export type Period = 'today' | 'week'

export const useEntriesStore = defineStore('entries', () => {
  const period = ref<Period>('today')
  const items = ref<Entry[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  let reqId = 0

  async function refresh() {
    const now = new Date()
    const date_from = isoDate(now)
    const date_to = period.value === 'today' ? date_from : isoDate(addDays(now, 6))
    const id = ++reqId
    loading.value = true
    error.value = null
    try {
      const res = await listEntries({ date_from, date_to })
      if (id === reqId) items.value = [...res.items].sort(compareEntries)
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

  return { period, items, loading, error, refresh, setPeriod }
})

function compareEntries(a: Entry, b: Entry): number {
  const ka = `${a.date ?? '9999-99-99'} ${a.time ?? '99:99'}`
  const kb = `${b.date ?? '9999-99-99'} ${b.time ?? '99:99'}`
  return ka.localeCompare(kb)
}
