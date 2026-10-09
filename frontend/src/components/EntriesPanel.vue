<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useEntriesStore, type Period } from '@/stores/entries'
import EntryItem from './EntryItem.vue'

const store = useEntriesStore()
const { period, items, loading, error } = storeToRefs(store)

const tabs: { id: Period; label: string }[] = [
  { id: 'today', label: 'Сегодня' },
  { id: 'week', label: 'Неделя' },
  { id: 'undated', label: 'Без срока' },
]

const emptyText: Record<Period, string> = {
  today: 'На сегодня записей нет',
  week: 'На неделю записей нет',
  undated: 'Задач без срока нет',
}

const REFRESH_MS = 60_000
let timer: number | undefined

onMounted(() => {
  void store.refresh()
  timer = window.setInterval(() => {
    if (document.visibilityState === 'visible') void store.refresh(true)
  }, REFRESH_MS)
})
onBeforeUnmount(() => window.clearInterval(timer))
</script>

<template>
  <aside class="panel entries" aria-label="Записи">
    <div class="entries-head">
      <div class="tabs" role="tablist">
        <button
          v-for="t in tabs"
          :key="t.id"
          type="button"
          role="tab"
          :aria-selected="period === t.id"
          :class="{ active: period === t.id }"
          @click="store.setPeriod(t.id)"
        >
          {{ t.label }}
        </button>
      </div>
      <span v-if="loading" class="spinner small" aria-label="Обновляю" />
    </div>

    <div v-if="error" class="banner error">{{ error }}</div>
    <p v-if="!items.length && !loading" class="empty">{{ emptyText[period] }}</p>
    <ul class="entry-list" :class="{ dim: loading }">
      <EntryItem
        v-for="e in items"
        :key="`${e.id}:${e.date ?? ''}`"
        :entry="e"
        @toggle="store.toggleDone"
        @save="store.save"
        @remove="store.remove"
      />
    </ul>
  </aside>
</template>
