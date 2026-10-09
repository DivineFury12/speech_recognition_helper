<script setup lang="ts">
import { onMounted } from 'vue'
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

onMounted(() => store.refresh())
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
      <button type="button" class="link" :disabled="loading" @click="store.refresh()">
        {{ loading ? 'Загрузка…' : 'Обновить' }}
      </button>
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
