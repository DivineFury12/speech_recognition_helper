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
]

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
    <p v-else-if="!items.length && !loading" class="empty">Записей нет</p>
    <ul class="entry-list" :class="{ dim: loading }">
      <EntryItem v-for="e in items" :key="e.id" :entry="e" />
    </ul>
  </aside>
</template>
