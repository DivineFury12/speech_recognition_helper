<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useEntriesStore, type KindFilter, type Period } from '@/stores/entries'
import { useEditor } from '@/composables/useEditor'
import EntryCard from './EntryCard.vue'

const store = useEntriesStore()
const { period, sections, counts, loading, loaded, error, kind, showDone } = storeToRefs(store)
const editor = useEditor()

const FILTERS: { id: KindFilter; label: string }[] = [
  { id: 'all', label: 'Все' },
  { id: 'task', label: 'Задачи' },
  { id: 'meeting', label: 'Встречи' },
  { id: 'reminder', label: 'Напоминания' },
]
const PERIODS: { id: Period; label: string }[] = [
  { id: 'today', label: 'Сегодня' },
  { id: 'week', label: 'На неделю' },
  { id: 'month', label: 'Месяц' },
  { id: 'undated', label: 'Без срока' },
]

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
  <section class="panel agenda" aria-label="Записи">
    <header class="agenda-head">
      <div class="agenda-title">
        <h2>Записи</h2>
        <span v-if="loaded" class="agenda-count">
          {{ counts.total ? `${counts.total} активных` : 'всё сделано' }}
          <template v-if="counts.overdue"> · <b class="late">{{ counts.overdue }} просрочено</b></template>
        </span>
      </div>
      <button type="button" class="add-btn" aria-label="Новая запись" title="Новая запись" @click="editor.create()">
        <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path fill="currentColor" d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z" /></svg>
      </button>
    </header>

    <div class="filters" role="toolbar" aria-label="Период">
      <button
        v-for="p in PERIODS"
        :key="p.id"
        type="button"
        class="filter-chip"
        :class="{ active: period === p.id }"
        :aria-pressed="period === p.id"
        @click="store.setPeriod(p.id)"
      >
        {{ p.label }}
      </button>
    </div>
    <div class="filters" role="toolbar" aria-label="Тип записи">
      <button
        v-for="f in FILTERS"
        :key="f.id"
        type="button"
        class="filter-chip"
        :class="{ active: kind === f.id }"
        :aria-pressed="kind === f.id"
        @click="kind = f.id"
      >
        {{ f.label }}
      </button>
      <label class="done-toggle">
        <input v-model="showDone" type="checkbox" />
        выполненные
      </label>
    </div>

    <div v-if="error" class="banner error">{{ error }}</div>

    <div class="agenda-scroll">
      <div v-if="!loaded && loading" class="skeleton-list" aria-label="Загружаю записи">
        <div v-for="n in 4" :key="n" class="skeleton-card" />
      </div>
      <template v-else>
        <section v-for="s in sections" :key="s.id" class="day" :class="s.id">
          <h3 class="day-head">
            {{ s.title }}<small v-if="s.hint">{{ s.hint }}</small>
            <span v-if="s.items.length" class="day-count">{{ s.items.length }}</span>
          </h3>
          <TransitionGroup tag="ul" name="list" class="card-list">
            <EntryCard
              v-for="e in s.items"
              :key="`${e.id}:${e.date ?? ''}`"
              :entry="e"
              :show-date="s.id === 'overdue'"
              @toggle="store.toggleDone"
              @open="editor.edit"
              @remove="store.remove"
            />
          </TransitionGroup>
          <div v-if="!s.items.length" class="day-empty">
            <svg viewBox="0 0 64 64" width="44" height="44" aria-hidden="true">
              <circle cx="32" cy="32" r="22" fill="none" stroke="currentColor" stroke-width="3" opacity="0.35" />
              <path d="M22 33l7 7 14-15" fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
            </svg>
            <span>{{ s.id === 'undated' ? 'Записей без срока нет' : 'Свободный день. Скажите «Поставь встречу…» или нажмите +' }}</span>
          </div>
        </section>
        <p v-if="!sections.length" class="day-empty">За выбранный период записей нет</p>
      </template>
    </div>
  </section>
</template>
