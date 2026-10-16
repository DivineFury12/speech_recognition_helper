<script setup lang="ts">
import { computed, ref } from 'vue'
import type { Entry, Recurrence } from '@/api/types'
import { formatDate } from '@/utils/time'

const props = defineProps<{ entry: Entry; compact?: boolean; showDate?: boolean }>()
const emit = defineEmits<{
  toggle: [entry: Entry]
  open: [entry: Entry]
  remove: [entry: Entry]
}>()

const KIND_LABEL: Record<Entry['kind'], string> = { task: 'Задача', reminder: 'Напоминание', meeting: 'Встреча' }
const REPEAT_LABEL: Record<NonNullable<Recurrence>, string> = {
  daily: 'каждый день',
  weekdays: 'по будням',
  weekly: 'каждую неделю',
  monthly: 'каждый месяц',
}

const done = computed(() => props.entry.status === 'completed')
const important = computed(() => props.entry.priority === 'high')

const endTime = computed(() => {
  const e = props.entry
  if (!e.time || e.kind !== 'meeting' || !e.duration_minutes) return null
  const [h, m] = e.time.split(':').map(Number)
  const end = h * 60 + m + e.duration_minutes
  return `${String(Math.floor(end / 60) % 24).padStart(2, '0')}:${String(end % 60).padStart(2, '0')}`
})

const lead = computed(() => {
  const minutes = props.entry.remind_minutes
  if (!minutes) return null
  if (minutes % 1440 === 0) return `за ${minutes / 1440} д`
  if (minutes % 60 === 0) return `за ${minutes / 60} ч`
  return `за ${minutes} мин`
})

const offset = ref(0)
let startX = 0
let startY = 0
let swiping = false

function onTouchStart(e: TouchEvent) {
  startX = e.touches[0].clientX
  startY = e.touches[0].clientY
  swiping = false
}

function onTouchMove(e: TouchEvent) {
  const dx = e.touches[0].clientX - startX
  const dy = e.touches[0].clientY - startY
  if (!swiping && Math.abs(dx) > 12 && Math.abs(dx) > Math.abs(dy)) swiping = true
  if (swiping) offset.value = Math.max(-96, Math.min(0, dx))
}

function onTouchEnd() {
  offset.value = offset.value < -56 ? -88 : 0
}

function openCard() {
  if (offset.value !== 0) {
    offset.value = 0
    return
  }
  emit('open', props.entry)
}
</script>

<template>
  <li class="card-wrap" :class="{ compact }">
    <button v-if="!compact" type="button" class="card-delete" tabindex="-1" aria-hidden="true" @click="emit('remove', entry)">
      Удалить
    </button>
    <div
      class="card"
      :class="[entry.kind, { done, important }]"
      :style="offset ? { transform: `translateX(${offset}px)` } : undefined"
      @touchstart.passive="onTouchStart"
      @touchmove.passive="onTouchMove"
      @touchend="onTouchEnd"
    >
      <button
        v-if="!compact"
        type="button"
        class="check"
        :class="{ on: done }"
        :aria-label="done ? 'Вернуть в работу' : 'Отметить выполненной'"
        :aria-pressed="done"
        @click.stop="emit('toggle', entry)"
      >
        <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path d="M5 12.5l4.2 4.2L19 7" /></svg>
      </button>
      <button type="button" class="card-body" :aria-label="`Открыть: ${entry.title}`" @click="openCard">
        <span class="card-time">
          <template v-if="entry.time">
            <b>{{ entry.time }}</b>
            <small v-if="endTime">{{ endTime }}</small>
          </template>
          <small v-else>весь день</small>
        </span>
        <span class="card-main">
          <span class="card-title">
            <svg v-if="important" class="flag" viewBox="0 0 24 24" width="13" height="13" aria-label="Важно">
              <path fill="currentColor" d="M5 21V4h9l.4 2H20v10h-7l-.4-2H7v7H5Z" />
            </svg>
            {{ entry.title }}
          </span>
          <span class="card-meta">
            <span class="kind-dot" aria-hidden="true" />{{ KIND_LABEL[entry.kind] }}
            <template v-if="showDate && entry.date"> · {{ formatDate(entry.date) }}</template>
            <template v-if="entry.recurrence"> · ↻ {{ REPEAT_LABEL[entry.recurrence] }}</template>
            <template v-if="lead"> · напомню {{ lead }}</template>
          </span>
        </span>
      </button>
    </div>
  </li>
</template>
