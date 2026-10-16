<script setup lang="ts">
import { computed, ref } from 'vue'
import type { Entry, EntryUpdate, Recurrence } from '@/api/types'
import { formatDate } from '@/utils/time'

const props = defineProps<{ entry: Entry; compact?: boolean }>()
const emit = defineEmits<{
  toggle: [entry: Entry]
  save: [entry: Entry, changes: EntryUpdate]
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

const when = computed(() => {
  const e = props.entry
  const parts: string[] = []
  if (e.date) parts.push(formatDate(e.date))
  if (e.time && e.kind === 'meeting' && e.duration_minutes) {
    const [hours, minutes] = e.time.split(':').map(Number)
    const end = hours * 60 + minutes + e.duration_minutes
    const endTime = `${String(Math.floor(end / 60) % 24).padStart(2, '0')}:${String(end % 60).padStart(2, '0')}`
    parts.push(`${e.time}–${endTime}${end >= 1440 ? ' (+1 день)' : ''}`)
  } else if (e.time) parts.push(e.time)
  if (e.duration_minutes) parts.push(`${e.duration_minutes} мин`)
  if (e.remind_minutes) parts.push(`напомню за ${e.remind_minutes >= 60 && e.remind_minutes % 60 === 0 ? `${e.remind_minutes / 60} ч` : `${e.remind_minutes} мин`}`)
  if (e.recurrence_until) parts.push(`до ${formatDate(e.recurrence_until)}`)
  return parts.join(' · ') || 'без срока'
})

const repeat = computed(() => (props.entry.recurrence ? REPEAT_LABEL[props.entry.recurrence] : ''))

const editing = ref(false)
const confirmDelete = ref(false)
const form = ref({ title: '', date: '', time: '', duration: '', recurrence: '' as '' | NonNullable<Recurrence> })

function startEdit() {
  const e = props.entry
  form.value = {
    title: e.title,
    date: e.date ?? '',
    time: e.time ?? '',
    duration: e.duration_minutes ? String(e.duration_minutes) : '',
    recurrence: e.recurrence ?? '',
  }
  confirmDelete.value = false
  editing.value = true
}

function submit() {
  const f = form.value
  const e = props.entry
  const changes: EntryUpdate = {}
  if (f.title.trim() && f.title.trim() !== e.title) changes.title = f.title.trim()
  if ((f.date || null) !== e.date) changes.date = f.date || null
  if ((f.time || null) !== e.time) changes.time = f.time || null
  const duration = f.duration ? Number(f.duration) : null
  if (e.kind === 'meeting' && duration !== (e.duration_minutes ?? null)) changes.duration_minutes = duration
  if ((f.recurrence || null) !== (e.recurrence ?? null)) changes.recurrence = f.recurrence || null
  if (Object.keys(changes).length) emit('save', e, changes)
  editing.value = false
}
</script>

<template>
  <li class="entry" :class="[entry.kind, { compact, done }]">
    <template v-if="!editing">
      <div class="entry-main">
        <input
          v-if="!compact"
          type="checkbox"
          class="done-box"
          :checked="done"
          :aria-label="done ? 'Вернуть в работу' : 'Отметить выполненной'"
          @change="emit('toggle', entry)"
        />
        <div class="entry-body">
          <span class="kind">{{ KIND_LABEL[entry.kind] }}<template v-if="repeat"> · ↻ {{ repeat }}</template></span>
          <span class="title">{{ entry.title }}</span>
          <span class="when">{{ when }}</span>
        </div>
      </div>
      <div v-if="!compact" class="entry-actions">
        <template v-if="!confirmDelete">
          <button type="button" class="icon-btn" title="Изменить" aria-label="Изменить" @click="startEdit">✎</button>
          <button type="button" class="icon-btn" title="Удалить" aria-label="Удалить" @click="confirmDelete = true">✕</button>
        </template>
        <template v-else>
          <span class="confirm-text">Удалить?</span>
          <button type="button" class="link danger" @click="emit('remove', entry)">Да</button>
          <button type="button" class="link" @click="confirmDelete = false">Нет</button>
        </template>
      </div>
    </template>
    <form v-else class="entry-edit" @submit.prevent="submit" @keydown.esc="editing = false">
      <input v-model="form.title" class="field wide" aria-label="Название" required />
      <div class="edit-row">
        <input v-model="form.date" type="date" class="field" aria-label="Дата" />
        <input v-model="form.time" type="time" class="field" aria-label="Время" />
        <input
          v-if="entry.kind === 'meeting'"
          v-model="form.duration"
          type="number"
          min="1"
          class="field short"
          aria-label="Длительность, мин"
          placeholder="мин"
        />
      </div>
      <select v-model="form.recurrence" class="field wide" aria-label="Повтор">
        <option value="">Без повтора</option>
        <option value="daily">Каждый день</option>
        <option value="weekdays">По будням</option>
        <option value="weekly">Каждую неделю</option>
        <option value="monthly">Каждый месяц</option>
      </select>
      <div class="edit-row end">
        <button type="button" class="link" @click="editing = false">Отмена</button>
        <button type="submit" class="small-btn">Сохранить</button>
      </div>
    </form>
  </li>
</template>
