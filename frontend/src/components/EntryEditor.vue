<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { Entry, EntryCreate, EntryUpdate, Recurrence } from '@/api/types'
import { useEditor } from '@/composables/useEditor'
import { useEntriesStore } from '@/stores/entries'
import { isoDate, timezone } from '@/utils/time'

const store = useEntriesStore()
const { current, creating, open, close } = useEditor()

const KINDS: { id: Entry['kind']; label: string }[] = [
  { id: 'task', label: 'Задача' },
  { id: 'reminder', label: 'Напоминание' },
  { id: 'meeting', label: 'Встреча' },
]
const LEADS = [
  { value: '', label: 'По умолчанию' },
  { value: '0', label: 'В момент начала' },
  { value: '5', label: 'За 5 минут' },
  { value: '10', label: 'За 10 минут' },
  { value: '15', label: 'За 15 минут' },
  { value: '30', label: 'За 30 минут' },
  { value: '60', label: 'За час' },
  { value: '1440', label: 'За день' },
]

const form = ref({
  kind: 'task' as Entry['kind'],
  title: '',
  date: '',
  time: '',
  duration: '60',
  recurrence: '' as '' | NonNullable<Recurrence>,
  until: '',
  lead: '',
  important: false,
})
const saving = ref(false)
const confirmDelete = ref(false)
const problem = ref<string | null>(null)
const titleEl = ref<HTMLInputElement | null>(null)

const needsTime = computed(() => form.value.kind !== 'task')

watch(open, async (isOpen) => {
  if (!isOpen) return
  const e = current.value
  confirmDelete.value = false
  problem.value = null
  form.value = e
    ? {
        kind: e.kind,
        title: e.title,
        date: e.date ?? '',
        time: e.time ?? '',
        duration: String(e.duration_minutes ?? 60),
        recurrence: e.recurrence ?? '',
        until: e.recurrence_until ?? '',
        lead: e.remind_minutes == null ? '' : String(e.remind_minutes),
        important: e.priority === 'high',
      }
    : { kind: 'task', title: '', date: isoDate(new Date()), time: '', duration: '60', recurrence: '', until: '', lead: '', important: false }
  await nextTick()
  titleEl.value?.focus()
})

function validate(): string | null {
  const f = form.value
  if (!f.title.trim()) return 'Введите название'
  if (needsTime.value && (!f.date || !f.time)) return 'Для встречи и напоминания нужны дата и время'
  if (f.time && !f.date) return 'Время без даты не сохранится — укажите дату'
  if (f.recurrence && !f.date) return 'Для повтора нужна дата начала'
  return null
}

async function submit() {
  problem.value = validate()
  if (problem.value) return
  const f = form.value
  saving.value = true
  const lead = f.lead === '' ? null : Number(f.lead)
  let ok: boolean
  if (creating.value) {
    const body: EntryCreate = {
      kind: f.kind,
      title: f.title.trim(),
      date: f.date || null,
      time: f.time || null,
      timezone,
      duration_minutes: f.kind === 'meeting' ? Number(f.duration) || 60 : null,
      recurrence: f.recurrence || null,
      recurrence_until: f.recurrence && f.until ? f.until : null,
      remind_minutes: lead,
      priority: f.important ? 'high' : 'normal',
    }
    ok = await store.create(body)
  } else {
    const e = current.value!
    const changes: EntryUpdate = {}
    if (f.title.trim() !== e.title) changes.title = f.title.trim()
    if ((f.date || null) !== e.date) changes.date = f.date || null
    if ((f.time || null) !== e.time) changes.time = f.time || null
    if (e.kind === 'meeting' && Number(f.duration) !== (e.duration_minutes ?? 60)) changes.duration_minutes = Number(f.duration)
    if ((f.recurrence || null) !== (e.recurrence ?? null)) changes.recurrence = f.recurrence || null
    if ((f.recurrence && f.until ? f.until : null) !== (e.recurrence_until ?? null)) changes.recurrence_until = f.recurrence && f.until ? f.until : null
    if (lead !== (e.remind_minutes ?? null)) changes.remind_minutes = lead
    if (f.important !== (e.priority === 'high')) changes.priority = f.important ? 'high' : 'normal'
    ok = Object.keys(changes).length ? await store.save(e, changes) : true
  }
  saving.value = false
  if (ok) close()
  else problem.value = store.error
}

async function remove() {
  if (!current.value) return
  saving.value = true
  const ok = await store.remove(current.value)
  saving.value = false
  if (ok) close()
}
</script>

<template>
  <Teleport to="body">
    <Transition name="sheet">
      <div v-if="open" class="sheet-backdrop" @click.self="close" @keydown.esc="close">
        <form
          class="sheet"
          role="dialog"
          aria-modal="true"
          :aria-label="creating ? 'Новая запись' : 'Изменить запись'"
          @submit.prevent="submit"
        >
          <div class="sheet-grip" aria-hidden="true" />
          <header class="sheet-head">
            <h2>{{ creating ? 'Новая запись' : 'Запись' }}</h2>
            <button type="button" class="help-close" aria-label="Закрыть" @click="close">✕</button>
          </header>

          <div class="segmented" role="radiogroup" aria-label="Тип записи">
            <button
              v-for="k in KINDS"
              :key="k.id"
              type="button"
              role="radio"
              :aria-checked="form.kind === k.id"
              :class="[k.id, { active: form.kind === k.id }]"
              :disabled="!creating"
              @click="form.kind = k.id"
            >
              {{ k.label }}
            </button>
          </div>

          <label class="form-row">
            <span>Название</span>
            <input ref="titleEl" v-model="form.title" class="field" maxlength="500" placeholder="Например, позвонить Ивану" />
          </label>

          <div class="form-grid">
            <label class="form-row">
              <span>Дата</span>
              <input v-model="form.date" type="date" class="field" />
            </label>
            <label class="form-row">
              <span>Время</span>
              <input v-model="form.time" type="time" class="field" />
            </label>
            <label v-if="form.kind === 'meeting'" class="form-row">
              <span>Длительность, мин</span>
              <input v-model="form.duration" type="number" min="5" max="1440" step="5" class="field" />
            </label>
            <label class="form-row">
              <span>Напомнить</span>
              <select v-model="form.lead" class="field" :disabled="!form.time">
                <option v-for="l in LEADS" :key="l.value" :value="l.value">{{ l.label }}</option>
              </select>
            </label>
          </div>

          <div class="form-grid">
            <label class="form-row">
              <span>Повтор</span>
              <select v-model="form.recurrence" class="field">
                <option value="">Без повтора</option>
                <option value="daily">Каждый день</option>
                <option value="weekdays">По будням</option>
                <option value="weekly">Каждую неделю</option>
                <option value="monthly">Каждый месяц</option>
              </select>
            </label>
            <label v-if="form.recurrence" class="form-row">
              <span>До даты</span>
              <input v-model="form.until" type="date" class="field" :min="form.date" />
            </label>
          </div>

          <label class="toggle-row">
            <input v-model="form.important" type="checkbox" />
            <span class="toggle" aria-hidden="true" />
            <span>Важно</span>
          </label>

          <Transition name="fade">
            <p v-if="problem" class="form-problem" role="alert">{{ problem }}</p>
          </Transition>

          <footer class="sheet-actions">
            <template v-if="!creating">
              <button v-if="!confirmDelete" type="button" class="link danger" @click="confirmDelete = true">Удалить</button>
              <span v-else class="confirm-inline">
                Точно удалить?
                <button type="button" class="link danger" :disabled="saving" @click="remove">Да</button>
                <button type="button" class="link" @click="confirmDelete = false">Нет</button>
              </span>
            </template>
            <span class="spacer" />
            <button type="button" class="ghost-btn" @click="close">Отмена</button>
            <button type="submit" class="primary-btn" :disabled="saving">
              <span v-if="saving" class="spinner small light" aria-hidden="true" />
              {{ creating ? 'Создать' : 'Сохранить' }}
            </button>
          </footer>
        </form>
      </div>
    </Transition>
  </Teleport>
</template>
