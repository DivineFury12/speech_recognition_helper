<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'

const open = ref(false)
const pinned = ref(false)
const root = ref<HTMLElement | null>(null)
let closeTimer: number | undefined

function show() {
  window.clearTimeout(closeTimer)
  open.value = true
}

function scheduleClose() {
  if (pinned.value) return
  window.clearTimeout(closeTimer)
  closeTimer = window.setTimeout(() => (open.value = false), 250)
}

function toggle() {
  pinned.value = !pinned.value || !open.value
  open.value = pinned.value
}

function close() {
  pinned.value = false
  open.value = false
}

function onDocumentClick(e: MouseEvent) {
  if (open.value && root.value && !root.value.contains(e.target as Node)) close()
}

onMounted(() => document.addEventListener('click', onDocumentClick))
onBeforeUnmount(() => {
  document.removeEventListener('click', onDocumentClick)
  window.clearTimeout(closeTimer)
})

const groups: { title: string; items: string[] }[] = [
  {
    title: 'Создать',
    items: [
      'Напомни завтра в 10 утра позвонить Ивану',
      'Добавь встречу с командой в четверг в 15:00 на час',
      'Поставь задачу купить билеты до конца недели',
      'Планёрка каждый понедельник в 10',
    ],
  },
  {
    title: 'Посмотреть',
    items: ['Что у меня сегодня?', 'Какие задачи на завтра?', 'Что запланировано на следующей неделе?'],
  },
  {
    title: 'Найти',
    items: ['Найди задачу про презентацию', 'Когда у меня встреча с Иваном?', 'Можно другими словами — ищу по смыслу'],
  },
  {
    title: 'Изменить',
    items: [
      'А перенеси её на завтра — про последнюю запись',
      'Перенеси встречу с дизайнером на 17:00',
      'Переименуй её в …',
      'Отметь задачу про отчёт выполненной, или просто «Готово»',
      'Удали напоминание про таблетки',
    ],
  },
  {
    title: 'Ещё',
    items: [
      'Если не хватает данных — переспрошу',
      'Предупрежу, если встречи пересекаются',
      'Голосом: кнопка микрофона, до 25 секунд',
      '«Отмена» сбрасывает уточнение',
    ],
  },
]
</script>

<template>
  <div ref="root" class="help" @mouseenter="show" @mouseleave="scheduleClose" @keydown.esc="close">
    <button
      type="button"
      class="icon-action help-btn"
      :class="{ active: open }"
      aria-label="Что умеет ассистент"
      :aria-expanded="open"
      aria-controls="help-pop"
      @click.stop="toggle"
    >
      ?
    </button>
    <div v-show="open" id="help-pop" class="help-pop" role="dialog" aria-label="Что умеет ассистент">
      <div class="help-card">
        <div class="help-head">
          <span class="help-title">Что я умею</span>
          <button type="button" class="help-close" aria-label="Закрыть" @click="close">✕</button>
        </div>
        <div class="help-scroll">
          <div v-for="g in groups" :key="g.title" class="help-group">
            <span class="help-group-title">{{ g.title }}</span>
            <ul>
              <li v-for="item in g.items" :key="item">{{ item }}</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
