<script setup lang="ts">
import { ref } from 'vue'

const open = ref(false)

const groups: { title: string; items: string[] }[] = [
  {
    title: 'Создать',
    items: [
      '«Напомни завтра в 10 утра позвонить Ивану»',
      '«Добавь встречу с командой в четверг в 15:00 на час»',
      '«Поставь задачу купить билеты до конца недели»',
      '«Планёрка каждый понедельник в 10»',
    ],
  },
  {
    title: 'Посмотреть',
    items: ['«Что у меня сегодня?»', '«Какие задачи на завтра?»', '«Что запланировано на следующей неделе?»'],
  },
  {
    title: 'Найти',
    items: ['«Найди задачу про презентацию»', '«Когда у меня встреча с Иваном?»', 'Можно другими словами — ищу по смыслу'],
  },
  {
    title: 'Изменить',
    items: [
      '«А перенеси её на завтра» — про последнюю запись',
      '«Перенеси встречу с дизайнером на 17:00»',
      '«Переименуй её в …»',
      '«Отметь задачу про отчёт выполненной», «Готово»',
      '«Удали напоминание про таблетки»',
    ],
  },
  {
    title: 'Ещё',
    items: [
      'Если не хватает данных — переспрошу',
      'Предупрежу, если встречи пересекаются',
      'Голосом: кнопка микрофона, до 25 секунд',
      '«Отмена» — сбросить уточнение',
    ],
  },
]
</script>

<template>
  <div class="help" @mouseenter="open = true" @mouseleave="open = false">
    <button
      type="button"
      class="help-btn"
      aria-label="Что умеет ассистент"
      :aria-expanded="open"
      aria-controls="help-pop"
      @click="open = !open"
      @focus="open = true"
      @blur="open = false"
      @keydown.esc="open = false"
    >
      ?
    </button>
    <div v-show="open" id="help-pop" class="help-pop" role="tooltip">
      <p class="help-title">Что я умею</p>
      <div v-for="g in groups" :key="g.title" class="help-group">
        <span class="help-group-title">{{ g.title }}</span>
        <ul>
          <li v-for="item in g.items" :key="item">{{ item }}</li>
        </ul>
      </div>
    </div>
  </div>
</template>
