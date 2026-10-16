<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { downloadCalendar, getCalendarLink, rotateCalendarLink } from '@/api/client'

const emit = defineEmits<{ notify: [text: string] }>()

const open = ref(false)
const busy = ref(false)
const root = ref<HTMLElement | null>(null)
async function run(action: () => Promise<string>) {
  if (busy.value) return
  busy.value = true
  try {
    emit('notify', await action())
    open.value = false
  } catch (e) {
    emit('notify', e instanceof Error ? e.message : 'Не получилось')
  } finally {
    busy.value = false
  }
}

const download = () =>
  run(async () => {
    const blob = await downloadCalendar()
    const href = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = href
    a.download = 'assistant.ics'
    a.click()
    URL.revokeObjectURL(href)
    return 'Файл assistant.ics скачан — откройте его, чтобы добавить записи в календарь'
  })

const copyLink = () =>
  run(async () => {
    const { url } = await getCalendarLink()
    await navigator.clipboard.writeText(url)
    return 'Ссылка скопирована. В календаре выберите «Добавить по URL» — нужен сервер, доступный из интернета'
  })

const rotate = () =>
  run(async () => {
    if (!window.confirm('Старая ссылка перестанет работать. Выпустить новую?')) return 'Ссылка не изменилась'
    const { url } = await rotateCalendarLink()
    await navigator.clipboard.writeText(url).catch(() => undefined)
    return 'Новая ссылка выпущена и скопирована, старая больше не работает'
  })

function toggle() {
  open.value = !open.value
}

function onDocumentClick(e: MouseEvent) {
  if (open.value && root.value && !root.value.contains(e.target as Node)) open.value = false
}

onMounted(() => document.addEventListener('click', onDocumentClick))
onBeforeUnmount(() => document.removeEventListener('click', onDocumentClick))
</script>

<template>
  <div ref="root" class="calendar-menu" @keydown.esc="open = false">
    <button
      type="button"
      class="icon-action"
      :aria-expanded="open"
      title="Календарь"
      aria-label="Календарь"
      @click.stop="toggle"
    >
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
        <path fill="currentColor" d="M19 4h-1V2h-2v2H8V2H6v2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2Zm0 16H5V9h14v11ZM7 11h5v5H7v-5Z" />
      </svg>
    </button>
    <div v-show="open" class="menu" role="menu" aria-label="Календарь">
      <div class="menu-title">Календарь</div>
      <button type="button" class="menu-item" role="menuitem" :disabled="busy" @click="download">
        Скачать файл .ics
        <small>для Apple, Outlook или любого календаря</small>
      </button>
      <button type="button" class="menu-item" role="menuitem" :disabled="busy" @click="copyLink">
        Скопировать ссылку для подписки
      </button>
      <button type="button" class="menu-item danger" role="menuitem" :disabled="busy" @click="rotate">
        Выпустить новую ссылку
        <small>если старая попала не в те руки</small>
      </button>
    </div>
  </div>
</template>
