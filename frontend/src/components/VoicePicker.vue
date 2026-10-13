<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { BROWSER_VOICE, VOICE_OFF, useSpeech } from '@/composables/useSpeech'

const {
  voice,
  volume,
  enabled,
  currentName,
  serverVoices,
  serverAvailable,
  serverLoading,
  previewing,
  browserSupported,
  loadVoices,
  preview,
  stop,
} = useSpeech()

const open = ref(false)
const root = ref<HTMLElement | null>(null)

const options = computed(() => [
  ...serverVoices.value.map((v) => ({
    id: v.id,
    name: v.name,
    note: serverAvailable.value ? v.description : serverLoading.value ? 'загружается…' : 'недоступен',
    disabled: !serverAvailable.value,
  })),
  ...(browserSupported ? [{ id: BROWSER_VOICE, name: 'Браузер', note: 'встроенный голос системы', disabled: false }] : []),
])

function choose(id: string) {
  voice.value = id
  open.value = false
}

function togglePreview(id: string) {
  if (previewing.value === id) stop()
  else preview(id)
}

function onDocumentClick(e: MouseEvent) {
  if (open.value && root.value && !root.value.contains(e.target as Node)) open.value = false
}

onMounted(() => {
  void loadVoices()
  document.addEventListener('click', onDocumentClick)
})
onBeforeUnmount(() => document.removeEventListener('click', onDocumentClick))
</script>

<template>
  <div ref="root" class="voice-picker" @keydown.esc="open = false">
    <button
      type="button"
      class="voice-trigger"
      :class="{ active: enabled, open }"
      :aria-expanded="open"
      aria-haspopup="listbox"
      :title="enabled ? `Озвучка: ${currentName}` : 'Озвучка выключена'"
      @click.stop="open = !open; open && loadVoices()"
    >
      <svg v-if="enabled" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
        <path fill="currentColor" d="M3 9v6h4l5 5V4L7 9H3Zm13.5 3A4.5 4.5 0 0 0 14 8v8a4.47 4.47 0 0 0 2.5-4ZM14 3.23v2.06a7 7 0 0 1 0 13.42v2.06A9 9 0 0 0 14 3.23Z" />
      </svg>
      <svg v-else viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
        <path fill="currentColor" d="M16.5 12A4.5 4.5 0 0 0 14 8v2.18l2.45 2.45c.03-.2.05-.41.05-.63ZM19 12c0 .94-.2 1.82-.54 2.64l1.51 1.51A8.8 8.8 0 0 0 21 12a9 9 0 0 0-7-8.77v2.06A7 7 0 0 1 19 12ZM4.27 3 3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06a9 9 0 0 0 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3ZM12 4 9.91 6.09 12 8.18V4Z" />
      </svg>
      <span class="voice-label">{{ enabled ? currentName : 'Озвучка' }}</span>
      <svg class="chevron" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
        <path fill="currentColor" d="M7 10l5 5 5-5z" />
      </svg>
    </button>

    <div v-show="open" class="voice-menu" role="listbox" aria-label="Голос озвучки">
      <div class="voice-menu-title">Голос ответов</div>
      <div
        v-for="o in options"
        :key="o.id"
        class="voice-option"
        :class="{ selected: voice === o.id, disabled: o.disabled }"
        role="option"
        :aria-selected="voice === o.id"
        :aria-disabled="o.disabled"
      >
        <button type="button" class="voice-choose" :disabled="o.disabled" @click="choose(o.id)">
          <span class="voice-avatar" :class="o.id">{{ o.name.slice(0, 1) }}</span>
          <span class="voice-text">
            <span class="voice-name">{{ o.name }}</span>
            <span class="voice-note">{{ o.note }}</span>
          </span>
          <svg v-if="voice === o.id" class="voice-check" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path fill="currentColor" d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
          </svg>
        </button>
        <button
          type="button"
          class="voice-play"
          :disabled="o.disabled"
          :aria-label="previewing === o.id ? 'Остановить' : `Послушать голос ${o.name}`"
          :title="previewing === o.id ? 'Остановить' : 'Послушать'"
          @click="togglePreview(o.id)"
        >
          <svg v-if="previewing === o.id" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M6 6h12v12H6z" /></svg>
          <svg v-else viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M8 5v14l11-7z" /></svg>
        </button>
      </div>
      <div class="voice-divider" />
      <label class="voice-volume">
        <span>Громкость</span>
        <input v-model.number="volume" type="range" min="0" max="1" step="0.05" aria-label="Громкость озвучки" />
        <span class="voice-volume-value">{{ Math.round(volume * 100) }}%</span>
      </label>
      <div class="voice-divider" />
      <div class="voice-option" :class="{ selected: voice === VOICE_OFF }" role="option" :aria-selected="voice === VOICE_OFF">
        <button type="button" class="voice-choose" @click="choose(VOICE_OFF)">
          <span class="voice-avatar off">—</span>
          <span class="voice-text"><span class="voice-name">Без озвучки</span></span>
          <svg v-if="voice === VOICE_OFF" class="voice-check" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path fill="currentColor" d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>
