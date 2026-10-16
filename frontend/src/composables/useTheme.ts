import { computed, ref, watch } from 'vue'

export type ThemeMode = 'system' | 'light' | 'dark'

const STORAGE_KEY = 'theme'
const ORDER: ThemeMode[] = ['system', 'light', 'dark']
const LABEL: Record<ThemeMode, string> = {
  system: 'Тема как в системе',
  light: 'Светлая тема',
  dark: 'Тёмная тема',
}

function readMode(): ThemeMode {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved === 'light' || saved === 'dark' ? saved : 'system'
  } catch {
    return 'system'
  }
}

const media = typeof window !== 'undefined' ? window.matchMedia('(prefers-color-scheme: dark)') : null
const systemDark = ref(media?.matches ?? false)
media?.addEventListener('change', (e) => (systemDark.value = e.matches))

const mode = ref<ThemeMode>(readMode())
const dark = computed(() => (mode.value === 'system' ? systemDark.value : mode.value === 'dark'))

function apply() {
  const root = document.documentElement
  if (mode.value === 'system') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', mode.value)
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', dark.value ? '#0f1115' : '#f3f4f7')
}

watch([mode, systemDark], () => {
  try {
    localStorage.setItem(STORAGE_KEY, mode.value)
  } catch {}
  apply()
})

export function initTheme() {
  apply()
}

export function useTheme() {
  function cycle() {
    mode.value = ORDER[(ORDER.indexOf(mode.value) + 1) % ORDER.length]
  }
  const label = computed(() => LABEL[mode.value])
  return { mode, dark, label, cycle }
}
