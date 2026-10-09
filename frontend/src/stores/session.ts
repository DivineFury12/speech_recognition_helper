import { defineStore } from 'pinia'
import { ref } from 'vue'
import { ApiError, getSession, onAuthError, unlockSession } from '@/api/client'

export const useSessionStore = defineStore('session', () => {
  const loaded = ref(false)
  const unlocked = ref(true)
  const accessRequired = ref(false)
  const llmProvider = ref<string | null>(null)
  const asrEnabled = ref<boolean | null>(null)
  const error = ref<string | null>(null)
  const unlocking = ref(false)

  onAuthError(() => {
    unlocked.value = false
    accessRequired.value = true
  })

  async function load() {
    try {
      const s = await getSession()
      unlocked.value = s.unlocked
      accessRequired.value = s.access_code_required
      llmProvider.value = s.llm_provider
      asrEnabled.value = s.asr_enabled
      error.value = null
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Не удалось получить сессию'
    } finally {
      loaded.value = true
    }
  }

  async function unlock(code: string): Promise<boolean> {
    unlocking.value = true
    error.value = null
    try {
      await unlockSession(code)
      await load()
      return unlocked.value
    } catch (e) {
      error.value = e instanceof ApiError ? e.message : 'Не удалось проверить код'
      return false
    } finally {
      unlocking.value = false
    }
  }

  return { loaded, unlocked, accessRequired, llmProvider, asrEnabled, error, unlocking, load, unlock }
})
