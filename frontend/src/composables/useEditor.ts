import { ref } from 'vue'
import type { Entry } from '@/api/types'

const current = ref<Entry | null>(null)
const creating = ref(false)
const open = ref(false)

export function useEditor() {
  function edit(entry: Entry) {
    current.value = entry
    creating.value = false
    open.value = true
  }
  function create() {
    current.value = null
    creating.value = true
    open.value = true
  }
  function close() {
    open.value = false
  }
  return { current, creating, open, edit, create, close }
}
