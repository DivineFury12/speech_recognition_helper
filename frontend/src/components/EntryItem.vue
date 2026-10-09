<script setup lang="ts">
import { computed } from 'vue'
import type { Entry } from '@/api/types'
import { formatDate } from '@/utils/time'

const props = defineProps<{ entry: Entry; compact?: boolean }>()

const KIND_LABEL: Record<Entry['kind'], string> = { task: 'Задача', reminder: 'Напоминание', meeting: 'Встреча' }

const when = computed(() => {
  const e = props.entry
  const parts: string[] = []
  if (e.date) parts.push(formatDate(e.date))
  if (e.time) parts.push(e.time)
  if (e.duration_minutes) parts.push(`${e.duration_minutes} мин`)
  return parts.join(' · ') || 'без срока'
})
</script>

<template>
  <li class="entry" :class="[entry.kind, { compact, done: entry.status === 'completed' }]">
    <span class="kind">{{ KIND_LABEL[entry.kind] }}</span>
    <span class="title">{{ entry.title }}</span>
    <span class="when">{{ when }}</span>
  </li>
</template>
