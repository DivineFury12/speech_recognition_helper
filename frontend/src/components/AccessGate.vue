<script setup lang="ts">
import { ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useSessionStore } from '@/stores/session'

const session = useSessionStore()
const { error, unlocking } = storeToRefs(session)
const code = ref('')

async function submit() {
  if (!code.value.trim()) return
  if (await session.unlock(code.value)) code.value = ''
}
</script>

<template>
  <section class="gate panel" aria-labelledby="gate-title">
    <h2 id="gate-title">Нужен код доступа</h2>
    <p class="muted">Прототип закрыт кодом, чтобы посторонние не тратили лимиты модели. Код выдаёт владелец стенда.</p>
    <form class="gate-form" @submit.prevent="submit">
      <input
        v-model="code"
        type="password"
        class="field wide"
        autocomplete="current-password"
        aria-label="Код доступа"
        placeholder="Код доступа"
        autofocus
      />
      <button type="submit" class="send" :disabled="unlocking || !code.trim()">
        {{ unlocking ? 'Проверяю…' : 'Войти' }}
      </button>
    </form>
    <p v-if="error" class="banner error" role="alert">{{ error }}</p>
  </section>
</template>
