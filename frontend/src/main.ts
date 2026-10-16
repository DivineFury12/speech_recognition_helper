import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import '@fontsource-variable/inter'
import './style.css'
import './app.css'
import { initTheme } from './composables/useTheme'

initTheme()

async function bootstrap() {
  if (import.meta.env.VITE_USE_MOCKS === 'true') {
    const { worker } = await import('./mocks/browser')
    await worker.start({ onUnhandledRequest: 'bypass' })
  }
  createApp(App).use(createPinia()).mount('#app')
  if (import.meta.env.PROD && import.meta.env.VITE_USE_MOCKS !== 'true' && 'serviceWorker' in navigator) {
    window.addEventListener('load', () => void navigator.serviceWorker.register('/sw.js').catch(() => undefined))
  }
}

void bootstrap()
