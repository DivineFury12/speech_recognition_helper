# Голосовой ассистент для задач и встреч

## Локальный запуск

Нужны Docker Compose и Node.js с npm. Для GPU-варианта Docker должен иметь доступ к NVIDIA GPU.

1. Подготовьте настройки бэкенда в PowerShell:

   ```powershell
   Copy-Item backend/.env.example backend/.env
   ```

   Укажите в `backend/.env` свой `GIGACHAT_AUTH_KEY` и доступную вам модель `GIGACHAT_MODEL`.

2. Запустите бэкенд и PostgreSQL:

   ```powershell
   cd backend
   docker compose up -d --build
   ```

   Если GPU нет, используйте `docker compose -f docker-compose.cpu.yml up -d --build`. При первом запуске скачиваются веса моделей.

3. В другом терминале запустите фронтенд из корня проекта:

   ```powershell
   Copy-Item frontend/.env.example frontend/.env
   # В frontend/.env установите VITE_USE_MOCKS=false
   cd frontend
   npm ci
   npm run dev
   ```

Откройте [приложение](http://localhost:5173). [Проверка бэкенда](http://localhost:8000/health) и [Swagger](http://localhost:8000/docs) доступны отдельно. Дополнительные настройки описаны в [README бэкенда](backend/README.md).
