# GigaAM-v3 ASR backend

## Запуск

GPU (Docker Desktop с поддержкой NVIDIA GPU или NVIDIA Container Toolkit на Linux):

    docker compose up --build

CPU:

    docker compose -f docker-compose.cpu.yml up --build

При первом старте скачиваются веса модели, подождите пару минут.
Для GPU Docker Compose требует доступную CUDA и завершит запуск с ошибкой, если
модель не удалось разместить на видеокарте. Веса сохраняются в томе `model-cache`.

## Проверка

    curl http://localhost:8000/health
    curl -F "file=@example.wav" http://localhost:8000/transcribe

Swagger: http://localhost:8000/docs
В ответе `/health` поле `device` должно быть `cuda:0` при запуске GPU-варианта.

## Эндпоинты

- GET  /health
- POST /transcribe            — файл до ~25 сек, ответ {"text": "..."}
- POST /transcribe_longform   — длинное аудио, нужен HF_TOKEN
                                (HF_TOKEN=xxx docker compose up --build)

## Смена модели

Переменная GIGAAM_MODEL: v3_e2e_rnnt (по умолчанию), v3_e2e_ctc,
v3_rnnt, v3_ctc.
