# Голосовой ассистент для задач и встреч

Создаёт, ищет и изменяет задачи, встречи и напоминания текстом или голосом. GigaAM распознаёт речь локально; GigaChat разбирает команды; PostgreSQL хранит записи и историю диалога.

## Локальный запуск без GPU

Нужен Docker. Скопируйте `backend/.env.example` в `backend/.env` и заполните `GIGACHAT_AUTH_KEY`. Запуск из корня проекта:

```bash
cd backend
docker compose up -d --build
```

Откройте [localhost:8080](http://localhost:8080). Первый запуск собирает образы и загружает модели.

`run.ps1` не обязателен: это короткая команда для Windows, которая проверяет Docker и запускает тот же Compose. Из корня проекта можно выполнить `.\run.ps1`; для режима без распознавания речи — `.\run.ps1 -NoVoice`.

## Публикация на одном сервере

Нужны Docker, домен с DNS-записью на сервер и открытые порты 80/443. В `backend/.env` заполните `APP_DOMAIN`, `GIGACHAT_AUTH_KEY`, `POSTGRES_PASSWORD`, `SESSION_SECRET` и `ACCESS_CODE`. `SESSION_SECRET` должен быть случайной длинной строкой. Запуск из корня проекта:

```bash
cd backend
docker compose -f docker-compose.yml -f compose.release.yml up -d --build --remove-orphans
```

На Windows из корня проекта можно вместо этой команды выполнить `.\run.ps1 -Release`. Скрипт не нужен на сервере. Caddy выдаст HTTPS-сертификат для указанного домена. Интерфейс доступен по `https://<APP_DOMAIN>`.

Контейнеры: `frontend` (веб), `api` (команды, GigaChat, БД), `inference` (GigaAM, эмбеддинги, озвучка на CPU), `db` (PostgreSQL). В интернет открываются только 80/443; локально также доступен 8080 на `127.0.0.1`. Данные БД и кэш моделей лежат в томах Docker.
