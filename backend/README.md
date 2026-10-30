# Backend

`api` принимает HTTP-запросы, ведёт диалог и хранит записи в PostgreSQL. Он отправляет аудио, запросы эмбеддингов и озвучки во внутренний сервис `inference`. Внешняя модель GigaChat вызывается только из `api`.

Основной способ запуска из этой папки: `docker compose up -d --build`. Публичный релиз: `docker compose -f docker-compose.yml -f compose.release.yml up -d --build --remove-orphans` после настройки `backend/.env`. `run.ps1` — необязательная Windows-обёртка для тех же команд. Подробнее — в [основном README](../README.md).

Проверка: `docker compose ps`, `docker compose logs -f api inference`, [localhost:8080/health](http://localhost:8080/health). Миграции Alembic запускаются при старте `api`.

При смене `EMBEDDING_MODEL` пересоздайте оба сервиса и перестройте векторы: `docker compose exec api python -m src.reindex_embeddings`.
