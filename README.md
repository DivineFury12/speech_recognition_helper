# Голосовой ассистент для задач и встреч

> **Это тестовая экспериментальная ветка `experimental/gigachat-backend`.**
> Здесь бэкенд разбирает команды через GigaChat. Ветка нужна, чтобы проверить
> подход, прежде чем что-то переносить в `main`. Переключиться обратно на
> локальную модель можно одной переменной окружения (см. «Смена модели»).

## Что сделано

**Бэкенд (`backend/`) — все ручки из `openapi.yaml`:**

| Ручка | Что делает |
| --- | --- |
| `POST /v1/assistant/messages` | Текстовая команда → ответ ассистента (`AssistantTurn`) |
| `POST /v1/assistant/audio` | Голос → GigaAM → та же логика, что у текста |
| `GET /v1/entries` | Записи за период (`date_from`, `date_to`, фильтры `kind`, `status`) |
| `GET /v1/entries/search` | Поиск записей по словам с учётом окончаний, без дословного совпадения |
| `POST /v1/entries`, `GET/PATCH/DELETE /v1/entries/{id}` | Прямая работа с записями |
| `GET /v1/conversations/{id}/messages` | История диалога |
| `GET /health`, `POST /transcribe`, `POST /transcribe_longform` | Как были, распознавание речи GigaAM |
| `POST /v1/speech/synthesize` | Пока отвечает 501: озвучивание не сделано |

Что умеет ассистент:

- создаёт задачи, напоминания и встречи: «Напомни завтра в 10 утра позвонить Ивану»;
- переспрашивает, если чего-то не хватает: «Поставь встречу с Иваном завтра» → «На какое время?» → «В 15:00»;
- показывает записи за период: «Что у меня сегодня / на неделе / на следующей неделе?»;
- ищет старые записи: «Найди задачу про презентацию»;
- отменяет уточнение по слову «отмена».

Как устроено:

1. Голос распознаёт GigaAM локально, как и раньше.
2. Текст разбирает модель (GigaChat). Она возвращает JSON: намерение, тип записи, название, дату и время словами.
3. Даты и время считает код по часовому поясу пользователя. Модель только находит фразу «завтра в 3 дня».
4. Если модель недоступна, не настроена или вернула мусор, тот же запрос разбирают правила на регулярках. Ассистент продолжает работать.
5. Записи, диалоги и черновики уточнений лежат в SQLite (`DB_PATH`).

**Фронтенд (`frontend/`)** — Vue 3 + TypeScript + Vite, тот же, что в `main`: чат, микрофон до 25 с, список записей на сегодня и неделю, моки MSW.

## Смена модели

Модель выбирает переменная `LLM_PROVIDER`:

| Значение | Что используется |
| --- | --- |
| `gigachat` | GigaChat API, нужен `GIGACHAT_CREDENTIALS` |
| `local` | Локальная Qwen из `backend/ml.py`, нужен `pip install -r requirements-local.txt` |
| `rules` | Только правила, без модели |
| не задано | `gigachat`, если задан ключ, иначе `rules` |

Все провайдеры реализуют один интерфейс `complete(system, user) -> str`
(`backend/assistant/llm/`). Чтобы подключить другую модель, достаточно добавить
файл с классом и строку в `create_provider`.

## Запуск

### Бэкенд локально (без Docker)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env          # и вписать ключ в GIGACHAT_CREDENTIALS
uvicorn app:app --host 127.0.0.1 --port 8000
```

`backend/.env` подхватывается автоматически, переменные окружения важнее него.
Ключ можно назвать `GIGACHAT_CREDENTIALS` или `GIGAKEY`.

В логе при старте видно, что включилось: `LLM provider: gigachat` или `rules`.
Без ключа сервер работает на правилах. Если GigaAM не установлен, сервер
стартует без распознавания речи: текст работает, голосовые ручки отвечают 503.

### Голос локально (Windows, NVIDIA)

Проверено на RTX 2060 6 ГБ: GigaAM на `cuda:0`, фраза распознаётся за ~1 с.

```powershell
winget install Gyan.FFmpeg          # браузер присылает webm/opus, без ffmpeg не декодировать
# перезапустить терминал, чтобы ffmpeg попал в PATH
cd backend
.venv\Scripts\activate
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
"torch==2.10.0+cu126`ntorchaudio==2.10.0+cu126" | Out-File -Encoding ascii constraints.txt
pip install -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu126 "gigaam[longform] @ git+https://github.com/salute-developers/GigaAM.git"
del constraints.txt
```

`constraints.txt` нужен, чтобы установка GigaAM не заменила CUDA-сборку torch на
CPU-версию из PyPI. Без видеокарты ставьте torch с `--index-url .../whl/cpu`.

При первом старте сервер скачает веса GigaAM (~430 МБ) в `backend/.cache/gigaam`.
Готовность: `GET /health` вернёт `"device": "cuda:0"`.

### Бэкенд в Docker (с распознаванием речи)

```bash
cd backend
set GIGACHAT_CREDENTIALS=<ключ авторизации>
docker compose -f docker-compose.cpu.yml up --build   # или docker compose up --build для GPU
```

Ключ Compose берёт из `backend/.env` (нужен Docker Compose 2.24+). Записи хранятся
в томе `assistant-data`, веса моделей — в `model-cache`.

### Фронтенд

```bash
cd frontend
npm install
npm run dev        # http://127.0.0.1:5173
```

В `frontend/.env.local` поставьте `VITE_USE_MOCKS=false`, чтобы ходить в настоящий бэкенд.
Dev-сервер проксирует `/v1` и `/health` на `127.0.0.1:8000`, CORS не нужен.

### Тесты

```bash
cd backend
pytest
```

73 теста: разбор дат и времени, сценарии ассистента (в том числе уточнения и
отказ модели), HTTP-контракт. Модель в тестах подменяется заглушкой, ключ не нужен.

## GigaChat: где взять ключ

1. Зарегистрироваться на [developers.sber.ru](https://developers.sber.ru/studio) и создать проект GigaChat API.
2. Скопировать **ключ авторизации** (Authorization key) в `GIGACHAT_CREDENTIALS` в `backend/.env`.
3. Для физлиц `GIGACHAT_SCOPE=GIGACHAT_API_PERS`, для юрлиц — `GIGACHAT_API_B2B` или `GIGACHAT_API_CORP`.
4. Модель — `GigaChat-2` (Lite, бесплатный пакет). Имя `GigaChat` API этого ключа не принимает.
5. GigaChat использует сертификат НУЦ Минцифры. Публичные корневой и промежуточный
   сертификаты лежат в `backend/certs/russian_trusted_ca.pem` и подключаются сами.
   Промежуточный действует до марта 2027, потом его нужно обновить с
   [gosuslugi.ru/crt](https://www.gosuslugi.ru/crt).

Ключ живёт только в окружении бэкенда, во фронт и в git он не попадает.

## Что не сделано и что проверить

- **GigaChat проверен через HTTP** на 10 фразах (создание, уточнение, отмена,
  «через 2 часа», поиск, списки): все разобраны моделью, без отката на правила,
  ответ около 0,5 с. На живых формулировках промпт в `backend/assistant/nlu.py`
  может потребовать доводки.
- **Голос проверен локально** на RTX 2060: синтезированные русские фразы в webm/opus
  → GigaAM → GigaChat, включая уточнение по `conversation_id`. С живым микрофоном
  в браузере и в Docker не проверялся.
- Поиск пока по словам (с учётом окончаний). Эмбеддинги FRIDA из `ml.py` можно подключить следующим шагом.
- Озвучивание ответа (`/v1/speech/synthesize`) не реализовано.
- Один пользователь, без авторизации, как и предполагает контракт.
