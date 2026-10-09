#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
PROD=0
[[ "${1:-}" == "--prod" ]] && PROD=1

if [[ ! -x "$BACKEND/.venv/bin/python" ]]; then
  echo "==> Создаю backend/.venv"
  python3 -m venv "$BACKEND/.venv"
  "$BACKEND/.venv/bin/python" -m pip install -q -r "$BACKEND/requirements-dev.txt"
fi
if [[ ! -f "$BACKEND/.env" ]]; then
  cp "$BACKEND/.env.example" "$BACKEND/.env"
  echo "Создан backend/.env — впишите ключ GigaChat в GIGACHAT_CREDENTIALS"
fi
if [[ ! -d "$FRONTEND/node_modules" ]]; then
  echo "==> npm install"
  (cd "$FRONTEND" && npm install)
fi

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if [[ $PROD == 1 ]]; then
  (cd "$FRONTEND" && VITE_USE_MOCKS=false VITE_API_URL= npm run build)
  (cd "$BACKEND" && exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000) & pids+=($!)
  echo "Откройте http://127.0.0.1:8000"
else
  (cd "$BACKEND" && exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000) & pids+=($!)
  (cd "$FRONTEND" && exec npm run dev) & pids+=($!)
  echo "Откройте http://127.0.0.1:5173"
fi
wait
