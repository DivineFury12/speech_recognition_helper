FROM node:20-alpine AS frontend
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ENV VITE_USE_MOCKS=false VITE_API_URL=
RUN npm run build

FROM python:3.11-slim AS lite
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DB_PATH=/data/assistant.db FRONTEND_DIST=/app/frontend \
    VOICE_ENABLED=false EMBEDDING_MODEL=none
WORKDIR /app
COPY backend/requirements-lite.txt .
RUN pip install -r requirements-lite.txt
COPY backend/app.py backend/ml.py ./
COPY backend/assistant ./assistant
COPY backend/certs ./certs
COPY --from=frontend /src/dist ./frontend
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim AS cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models/hf \
    DB_PATH=/data/assistant.db FRONTEND_DIST=/app/frontend \
    COQUI_TOS_AGREED=1 TTS_MODEL_DIR=/models/xtts_v2 EMBEDDING_CACHE=/models/embeddings GIGAAM_CACHE=/models/gigaam
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt \
 && pip install "gigaam[longform] @ git+https://github.com/salute-developers/GigaAM.git"
COPY backend/app.py backend/ml.py ./
COPY backend/assistant ./assistant
COPY backend/certs ./certs
COPY backend/voices ./voices
RUN python -m assistant.models_download
COPY --from=frontend /src/dist ./frontend
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime AS gpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models/hf \
    DB_PATH=/data/assistant.db FRONTEND_DIST=/app/frontend \
    COQUI_TOS_AGREED=1 TTS_MODEL_DIR=/models/xtts_v2 EMBEDDING_CACHE=/models/embeddings GIGAAM_CACHE=/models/gigaam
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt .
RUN python -m pip install --break-system-packages -r requirements.txt \
 && python -m pip install --break-system-packages "gigaam[longform] @ git+https://github.com/salute-developers/GigaAM.git"
COPY backend/app.py backend/ml.py ./
COPY backend/assistant ./assistant
COPY backend/certs ./certs
COPY backend/voices ./voices
RUN python -m assistant.models_download
COPY --from=frontend /src/dist ./frontend
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
