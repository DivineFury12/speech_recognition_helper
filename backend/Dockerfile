FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    XDG_CACHE_HOME=/cache \
    DB_PATH=/data/assistant.db

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --break-system-packages -r requirements.txt \
 && python -m pip install --break-system-packages "gigaam[longform] @ git+https://github.com/salute-developers/GigaAM.git"

COPY app.py ml.py ./
COPY assistant ./assistant

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
