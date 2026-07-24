FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# Chromium için gerekli sistem bağımlılıkları
RUN apt-get update && \
    apt-get install -y wget curl libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 libgtk-3-0 libasound2 && \
    rm -rf /var/lib/apt/lists/*

# Bağımlılık dosyası
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 100 -r requirements.txt

# HuggingFace model cache'i /opt/huggingface'te — bu dizin runtime'daki
# `embedding_data:/app/data` volume mount'unun ALTINDA DEĞİL, o yüzden build'de
# buraya bake edilen model runtime'da volume tarafından ÖRTÜLMEZ (RAG v2 KN2).
# HF_HOME set etmezsek modeller ~/.cache/huggingface'e iner ve her container
# yeniden yaratmada kaybolur (ilk sorgu ~2GB reranker'ı tekrar indirir).
ENV HF_HOME=/opt/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/huggingface

# Opsiyonel build-time reranker bake (RAG v2 — Faz 4).
# VARSAYILAN BOŞ = bake YOK (reranker kapali, image'a ~2GB eklenmez).
# Reranker acilirken determinizm icin bake et + revision PINLE:
#   docker build --build-arg RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
#                --build-arg RERANKER_REVISION=<commit-sha> .
# Yalniz model adi build'i tekrarlanabilir yapmaz — revision sart.
ARG RERANKER_MODEL=
ARG RERANKER_REVISION=
RUN if [ -n "$RERANKER_MODEL" ]; then \
      echo "Baking reranker $RERANKER_MODEL (revision=${RERANKER_REVISION:-none})" && \
      RM="$RERANKER_MODEL" RR="$RERANKER_REVISION" python -c "import os; from sentence_transformers import CrossEncoder; CrossEncoder(os.environ['RM'], revision=(os.environ['RR'] or None))"; \
    else \
      echo "RERANKER_MODEL bos — bake atlandi (reranker kapali)"; \
    fi

# Uygulama dosyalarını kopyala
COPY . .

EXPOSE 5003

# Production: yalnız HTTP API (arama). Ingest ayrı container'da `python worker.py`
# koşar (compose: tinnten-embedding-ingest) — Faz 0 ayrımı.
# --threads 8 (gthread): FAISS search + encode C++'ta GIL'i bıraktığı için istekler
# gerçekten paralelleşir; çoklu --workers KULLANILMAZ (her worker 714MB index'i ayrıca
# belleğe yükler ve EmbeddingEngine._lock process-local'dır).
# --timeout 600→120: uzun istekler gthread'de heartbeat'i bloklamaz; 120 sn üstü
# takılı istek artık worker'ı öldürüp kuyruğu temizler.
CMD ["gunicorn", "--bind", "0.0.0.0:5003", "--workers", "1", "--threads", "8", "--timeout", "120", "wsgi:app"]
