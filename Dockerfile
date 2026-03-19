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

# Uygulama dosyalarını kopyala
COPY . .

EXPOSE 5003

# Production: API + background workers through wsgi entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:5003", "--workers", "1", "--timeout", "600", "wsgi:app"]
