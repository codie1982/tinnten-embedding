FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Chromium için gerekli sistem bağımlılıkları
RUN apt-get update && \
    apt-get install -y wget curl libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 libgtk-3-0 libasound2 && \
    rm -rf /var/lib/apt/lists/*

# Bağımlılık dosyası
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

EXPOSE 5003

# Production: Gunicorn ile başlat
CMD ["gunicorn", "--bind", "0.0.0.0:5003", "--workers=2", "--timeout=600", "embedding:app"]
