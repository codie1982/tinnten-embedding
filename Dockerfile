FROM python:3.11

WORKDIR /app

# Chromium için gerekli sistem bağımlılıkları
RUN apt-get update && \
    apt-get install -y wget curl libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 libgtk-3-0 libasound2 && \
    rm -rf /var/lib/apt/lists/*

# Bağımlılık dosyası
COPY requirements.txt ./

# Python bağımlılıklarını kur
RUN pip install --no-cache-dir -r requirements.txt

# Crawl4AI kurulumunu yap
RUN crawl4ai-setup

# !!! Asıl eksik kısım chromium install
RUN playwright install chromium

# Opsiyonel: Kurulum doğrulama
RUN crawl4ai-doctor

# Uygulama dosyalarını kopyala
COPY . .

EXPOSE 5003

# Production: Gunicorn ile başlat
CMD ["gunicorn", "--bind", "0.0.0.0:5003", "--workers=2", "--timeout=600", "embedding:app"]