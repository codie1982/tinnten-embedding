# Python base image
FROM python:3.11

# Çalışma dizini
WORKDIR /app

# Bağımlılık dosyasını kopyala
COPY requirements.txt ./

# Bağımlılıkları yükle
RUN pip install -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Uygulamanın dışa açılacağı port
EXPOSE 5003

# Production: Gunicorn ile başlat
CMD ["python", "embedding.py"]