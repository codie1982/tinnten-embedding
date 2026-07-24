"""HTTP (arama/API) giriş noktası — Faz 0 ayrımı.

Ingest worker'ı ARTIK burada başlamaz: RabbitMQ consumer'ı ayrı bir
container'da `python worker.py` olarak koşar (docker-compose:
tinnten-embedding-ingest). Eskiden ikisi aynı gunicorn process'indeydi ve
CPU-bound encode() + tek sync worker yüzünden HTTP istekleri kuyrukta
22-28 sn bekliyordu (401 auth reddi bile) — arama timeout'larının kök
nedeni buydu.

İki process aynı FAISS dosyalarını paylaşır; yazma `_write_lock()` (fcntl)
ile cross-process güvenli, `_save_index()` atomik (tmp + os.replace),
okuyan `reload_if_updated()` ile mtime'dan tazelenir.
"""
from app import app, warmup_embedding_models

warmup_embedding_models()

if __name__ == "__main__":
    app.run()
