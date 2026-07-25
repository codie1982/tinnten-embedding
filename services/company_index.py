"""
Per-company FAISS index yolu — TEK KAYNAK.

Bu mantık daha önce ÜÇ yerde ayrı ayrı kopyalanmıştı (`app.py`,
`workers/ingest_worker.py`, `scripts/migrate_global_to_per_company_faiss.py`).
Kopyalar sessizce ayrışabilir ve ayrıştıklarında hata GÖRÜNMEZ: web bir dosyaya
yazar, worker başka bir dosyaya yazar, arama üçüncüsünü okur — hiçbiri istisna
atmaz, yalnızca sonuçlar kaybolur. O yüzden sanitizasyon + yol üretimi burada
tektir ve üç çağıran da buradan import eder.

Saf fonksiyonlar: faiss/mongo/flask bağımlılığı YOK, doğrudan test edilebilir.
"""
from __future__ import annotations

import os
import re

# Firma index'lerinin global index'in yanında toplandığı alt dizin.
COMPANY_INDEX_DIRNAME = "company"

# Per-company FAISS — SABİT, env'den okunmaz.
#
# Eskiden `PER_COMPANY_FAISS_ENABLED` env'iydi ve app.py ile ingest_worker.py onu
# AYRI AYRI okuyordu. İki process farklı değer görürse (biri restart edilip
# diğeri edilmezse) yazma ve okuma farklı dosyalara gider; sonuç istisna değil,
# SESSİZ veri kaybıdır. Tek sabit bu sınıf hatayı tamamen kaldırır.
#
# Prod'da bu bayrak hiç açılmamıştı (.env / .env.prod / docker-compose'un
# hiçbirinde yoktu) — firma bazlı index kodda vardı, fiziksel olarak yoktu.
PER_COMPANY_FAISS_ENABLED = True

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def sanitize_company_id(company_id: str | None) -> str:
    """
    company_id'yi dosya-adı-güvenli hâle getirir (path traversal savunması).

    `..`, `/`, NUL ve benzeri her şey düşer; geriye yalnız `[A-Za-z0-9_-]` kalır.
    Tamamen elenirse "" döner — çağıran bunu "firma yok" olarak yorumlamalı.
    """
    return _UNSAFE_CHARS.sub("", str(company_id or "").strip())


def company_index_path(base_index_path: str, company_id: str | None) -> str:
    """
    Firma-scope chunk FAISS index yolu.

    company_id yoksa/boşsa GLOBAL index yolunu döner (personal / firmasız içerik
    orada kalır). Aksi halde `<global_index_dizini>/company/<companyId>.index`.
    """
    cid = sanitize_company_id(company_id)
    if not cid:
        return base_index_path
    base_dir = os.path.dirname(base_index_path) or "."
    return os.path.join(base_dir, COMPANY_INDEX_DIRNAME, f"{cid}.index")


def resolve_chunk_index_path(getenv=os.getenv, default: str = "faiss.index") -> str:
    """
    Global chunk index yolunu env'den çözer.

    Anahtar sırası web (`app.py`) ve ingest worker için AYNI olmak ZORUNDA.
    Daha önce worker yalnız CHUNK_INDEX_PATH/FAISS_INDEX_PATH'e bakıyordu; ortamda
    yalnız CONTENT_INDEX_PATH set edilseydi web ile worker FARKLI dosyalara
    yazardı ve bu sessizce "indexlendi ama arama 0 dönüyor" olarak görünürdü.
    """
    for key in ("CHUNK_INDEX_PATH", "CONTENT_INDEX_PATH", "FAISS_INDEX_PATH", "INDEX_PATH"):
        value = getenv(key)
        if value and value.strip():
            return value.strip()
    return default
