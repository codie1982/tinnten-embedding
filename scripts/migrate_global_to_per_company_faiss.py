"""
Adım 3 — Global chunk FAISS → per-company FAISS MIGRATION (vector-copy).

PER_COMPANY_FAISS_ENABLED açılmadan ÖNCE çalıştırılır. Global index'teki her chunk'ın
vektörünü `index.reconstruct(faiss_id)` ile OKUR (re-embed YOK → deterministik + hızlı;
model sabit) ve chunk'ın firmasına (`metadata.companyId`, yoksa top-level `company_id`)
göre `company/<companyId>.index`'e AYNI faiss_id ile yazar. faiss_id global benzersiz
olduğu için çakışma yok; MongoDB chunk kayıtlarına DOKUNMAZ; global index'i SİLMEZ
(rollback güvenli — bayrağı geri kapatınca okuma/yazım global'e döner).

Firma-sız (personal) chunk'lar global'de KALIR (taşınmaz).

Kullanım:
    python scripts/migrate_global_to_per_company_faiss.py            # DRY-RUN (rapor)
    python scripts/migrate_global_to_per_company_faiss.py --apply    # gerçek yazım
    python scripts/migrate_global_to_per_company_faiss.py --apply --company <id>
    # --global-index yolu env CHUNK_INDEX_PATH'ten çözülür; --mongo-uri / --db override.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Saf yardımcılar (faiss/mongo bağımsız — test edilebilir)
# ---------------------------------------------------------------------------
def resolve_company_id(chunk: Dict[str, Any]) -> str | None:
    """
    Chunk'ın firmasını çözer: önce metadata.companyId (per-sayfa/fetcher chunk'ları),
    sonra top-level company_id (legacy tek-doc). Firma yoksa None (personal → global'de kalır).
    """
    md = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    cid = md.get("companyId") or md.get("company_id") or chunk.get("company_id")
    cid = str(cid).strip() if cid else ""
    return cid or None


def sanitize_company_id(company_id: str) -> str:
    """app.py `_company_chunk_index_path` ile AYNI dosya-adı-güvenli dönüşüm."""
    return re.sub(r"[^A-Za-z0-9_-]", "", str(company_id or "").strip())


def group_faiss_ids_by_company(chunks: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, List[int]], int, int]:
    """
    chunk kayıtlarını firma → [faiss_id] olarak gruplar.
    Dönüş: (grup, company_less_sayisi, faiss_id_yok_sayisi).
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    company_less = 0
    no_faiss = 0
    for c in chunks:
        fid = c.get("faiss_id")
        if not isinstance(fid, (int, float)):
            no_faiss += 1
            continue
        cid = resolve_company_id(c)
        if not cid:
            company_less += 1
            continue
        groups[cid].append(int(fid))
    return dict(groups), company_less, no_faiss


def company_index_path(global_index_path: str, company_id: str) -> str | None:
    cid = sanitize_company_id(company_id)
    if not cid:
        return None
    base_dir = os.path.dirname(global_index_path) or "."
    return os.path.join(base_dir, "company", f"{cid}.index")


# ---------------------------------------------------------------------------
# faiss/mongo bağımlı (yalnız çalışma zamanı)
# ---------------------------------------------------------------------------
def _load_chunks(mongo_uri: str, db_name: str, company: str | None):
    from pymongo import MongoClient

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=30000)
    coll = client[db_name]["embedding_chunks"]
    query: Dict[str, Any] = {}
    if company:
        query = {"$or": [{"metadata.companyId": company}, {"company_id": company}]}
    proj = {"_id": 0, "faiss_id": 1, "company_id": 1, "metadata.companyId": 1, "metadata.company_id": 1}
    return list(coll.find(query, proj))


def _prepare_reconstruct(index):
    """IndexIDMap → make_direct_map gerekir; IndexIDMap2 doğrudan reconstruct eder."""
    try:
        index.reconstruct(int(index.id_map.at(0))) if hasattr(index, "id_map") else index.reconstruct(0)
        return
    except Exception:
        pass
    try:
        index.make_direct_map()
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] make_direct_map başarısız: {exc}")


def _build_company_index(global_index, faiss_ids: List[int], dim: int):
    import faiss
    import numpy as np

    base = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIDMap2(base)
    vecs: List[Any] = []
    ids: List[int] = []
    missing = 0
    for fid in faiss_ids:
        try:
            vecs.append(global_index.reconstruct(int(fid)))
            ids.append(int(fid))
        except Exception:
            missing += 1
    if ids:
        arr = np.array(vecs, dtype="float32")
        idx.add_with_ids(arr, np.array(ids, dtype="int64"))
    return idx, len(ids), missing


def _write_index(index, path: str) -> None:
    import faiss

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    faiss.write_index(index, tmp)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Global→per-company FAISS vector-copy migration")
    ap.add_argument("--apply", action="store_true", help="Gerçek yazım (yoksa dry-run)")
    ap.add_argument("--company", default=None, help="Yalnız bu companyId")
    ap.add_argument("--global-index", default=os.getenv("CHUNK_INDEX_PATH") or "faiss.index")
    ap.add_argument("--mongo-uri", default=os.getenv("MONGO_URI") or os.getenv("FETCHER_MONGO_URI"))
    ap.add_argument("--db", default=os.getenv("EMBED_DB_NAME") or "tinnten-embedding")
    args = ap.parse_args()

    if not args.mongo_uri:
        print("HATA: MONGO_URI gerekli (env veya --mongo-uri).")
        return 2
    if not os.path.exists(args.global_index):
        print(f"HATA: global index yok: {args.global_index}")
        return 2

    import faiss

    print(f"Global index: {args.global_index}")
    global_index = faiss.read_index(args.global_index)
    dim = int(global_index.d)
    print(f"  d={dim} ntotal={global_index.ntotal} type={type(global_index).__name__}")

    chunks = _load_chunks(args.mongo_uri, args.db, args.company)
    groups, company_less, no_faiss = group_faiss_ids_by_company(chunks)
    print(
        f"Chunk: toplam={len(chunks)} firma={len(groups)} "
        f"firma_sız(personal→global kalır)={company_less} faiss_id_yok={no_faiss}"
    )
    if not groups:
        print("Taşınacak firma-chunk'ı yok.")
        return 0

    if args.apply:
        _prepare_reconstruct(global_index)

    total_written = 0
    total_missing = 0
    for cid, fids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        path = company_index_path(args.global_index, cid)
        if not path:
            print(f"  [skip] geçersiz companyId: {cid!r}")
            continue
        if not args.apply:
            print(f"  [dry] {cid}: {len(fids)} chunk → {path}")
            continue
        idx, written, missing = _build_company_index(global_index, fids, dim)
        _write_index(idx, path)
        total_written += written
        total_missing += missing
        print(f"  [ok] {cid}: {written} yazıldı (eksik reconstruct={missing}) → {path}")

    if args.apply:
        print(f"\nBİTTİ: {len(groups)} firma, {total_written} chunk yazıldı, {total_missing} eksik.")
        print("Global index DOKUNULMADI (rollback güvenli). Şimdi PER_COMPANY_FAISS_ENABLED=1 + embedding web/worker BİRLİKTE restart.")
    else:
        print("\nDRY-RUN — yazım yapılmadı. Gerçek çalıştırma için --apply ekle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
