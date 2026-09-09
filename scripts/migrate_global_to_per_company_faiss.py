"""
Adım 3 — Global chunk FAISS → per-company FAISS MIGRATION (vector-copy).

Per-company FAISS artık KOD SABİTİ (services/company_index.py) — bu script'in
"bayrağı açmadan önce koş" ön koşulu ortadan kalktı. Bunun yerine okuma yolu
`PER_COMPANY_FAISS_DUAL_READ` ile korunuyor: firma index'i henüz yokken arama
global'e düşer. Yani migration deploy'dan SONRA da koşulabilir; koşana kadar
firmalar global'den okumaya devam eder, retrieval kesilmez.

Global index'teki her GEÇERLİ chunk'ın
vektörünü `index.reconstruct(faiss_id)` ile OKUR (re-embed YOK → deterministik + hızlı;
model sabit) ve chunk'ın firmasına (`metadata.companyId`, yoksa top-level `company_id`)
göre `company/<companyId>.index`'e AYNI faiss_id ile yazar. faiss_id global benzersiz
olduğu için çakışma yok; MongoDB chunk kayıtlarına DOKUNMAZ; global index'i SİLMEZ
(rollback güvenli — bayrağı geri kapatınca okuma/yazım global'e döner).

Taşıma öncesinde chunk'ın canlı `embedding_documents` üst kaydı, şirket eşleşmesi ve
aktif ingest sürümü doğrulanır. Üst kaydı bulunmayan, şirketi yalnız chunk'a sonradan
yazılmış, kaldırılmış/devre dışı veya eski sürüm chunk'lar şirket indeksine TAŞINMAZ.
Firma-sız (personal) chunk'lar da global'de KALIR.

Kullanım:
    python scripts/migrate_global_to_per_company_faiss.py            # DRY-RUN (rapor)
    python scripts/migrate_global_to_per_company_faiss.py --apply    # gerçek yazım
    python scripts/migrate_global_to_per_company_faiss.py --apply --company <id>
    # --global-index yolu env CHUNK_INDEX_PATH'ten çözülür; --mongo-uri / --db override.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.company_index import (  # noqa: E402
    company_index_path as _company_index_path,
    resolve_chunk_index_path,
    sanitize_company_id as _sanitize_company_id,
)


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
    """app.py / ingest_worker ile AYNI dosya-adı-güvenli dönüşüm (tek kaynak)."""
    return _sanitize_company_id(company_id)


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


def filter_migratable_chunks(
    chunks: Iterable[Dict[str, Any]],
    documents: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Yalnız canlı üst kaydı ve aktif sürümü doğrulanan chunk'ları döndürür.

    Chunk üzerindeki ``company_id`` tek başına sahiplik kanıtı değildir. Eski/global
    verilerde bu alan toplu işlemlerle sonradan yazılmış olabilir. Bu yüzden migration
    ancak üst doküman da aynı şirketi gösteriyorsa chunk'ı şirket FAISS'ine taşır.
    """
    documents_by_id: Dict[str, Dict[str, Any]] = {}
    for document in documents:
        doc_id = str(document.get("doc_id") or "").strip()
        if doc_id:
            documents_by_id[doc_id] = document

    eligible: List[Dict[str, Any]] = []
    stats = {
        "parentless": 0,
        "inactive_parent": 0,
        "parent_company_missing": 0,
        "company_mismatch": 0,
        "stale_version": 0,
    }
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id") or "").strip()
        document = documents_by_id.get(doc_id)
        if not document:
            stats["parentless"] += 1
            continue
        if document.get("status") in {"removed", "disabled"}:
            stats["inactive_parent"] += 1
            continue

        chunk_company_id = resolve_company_id(chunk)
        document_company_id = resolve_company_id(document)
        if not document_company_id:
            stats["parent_company_missing"] += 1
            continue
        if chunk_company_id != document_company_id:
            stats["company_mismatch"] += 1
            continue

        active_version = document.get("active_ingest_version")
        if active_version and chunk.get("ingest_version") != active_version:
            stats["stale_version"] += 1
            continue
        eligible.append(chunk)
    return eligible, stats


def company_index_path(global_index_path: str, company_id: str) -> str | None:
    """
    Firma index yolu — servis koduyla AYNI fonksiyondan üretilir.

    Sözleşme farkı bilinçli: burada geçersiz/boş companyId `None` döner ki
    migration o firmayı ATLASIN. Servis tarafındaki paylaşılan fonksiyon aynı
    durumda global yola düşer; migration'da bu YANLIŞ olurdu (firma vektörlerini
    global index'in üzerine yazardık).
    """
    cid = sanitize_company_id(company_id)
    if not cid:
        return None
    return _company_index_path(global_index_path, cid)


# ---------------------------------------------------------------------------
# faiss/mongo bağımlı (yalnız çalışma zamanı)
# ---------------------------------------------------------------------------
def _load_chunks(
    mongo_uri: str,
    chunk_db_name: str,
    document_db_name: str,
    company: str | None,
):
    from pymongo import MongoClient

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=30000)
    chunk_coll = client[chunk_db_name]["embedding_chunks"]
    document_coll = client[document_db_name]["embedding_documents"]
    query: Dict[str, Any] = {}
    if company:
        query = {"$or": [{"metadata.companyId": company}, {"company_id": company}]}
    chunk_projection = {
        "_id": 0,
        "doc_id": 1,
        "faiss_id": 1,
        "ingest_version": 1,
        "company_id": 1,
        "metadata.companyId": 1,
        "metadata.company_id": 1,
    }
    chunks = list(chunk_coll.find(query, chunk_projection))
    doc_ids = sorted({str(chunk.get("doc_id")) for chunk in chunks if chunk.get("doc_id")})
    documents: List[Dict[str, Any]] = []
    document_projection = {
        "_id": 0,
        "doc_id": 1,
        "status": 1,
        "active_ingest_version": 1,
        "company_id": 1,
        "metadata.companyId": 1,
        "metadata.company_id": 1,
    }
    # Tek bir dev `$in` sorgusunun MongoDB BSON sınırına takılmasını önle.
    for offset in range(0, len(doc_ids), 5000):
        documents.extend(
            document_coll.find(
                {"doc_id": {"$in": doc_ids[offset : offset + 5000]}},
                document_projection,
            )
        )
    eligible, stats = filter_migratable_chunks(chunks, documents)
    return eligible, {"candidates": len(chunks), **stats}


class MigrationAbort(RuntimeError):
    """Sessizce bozuk index yazmak yerine göründüğü yerde durmak için."""


def _prepare_reconstruct(index) -> None:
    """
    Index'in faiss_id ile `reconstruct` yapabildiğini DOĞRULAR; yapamıyorsa durur.

    Eski sürüm burada `index.make_direct_map()` çağırıyordu — böyle bir metot
    IndexIDMap/IndexIDMap2 üzerinde YOKTUR (doğrusu `construct_rev_map`) ve hata
    yutuluyordu. Sonuç: düz bir IndexIDMap ile çalışıldığında her reconstruct
    patlıyor, `_build_company_index` hepsini yutuyor ve script her firma için BOŞ
    index yazıp `exit 0` ile "başarılı" dönüyordu. Artık sert hata veriyoruz.
    """
    import faiss

    inner = faiss.downcast_index(index) if hasattr(faiss, "downcast_index") else index
    kind = type(inner).__name__

    if not hasattr(inner, "id_map"):
        raise MigrationAbort(
            f"Global index tipi {kind} — faiss_id eşlemesi yok, id ile reconstruct edilemez. "
            "Migration bu index üzerinde ÇALIŞTIRILAMAZ."
        )
    if int(inner.ntotal) == 0:
        raise MigrationAbort("Global index boş (ntotal=0) — taşınacak vektör yok.")

    probe_id = int(inner.id_map.at(0))
    try:
        inner.reconstruct(probe_id)
        return
    except Exception:  # noqa: BLE001 — IndexIDMap2 için rev_map kurulmamış olabilir
        pass

    construct = getattr(inner, "construct_rev_map", None)
    if construct is None:
        raise MigrationAbort(
            f"Global index tipi {kind} id-bazlı reconstruct desteklemiyor "
            "(IndexIDMap2 bekleniyordu). Migration GÜVENLİ DEĞİL — durduruldu."
        )
    construct()
    try:
        inner.reconstruct(probe_id)
    except Exception as exc:  # noqa: BLE001
        raise MigrationAbort(
            f"construct_rev_map sonrası da reconstruct başarısız: {exc}"
        ) from exc


def _build_company_index(global_index, faiss_ids: List[int], dim: int, allow_missing: bool = False):
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
    if missing and not allow_missing:
        raise MigrationAbort(
            f"{missing}/{len(faiss_ids)} vektör reconstruct edilemedi. Eksik vektörle "
            "index yazmak sessiz veri kaybıdır — durduruldu (--allow-missing ile zorlanabilir)."
        )
    if not ids and faiss_ids:
        raise MigrationAbort(
            f"{len(faiss_ids)} chunk beklenirken hiçbiri okunamadı — BOŞ index yazılmayacak."
        )
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


def _marker_path(index_path: str) -> str:
    return f"{index_path}.migration.json"


def _migration_marker_check(index_path: str, force: bool) -> str | None:
    """
    Yazımı engelleyen bir sebep varsa açıklamasını döner, yoksa None.

    İki durumu engeller: (1) bu firma zaten migrate edilmiş — tekrar yazmak
    aradaki tüm ingest'i siler; (2) index dosyası var ama marker yok — onu bu
    script yazmamış demektir, üzerine yazmak veri kaybıdır.
    """
    marker = _marker_path(index_path)
    if os.path.exists(marker):
        if force:
            return None
        try:
            with open(marker, "r", encoding="utf-8") as fh:
                info = json.load(fh)
        except Exception:  # noqa: BLE001
            info = {}
        return (
            f"zaten migrate edilmiş ({info.get('completedAt', '?')}, "
            f"{info.get('written', '?')} vektör). Tekrar yazmak migration SONRASI "
            "eklenen vektörleri yok eder. Gerekiyorsa --force."
        )
    if os.path.exists(index_path) and not force:
        return (
            "index dosyası var ama migration marker'ı yok — bu dosyayı bu script "
            "yazmamış. Üzerine yazmak veri kaybı olur. Gerekiyorsa --force."
        )
    return None


def _write_migration_marker(index_path: str, source: str, written: int, missing: int, expected: int) -> None:
    payload = {
        "sourceIndexPath": source,
        "expected": expected,
        "written": written,
        "missing": missing,
        "completedAt": datetime.now(timezone.utc).isoformat(),
    }
    with open(_marker_path(index_path), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Global→per-company FAISS vector-copy migration")
    ap.add_argument("--apply", action="store_true", help="Gerçek yazım (yoksa dry-run)")
    ap.add_argument("--company", default=None, help="Yalnız bu companyId")
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="Reconstruct edilemeyen vektör olsa bile yaz (VERİ KAYBI riski)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Mevcut firma index'inin ÜZERİNE yaz (migration sonrası vektörleri YOK EDER)",
    )
    # Env anahtar sırası servis koduyla AYNI olmalı — aksi halde migration,
    # servisin gerçekte kullandığından BAŞKA bir global index'i okur.
    ap.add_argument("--global-index", default=resolve_chunk_index_path())
    ap.add_argument("--mongo-uri", default=os.getenv("MONGO_URI") or os.getenv("FETCHER_MONGO_URI"))
    ap.add_argument("--db", default=os.getenv("EMBED_DB_NAME") or "tinnten-embedding")
    ap.add_argument(
        "--document-db",
        default=os.getenv("EMBED_DOCUMENT_DB_NAME") or os.getenv("DB_TINNTEN") or "tinnten",
        help="embedding_documents koleksiyonunun bulunduğu ana veritabanı",
    )
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

    chunks, filtered = _load_chunks(args.mongo_uri, args.db, args.document_db, args.company)
    groups, company_less, no_faiss = group_faiss_ids_by_company(chunks)
    print(
        f"Chunk: aday={filtered['candidates']} geçerli={len(chunks)} firma={len(groups)} "
        f"firma_sız(personal→global kalır)={company_less} faiss_id_yok={no_faiss}"
    )
    print(
        "  elenen: "
        f"üst_kayıtsız={filtered['parentless']} "
        f"pasif_üst_kayıt={filtered['inactive_parent']} "
        f"üst_kayıt_firmasız={filtered['parent_company_missing']} "
        f"firma_uyuşmazlığı={filtered['company_mismatch']} "
        f"eski_sürüm={filtered['stale_version']}"
    )
    if not groups:
        print("Taşınacak firma-chunk'ı yok.")
        return 0

    if args.apply:
        try:
            _prepare_reconstruct(global_index)
        except MigrationAbort as exc:
            print(f"\nDURDURULDU: {exc}")
            return 3

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

        # Yeniden çalıştırma koruması: _write_index koşulsuz os.replace yapar, yani
        # ikinci bir koşu migration SONRASI yazılan tüm vektörleri yok eder.
        guard = _migration_marker_check(path, args.force)
        if guard:
            print(f"  [skip] {cid}: {guard}")
            continue

        try:
            idx, written, missing = _build_company_index(global_index, fids, dim, args.allow_missing)
        except MigrationAbort as exc:
            print(f"\nDURDURULDU ({cid}): {exc}")
            return 3
        _write_index(idx, path)
        _write_migration_marker(path, args.global_index, written, missing, len(fids))
        total_written += written
        total_missing += missing
        print(f"  [ok] {cid}: {written} yazıldı (eksik reconstruct={missing}) → {path}")

    if args.apply:
        print(f"\nBİTTİ: {len(groups)} firma, {total_written} chunk yazıldı, {total_missing} eksik.")
        if total_missing:
            print(f"UYARI: {total_missing} vektör eksik — bayrağı AÇMADAN önce sebebini araştırın.")
        print(
            "Global index DOKUNULMADI (rollback güvenli). Doğrulama: her firma için\n"
            "  GET /api/v10/company/<id>/index-stats  → drift ~0 olmalı\n"
            "ve aramada retrieval_meta.index_scope 'company' dönmeli ('global_fallback' değil."
        )
    else:
        print("\nDRY-RUN — yazım yapılmadı. Gerçek çalıştırma için --apply ekle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
