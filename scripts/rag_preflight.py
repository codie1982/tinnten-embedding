#!/usr/bin/env python3
"""
Hybrid retrieval önkoşul denetimi (RAG v2 — Faz 2).

Neden var: hybrid'in lexical dalı Mongo `$text` index'ine dayanır, ama index
oluşturma hataları `mongo_store._ensure_indexes` içinde SESSİZCE yutulur
(try/except pass). Index yoksa `$text` sorgusu hata vermez — sadece boş döner.
Sonuç: hybrid "açık" görünür, RRF füzyonu tek bacakla çalışır ve kimse fark
etmez. Bu script o sessiz arızayı deploy ÖNCESİ yakalar.

Üç şeyi ayrı ayrı doğrular (biri geçip diğeri kalabilir):
  1) `chunk_text_search` index'i GERÇEKTEN var mı        (listIndexes)
  2) çıplak `$text` sorgusu sonuç dönüyor mu             (index çalışıyor mu)
  3) `company_id` FİLTRESİYLE `$text` sonuç dönüyor mu   (gerçek sorgu şekli)

(3) ayrı test edilir çünkü gerçek aramalar daima firma filtresiyle gelir
(`_translate_chunk_filters`); index var olup filtreli sorgunun boş dönmesi
mümkündür (yanlış alan adı, company_id tip uyuşmazlığı vb.).

Ayrıca servisin capability'sini (`GET /api/v10/config`) raporlar: hybrid açık mı,
reranker yüklü mü.

Salt-okunur: yalnız listIndexes + find. Hiçbir yazma yok.

Exit: 0 tüm kontroller geçti, 1 kontrol başarısız, 2 kötü girdi/bağlantı.

Kullanım:
  python3 scripts/rag_preflight.py --company-id <ID> --probe-word iade
  python3 scripts/rag_preflight.py --base-url http://localhost:5003 --token $TOKEN
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

CONFIG_PATH = "/api/v10/config"
TEXT_INDEX_NAME = "chunk_text_search"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def check_text_index_exists(chunks) -> Tuple[bool, str]:
    """(1) `chunk_text_search` index'i listIndexes'te var mı?"""
    try:
        indexes = list(chunks.list_indexes())
    except Exception as exc:  # noqa: BLE001
        return False, f"listIndexes basarisiz: {exc}"
    names = [ix.get("name") for ix in indexes]
    if TEXT_INDEX_NAME in names:
        return True, f"'{TEXT_INDEX_NAME}' var"
    text_like = [ix.get("name") for ix in indexes if "text" in str(ix.get("key", {}))]
    return False, (
        f"'{TEXT_INDEX_NAME}' YOK. Mevcut: {names}. "
        f"(text-benzeri: {text_like or 'yok'}) → hybrid'in lexical bacagi sessizce bos doner."
    )


def check_bare_text_query(chunks, word: str) -> Tuple[bool, str]:
    """(2) Çıplak `$text` sorgusu çalışıyor ve sonuç dönüyor mu?"""
    try:
        rows = list(chunks.find({"$text": {"$search": word}}, {"_id": 1}).limit(1))
    except Exception as exc:  # noqa: BLE001
        return False, f"$text sorgusu HATA verdi: {exc}"
    if rows:
        return True, f"'{word}' icin sonuc dondu"
    return False, (
        f"'{word}' icin 0 sonuc. Index var ama eslesme yok — probe kelimesi "
        f"korpusta gecmiyor olabilir (--probe-word ile baska bir kelime dene)."
    )


def check_filtered_text_query(chunks, word: str, company_id: str) -> Tuple[bool, str]:
    """(3) `company_id` filtresiyle `$text` — gerçek arama bu şekilde gelir."""
    query: Dict[str, Any] = {"$text": {"$search": word}, "company_id": company_id}
    try:
        rows = list(chunks.find(query, {"_id": 1}).limit(1))
    except Exception as exc:  # noqa: BLE001
        return False, f"filtreli $text HATA verdi: {exc}"
    if rows:
        return True, f"company_id={company_id} + '{word}' icin sonuc dondu"
    # company_id yerine metadata.companyId altinda mi duruyor?
    alt = dict(query)
    alt.pop("company_id")
    alt["metadata.companyId"] = company_id
    try:
        alt_rows = list(chunks.find(alt, {"_id": 1}).limit(1))
    except Exception:  # noqa: BLE001
        alt_rows = []
    if alt_rows:
        return False, (
            f"top-level `company_id` ile 0 sonuc AMA `metadata.companyId` ile sonuc VAR → "
            f"filtre alan adi uyusmazligi; arama filtreleri bu firmada bos donebilir."
        )
    return False, (
        f"company_id={company_id} + '{word}' icin 0 sonuc. Firma id dogru mu, "
        f"bu firmanin chunk'lari var mi? (--probe-word degistirmeyi dene)"
    )


def fetch_capability(base_url: str, token: Optional[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(f"{base_url}{CONFIG_PATH}", headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json(), "ok"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid retrieval onkosul denetimi (read-only)")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI"))
    parser.add_argument("--embed-db", default=os.getenv("EMBED_DB_NAME", "tinnten-embedding"))
    parser.add_argument("--company-id", default=os.getenv("COMPANY_ID"))
    parser.add_argument(
        "--probe-word",
        default=os.getenv("RAG_PREFLIGHT_WORD", "iade"),
        help="Korpusta gecmesi beklenen bir kelime",
    )
    parser.add_argument("--base-url", default=os.getenv("EMBEDDING_BASE_URL"))
    parser.add_argument("--token", default=os.getenv("EMBEDDING_BEARER_TOKEN"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: List[Tuple[str, bool, str]] = []

    # --- Servis capability (opsiyonel) ---
    if args.base_url:
        capability, err = fetch_capability(args.base_url.rstrip("/"), args.token)
        if capability is None:
            results.append(("service /config", False, f"erisilemedi: {err}"))
        else:
            hybrid_on = bool(capability.get("hybrid_search_enabled"))
            results.append(
                (
                    "service /config",
                    True,
                    f"hybrid_enabled={hybrid_on} "
                    f"reranker_configured={capability.get('reranker_configured')} "
                    f"reranker_loaded={capability.get('reranker_loaded')}",
                )
            )
            if capability.get("reranker_load_error"):
                results.append(
                    ("reranker load", False, f"yukleme hatasi: {capability['reranker_load_error']}")
                )

    # --- Mongo $text kontrolleri ---
    if not args.mongo_uri:
        print("ERROR: --mongo-uri veya MONGO_URI gerekli", file=sys.stderr)
        return 2
    try:
        from pymongo import MongoClient
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: pymongo gerekli: {exc}", file=sys.stderr)
        return 2

    try:
        client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=15000)
        client.admin.command("ping")
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: Mongo'ya baglanilamadi: {type(exc).__name__}: {str(exc)[:160]}\n"
            f"       DB firewall arkasindaysa bu script'i sunucu icinden/container'dan calistir.",
            file=sys.stderr,
        )
        return 2

    chunks = client[args.embed_db]["embedding_chunks"]

    ok, msg = check_text_index_exists(chunks)
    results.append(("(1) chunk_text_search index", ok, msg))

    # Index yoksa $text sorgulari zaten anlamsiz — yine de calistirip
    # gercek hata mesajini gostermek teshis icin faydali.
    ok2, msg2 = check_bare_text_query(chunks, args.probe_word)
    results.append(("(2) ciplak $text", ok2, msg2))

    if args.company_id:
        ok3, msg3 = check_filtered_text_query(chunks, args.probe_word, args.company_id)
        results.append(("(3) company_id filtreli $text", ok3, msg3))
    else:
        results.append(
            ("(3) company_id filtreli $text", False, "ATLANDI: --company-id verilmedi (gercek sorgu sekli test edilmedi)")
        )

    # --- Rapor ---
    print("\n=== Preflight ===")
    width = max(len(name) for name, _, _ in results)
    failed = 0
    for name, ok, msg in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {name:<{width}}  {msg}")

    if failed:
        print(f"\n{failed} kontrol basarisiz → hybrid'i acmadan once duzelt.")
        return 1
    print("\nTum kontroller gecti.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
