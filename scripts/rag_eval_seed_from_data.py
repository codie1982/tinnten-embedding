#!/usr/bin/env python3
"""
Gerçek indexlenmiş içerikten eval seed set TASLAĞI üretir (RAG v2 — Faz 1).

Fikir: bir chunk'tan AYIRT EDİCİ bir dize çıkar (ürün kodu, fiyat, politika
başlığı), onu sorgu olarak kullan — o chunk'ın kendi kaynak URL'si ground-truth
etiketidir. Böylece elle etiketleme olmadan gerçekçi bir başlangıç seti çıkar.

Aday üretilen sorgu tipleri:
  exact   — ürün/model kodu (ABC-450X gibi)           → dense'in zayıf, BM25'in güçlü olduğu yer
  price   — tutar (1.499,90 TL gibi)                  → aynı şekilde lexical-ağırlıklı
  policy  — politika başlığı (İade ve Değişim gibi)   → markdown heading'den
  semantic— doküman başlığı/heading (doğal ifade)     → dense'in güçlü olduğu yer

`conversational` OTOMATİK ÜRETİLMEZ: "peki siyahı var mı" gibi bağlam-bağımlı
sorgular veriden türetilemez, elle yazılmalı (şablon olarak eklenir).

ÖNEMLİ — çıktı bir TASLAKTIR, olduğu gibi kullanma:
  * Sorgular içerikten türetilir, gerçek kullanıcı ifadesi değildir. Özellikle
    `semantic` adayları elle doğal dile çevrilmeli.
  * Etiketler "bu dizeyi içeren sayfalar" demektir; anlamsal alaka insan onayı ister.
  * Ayırt edicilik kontrol edilir (bir dize çok sayıda sayfada geçiyorsa elenir),
    ama yine de gözden geçir.

Salt-okunur: yalnız find/aggregate. Hiçbir yazma yok.

Kullanım:
  # once hangi firmalar var?
  python3 scripts/rag_eval_seed_from_data.py --survey

  # bir firma icin taslak uret
  python3 scripts/rag_eval_seed_from_data.py \
    --company-id <COMPANY_ID> --per-type 8 --out scripts/rag_eval_queries.draft.json

Exit: 0 ok, 1 veri yok/yetersiz, 2 kötü girdi/bağlantı.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Aday çıkarıcılar — saf fonksiyonlar (tests/test_rag_eval_seed.py)
# ---------------------------------------------------------------------------

# Ürün/model kodu: en az 2 harf + ayraç + en az 2 rakam (ABC-450X, XZ 1200, TR-99A)
# Sadece-rakam veya sadece-harf ELENİR (yıl, tutar, sıradan kelime olmasın).
PRODUCT_CODE_RE = re.compile(r"\b([A-Z]{2,}[-_ ]?\d{2,}[A-Z0-9]{0,4})\b")

# Tutar: 1.499,90 TL / 250 TL / ₺99 / 19.99 USD
PRICE_RE = re.compile(
    r"(?:(?:₺|\$|€)\s?\d{1,3}(?:[.\s]\d{3})*(?:,\d{1,2})?)"
    r"|(?:\b\d{1,3}(?:[.\s]\d{3})*(?:,\d{1,2})?\s?(?:TL|TRY|USD|EUR)\b)",
    re.IGNORECASE,
)

ATX_HEADING_RE = re.compile(r"^#{1,6}\s+(.*\S)\s*$", re.MULTILINE)

POLICY_KEYWORDS = (
    "iade", "değişim", "degisim", "kargo", "teslimat", "garanti", "gizlilik",
    "kvkk", "şartlar", "sartlar", "koşullar", "kosullar", "sözleşme", "sozlesme",
    "iptal", "ödeme", "odeme", "mesafeli",
)

# Otomatik türetilemeyen tip — elle yazılacak şablon.
CONVERSATIONAL_TEMPLATE = {
    "query": "TODO: konuşma bağlamlı bir sorgu yaz (ör. 'peki siyahı var mı')",
    "company_id": None,
    "relevant_urls": ["TODO: hangi sayfa cevaplamalı"],
    "relevant_doc_ids": [],
    "query_type": "conversational",
    "content_type": "TODO: schema|clean_markdown|raw_markdown|upload",
    "_note": "Bağlam-bağımlı sorgular veriden türetilemez — elle yaz. Orchestration "
             "katmanı bunu standalone sorguya çevirir; burada standalone halini etiketle.",
}


def extract_product_codes(text: str) -> List[str]:
    out: List[str] = []
    for m in PRODUCT_CODE_RE.finditer(text or ""):
        code = m.group(1).strip()
        if not re.search(r"\d", code) or not re.search(r"[A-Z]", code):
            continue
        out.append(code)
    return out


def extract_prices(text: str) -> List[str]:
    return [m.group(0).strip() for m in PRICE_RE.finditer(text or "")]


def extract_headings(text: str) -> List[str]:
    return [m.group(1).strip() for m in ATX_HEADING_RE.finditer(text or "")]


def is_policy_heading(heading: str) -> bool:
    low = (heading or "").casefold()
    return any(kw in low for kw in POLICY_KEYWORDS)


def infer_content_type(chunk: Dict[str, Any]) -> str:
    """Chunk metnine bakarak içerik türünü tahmin eder.

    Fetcher, schema extraction başarılıysa DÜZ `\\n`-join (başlıksız) metin
    indexler; markdown dalı ATX başlık taşır. Ayrım kritik: structure chunking
    yalnız başlıklı içerikte fayda verir.
    """
    source = str(chunk.get("source") or "").lower()
    doc_type = str(chunk.get("doc_type") or "").lower()
    if "upload" in source or "upload" in doc_type or "library" in source:
        return "upload"
    text = str(chunk.get("text") or "")
    if ATX_HEADING_RE.search(text):
        return "clean_markdown"
    return "schema"


# ---------------------------------------------------------------------------
# Mongo (salt-okunur)
# ---------------------------------------------------------------------------
def connect(uri: str, timeout_ms: int = 15000):
    from pymongo import MongoClient

    client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
    client.admin.command("ping")
    return client


def survey(chunks) -> None:
    total = chunks.estimated_document_count()
    print(f"\nembedding_chunks toplam: {total}")
    print("\n=== chunk / firma (top 15) ===")
    rows = chunks.aggregate(
        [
            {"$group": {"_id": "$company_id", "n": {"$sum": 1}, "docs": {"$addToSet": "$doc_id"}}},
            {"$project": {"n": 1, "docs": {"$size": "$docs"}}},
            {"$sort": {"n": -1}},
            {"$limit": 15},
        ]
    )
    print(f"{'company_id':<30}{'chunks':>9}{'docs':>7}")
    print("-" * 46)
    for r in rows:
        print(f"{str(r['_id'])[:28]:<30}{r['n']:>9}{r['docs']:>7}")


def count_pages_containing(chunks, company_id: str, needle: str, cap: int) -> Set[str]:
    """Bu dizeyi içeren sayfaların URL'leri (ayırt edicilik + çoklu etiket için).

    Regex taraması pahalı → `cap` ile sınırlanır.
    """
    escaped = re.escape(needle)
    cursor = chunks.find(
        {"company_id": company_id, "text": {"$regex": escaped}},
        {"metadata.url": 1, "doc_id": 1},
    ).limit(cap)
    urls: Set[str] = set()
    for row in cursor:
        url = (row.get("metadata") or {}).get("url")
        if url:
            urls.add(str(url))
    return urls


@dataclass
class Candidate:
    query: str
    query_type: str
    content_type: str
    urls: Set[str] = field(default_factory=set)
    doc_ids: Set[str] = field(default_factory=set)


def build_candidates(
    chunks,
    company_id: str,
    per_type: int,
    sample_size: int,
    max_pages_per_query: int,
    distinctness_cap: int,
) -> Tuple[List[Candidate], Dict[str, int]]:
    cursor = chunks.find(
        {"company_id": company_id},
        {"text": 1, "metadata": 1, "doc_id": 1, "source": 1, "doc_type": 1},
    ).limit(sample_size)

    seen_query: Set[str] = set()
    buckets: Dict[str, List[Candidate]] = defaultdict(list)
    stats: Dict[str, int] = defaultdict(int)

    for chunk in cursor:
        stats["chunks_scanned"] += 1
        text = str(chunk.get("text") or "")
        url = (chunk.get("metadata") or {}).get("url")
        doc_id = chunk.get("doc_id")
        content_type = infer_content_type(chunk)
        stats[f"content_type:{content_type}"] += 1

        proposals: List[Tuple[str, str]] = []
        for code in extract_product_codes(text)[:2]:
            proposals.append((code, "exact"))
        for price in extract_prices(text)[:1]:
            proposals.append((price, "price"))
        for heading in extract_headings(text)[:3]:
            if len(heading) < 4 or len(heading) > 80:
                continue
            proposals.append((heading, "policy" if is_policy_heading(heading) else "semantic"))

        for needle, qtype in proposals:
            if len(buckets[qtype]) >= per_type:
                continue
            key = needle.casefold()
            if key in seen_query:
                continue
            seen_query.add(key)

            urls = count_pages_containing(chunks, company_id, needle, distinctness_cap)
            if len(urls) > max_pages_per_query:
                # Çok yaygın → ayırt edici değil (ör. her sayfadaki menü linki)
                stats["dropped_not_distinctive"] += 1
                continue

            candidate = Candidate(query=needle, query_type=qtype, content_type=content_type)
            if urls:
                candidate.urls = urls
            elif url:
                candidate.urls = {str(url)}
            elif doc_id:
                candidate.doc_ids = {str(doc_id)}
            else:
                stats["dropped_no_label"] += 1
                continue
            buckets[qtype].append(candidate)
            stats[f"kept:{qtype}"] += 1

        if all(len(buckets[t]) >= per_type for t in ("exact", "price", "policy", "semantic")):
            break

    out: List[Candidate] = []
    for qtype in ("exact", "price", "policy", "semantic"):
        out.extend(buckets[qtype])
    return out, dict(stats)


def to_seed_entries(candidates: Sequence[Candidate], company_id: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for c in candidates:
        entry: Dict[str, Any] = {
            "query": c.query,
            "company_id": company_id,
            "relevant_urls": sorted(c.urls),
            "relevant_doc_ids": sorted(c.doc_ids),
            "query_type": c.query_type,
            "content_type": c.content_type,
        }
        if c.query_type == "semantic":
            entry["_review"] = (
                "TASLAK: bu, içerikten alınmış bir başlık — gerçek kullanıcı ifadesine "
                "çevir (ör. 'İş Laptopları' → 'iş için hafif dizüstü bilgisayar')."
            )
        else:
            entry["_review"] = "TASLAK: etiketleri ve alakayı doğrula."
        entries.append(entry)
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerçek veriden eval seed taslağı üret (read-only)")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI"))
    parser.add_argument("--embed-db", default=os.getenv("EMBED_DB_NAME", "tinnten-embedding"))
    parser.add_argument("--survey", action="store_true", help="Sadece firma/chunk dağılımını göster")
    parser.add_argument("--company-id", default=os.getenv("COMPANY_ID"))
    parser.add_argument("--per-type", type=int, default=8, help="Sorgu tipi başına aday sayısı")
    parser.add_argument("--sample-size", type=int, default=4000, help="Taranacak chunk üst sınırı")
    parser.add_argument(
        "--max-pages-per-query",
        type=int,
        default=3,
        help="Bir dize bundan fazla sayfada geçiyorsa ayırt edici değil → elenir",
    )
    parser.add_argument("--distinctness-cap", type=int, default=25, help="Ayırt edicilik taramasının üst sınırı")
    parser.add_argument("--out", default="scripts/rag_eval_queries.draft.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mongo_uri:
        print("ERROR: --mongo-uri veya MONGO_URI gerekli", file=sys.stderr)
        return 2

    try:
        client = connect(args.mongo_uri)
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: Mongo'ya bağlanılamadı: {type(exc).__name__}: {str(exc)[:180]}\n"
            f"       DB firewall arkasındaysa bu script'i sunucu içinden / container'dan "
            f"veya SSH tüneliyle çalıştır.",
            file=sys.stderr,
        )
        return 2

    chunks = client[args.embed_db]["embedding_chunks"]

    if args.survey:
        survey(chunks)
        return 0

    if not args.company_id:
        print("ERROR: --company-id gerekli (önce --survey ile listele)", file=sys.stderr)
        return 2

    log(f"Firma {args.company_id} için aday çıkarılıyor (sample_size={args.sample_size})")
    candidates, stats = build_candidates(
        chunks,
        args.company_id,
        args.per_type,
        args.sample_size,
        args.max_pages_per_query,
        args.distinctness_cap,
    )

    print("\n=== İstatistik ===")
    for key in sorted(stats):
        print(f"  {key:<32}{stats[key]}")

    if not candidates:
        print(
            "\nHiç aday çıkmadı. Firma id doğru mu, chunk'lar var mı? "
            "(--survey ile kontrol et)",
            file=sys.stderr,
        )
        return 1

    entries = to_seed_entries(candidates, args.company_id)
    template = dict(CONVERSATIONAL_TEMPLATE)
    template["company_id"] = args.company_id
    entries.append(template)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    by_type: Dict[str, int] = defaultdict(int)
    for e in entries:
        by_type[e["query_type"]] += 1

    print(f"\n{len(entries)} aday yazıldı → {out_path}")
    print("Tip dağılımı:", dict(by_type))
    print(
        "\nSONRAKİ ADIM — bu bir TASLAK:\n"
        "  1) `semantic` adaylarını gerçek kullanıcı ifadesine çevir.\n"
        "  2) `conversational` şablonunu elle doldur.\n"
        "  3) Etiketleri (relevant_urls) gözden geçir; her sorguyu TEK yolla etiketle.\n"
        "  4) `_review`/`_note` alanlarını sil, dosyayı rag_eval_queries.json olarak kaydet.\n"
        "  5) python3 scripts/rag_eval.py --queries <dosya>"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
