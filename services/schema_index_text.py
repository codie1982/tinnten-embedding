"""
Şema-çıkarım (extracted_content) → index metni indirgeme.

RAG v2 — Faz 6. Fetcher canlı yolunda embedding'e giden metin
`schema extraction > clean_markdown > raw markdown` önceliğiyle seçilir
(tinnten-fetcher `result_processor._select_index_content`). Ama fetcher'ın
DB-read yolu (`/fetcher`, `/fetcher/pages`) embedding tarafında yeniden okunur
ve şu an YALNIZ markdown kullanır → schema metni kaybolur, reindex canlı ile
byte-özdeş olmaz.

Bu modül fetcher'daki `_index_text_from_extracted_content` +
`_collect_text_leaves` fonksiyonlarının BİREBİR portudur. Saf/deterministik →
aynı `extracted_content` için aynı çıktı. `extracted_content` Mongo'da kalıcı
(`crawl_results.extracted.extracted_content`) ve şema uygulaması deterministik
(kalıcı css/xpath selector; LLM yalnız onboarding'de schema üretir), o yüzden
re-crawl gerekmeden faithful reindex mümkün.

Parity, tests/test_schema_index_text.py'de fetcher fonksiyonlarına karşı
doğrulanır — biri değişirse test kırılır.
"""
from __future__ import annotations

import json
import re
from typing import Any, List

# Fetcher `services/config.py` → `Config.INDEX_SCHEMA_MIN_CHARS` ile AYNI olmalı.
# Bu birebirlik faithfulness için KRİTİK: eşik farkı olursa, schema metni bu
# değerin altında kalan bir sayfada fetcher markdown'a düşer ama port schema'yı
# indexler (veya tersi) → reindex canlı ile ayrışır. Fetcher hâlihazırda 200.
DEFAULT_SCHEMA_MIN_CHARS = 200

_BARE_URI_RE = re.compile(r"^(https?://|data:)\S+$")


def collect_text_leaves(value: Any, parts: List[str], depth: int = 0, max_depth: int = 6) -> None:
    """extracted_content ağacındaki string yaprakları toplar.

    Çıplak URL / data-uri yaprakları metin değildir (img src, href alanları) →
    atlanır. Fetcher `_collect_text_leaves` ile birebir (max_depth=6, list cap=200).
    """
    if depth > max_depth or value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text and not _BARE_URI_RE.match(text):
            parts.append(text)
        return
    if isinstance(value, dict):
        for child in value.values():
            collect_text_leaves(child, parts, depth + 1, max_depth)
        return
    if isinstance(value, list):
        for child in value[:200]:
            collect_text_leaves(child, parts, depth + 1, max_depth)


def index_text_from_extracted_content(
    extracted_content: Any, *, min_chars: int = DEFAULT_SCHEMA_MIN_CHARS
) -> str:
    """Şema çıkarım çıktısını (JSON str | dict | list | düz metin) index metnine indirger.

    String yapraklar şema alan sırasıyla `\\n` ile birleşir. Toplam metin
    `min_chars` altındaysa "" döner → çağıran markdown'a düşer (şema o sayfada
    boş/yanlış çıkarmışsa içerik KAYBOLMAZ). Fetcher
    `_index_text_from_extracted_content` ile birebir.
    """
    if not extracted_content:
        return ""
    value = extracted_content
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            # Düz metin (ör. LLM extraction) — JSON değilse olduğu gibi kullan.
            return text if len(text) >= min_chars else ""
    parts: List[str] = []
    collect_text_leaves(value, parts)
    text = "\n".join(parts).strip()
    return text if len(text) >= min_chars else ""
