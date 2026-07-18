"""
FAZ 7 — source-type-farkında chunk stratejisi ('auto').

KN3: global CHUNK_STRATEGY=structure yanlış — fetcher schema-derived metni
başlıksız (\n-join) indexler, orada heading-path fayda vermez. 'auto', başlık
VARSA structure, YOKSA char seçer. Açık 'char'/'structure' override'ları
dokunulmadan geçer (davranış korunur).

`_resolve_chunk_strategy` saf/statik → worker'ı kurmaya gerek yok.
"""
import pytest

from workers.ingest_worker import IngestWorker


R = IngestWorker._resolve_chunk_strategy


# ---------------------------------------------------------------------------
# Override'lar dokunulmaz
# ---------------------------------------------------------------------------
def test_explicit_char_passthrough():
    assert R("char", "# Başlık\nmetin") == "char"


def test_explicit_structure_passthrough():
    # Başlık olmasa bile açık 'structure' korunur (kullanıcı bilerek seçti).
    assert R("structure", "başlıksız düz metin") == "structure"


def test_unknown_strategy_passthrough():
    assert R("whatever", "# Başlık") == "whatever"


# ---------------------------------------------------------------------------
# 'auto' içeriğe göre karar verir
# ---------------------------------------------------------------------------
def test_auto_with_atx_heading_picks_structure():
    text = "# İade Politikası\n\nÜcretsiz iade 14 gündür."
    assert R("auto", text) == "structure"


def test_auto_with_deep_heading_picks_structure():
    assert R("auto", "sunum\n\n### Alt Başlık\niçerik") == "structure"


def test_auto_without_heading_picks_char():
    # Schema-derived tipik metin: düz \n-join, başlık yok.
    text = "İş Laptopu\nHafif ve güçlü\n1.499 TL\n14 gün iade"
    assert R("auto", text) == "char"


def test_auto_hash_in_prose_is_not_heading():
    # Satır ortasındaki '#' heading değildir → char.
    assert R("auto", "fiyat #1 tercih, C# ile yazıldı") == "char"


def test_auto_hash_without_space_is_not_heading():
    # '#' + boşluksuz (ör. '#etiket') heading değildir.
    assert R("auto", "#etiket #kampanya") == "char"


def test_auto_empty_text_picks_char():
    assert R("auto", "") == "char"
    assert R("auto", None) == "char"


def test_auto_heading_must_be_line_start():
    # Girintili '#' (kod bloğu vb.) satır başı sayılmaz → char.
    assert R("auto", "    # girintili") == "char"
