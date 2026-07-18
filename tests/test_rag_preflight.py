"""
FAZ 2 — hybrid önkoşul denetiminin testleri.

Denetimin varlık sebebi: `mongo_store._ensure_indexes` text-index oluşturma
hatasını SESSİZCE yutuyor (try/except pass). Index yoksa `$text` hata vermez,
sadece boş döner → hybrid "açık" görünür ama lexical bacak ölüdür. Buradaki
testler, preflight'ın bu sessiz arızayı gerçekten yakaladığını kanıtlar.

mongomock `$text` desteklemediği için Mongo koleksiyonu MagicMock'lanır
(mevcut `tests/test_hybrid_retrieval.py` deseninin aynısı).
"""
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_preflight():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rag_preflight.py"
    spec = importlib.util.spec_from_file_location("rag_preflight", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rag_preflight"] = module
    spec.loader.exec_module(module)
    return module


pf = _load_preflight()


# ---------------------------------------------------------------------------
# (1) text index varligi
# ---------------------------------------------------------------------------
def test_index_check_passes_when_present(mocker):
    chunks = mocker.MagicMock()
    chunks.list_indexes.return_value = [
        {"name": "faiss_id_unique", "key": {"faiss_id": 1}},
        {"name": "chunk_text_search", "key": {"_fts": "text"}},
    ]
    ok, msg = pf.check_text_index_exists(chunks)
    assert ok is True
    assert "chunk_text_search" in msg


def test_index_check_fails_when_missing_and_explains_impact(mocker):
    """Asıl senaryo: index oluşturma sessizce yutulmuş."""
    chunks = mocker.MagicMock()
    chunks.list_indexes.return_value = [{"name": "faiss_id_unique", "key": {"faiss_id": 1}}]
    ok, msg = pf.check_text_index_exists(chunks)
    assert ok is False
    assert "YOK" in msg
    assert "lexical" in msg  # etkisini açıklıyor


def test_index_check_fails_when_listindexes_errors(mocker):
    chunks = mocker.MagicMock()
    chunks.list_indexes.side_effect = RuntimeError("yetki yok")
    ok, msg = pf.check_text_index_exists(chunks)
    assert ok is False
    assert "listIndexes basarisiz" in msg


# ---------------------------------------------------------------------------
# (2) ciplak $text
# ---------------------------------------------------------------------------
def _cursor(mocker, rows):
    cur = mocker.MagicMock()
    cur.limit.return_value = rows
    return cur


def test_bare_text_passes_when_rows_returned(mocker):
    chunks = mocker.MagicMock()
    chunks.find.return_value = _cursor(mocker, [{"_id": 1}])
    ok, msg = pf.check_bare_text_query(chunks, "iade")
    assert ok is True


def test_bare_text_fails_silently_on_zero_rows(mocker):
    """Index yoksa $text HATA VERMEZ, boş döner — sessiz arıza tam da bu."""
    chunks = mocker.MagicMock()
    chunks.find.return_value = _cursor(mocker, [])
    ok, msg = pf.check_bare_text_query(chunks, "iade")
    assert ok is False
    assert "0 sonuc" in msg


def test_bare_text_reports_query_error(mocker):
    chunks = mocker.MagicMock()
    chunks.find.side_effect = RuntimeError("text index required")
    ok, msg = pf.check_bare_text_query(chunks, "iade")
    assert ok is False
    assert "HATA" in msg


# ---------------------------------------------------------------------------
# (3) company_id filtreli $text — gercek sorgu sekli
# ---------------------------------------------------------------------------
def test_filtered_text_passes(mocker):
    chunks = mocker.MagicMock()
    chunks.find.return_value = _cursor(mocker, [{"_id": 1}])
    ok, msg = pf.check_filtered_text_query(chunks, "iade", "c1")
    assert ok is True


def test_filtered_text_detects_field_name_mismatch(mocker):
    """En sinsi arıza: index var, çıplak sorgu çalışır, ama firma filtresi
    yanlış alanda → gerçek aramalar boş döner.

    top-level `company_id` boş, `metadata.companyId` dolu ise preflight bunu
    isim uyuşmazlığı olarak raporlamalı.
    """
    chunks = mocker.MagicMock()

    def find_side_effect(query, *args, **kwargs):
        if "company_id" in query:
            return _cursor(mocker, [])          # top-level: bos
        if "metadata.companyId" in query:
            return _cursor(mocker, [{"_id": 1}])  # nested: dolu
        return _cursor(mocker, [])

    chunks.find.side_effect = find_side_effect
    ok, msg = pf.check_filtered_text_query(chunks, "iade", "c1")
    assert ok is False
    assert "metadata.companyId" in msg
    assert "uyusmazligi" in msg


def test_filtered_text_zero_everywhere(mocker):
    chunks = mocker.MagicMock()
    chunks.find.return_value = _cursor(mocker, [])
    ok, msg = pf.check_filtered_text_query(chunks, "iade", "c1")
    assert ok is False
    assert "0 sonuc" in msg


def test_filtered_text_reports_error(mocker):
    chunks = mocker.MagicMock()
    chunks.find.side_effect = RuntimeError("boom")
    ok, msg = pf.check_filtered_text_query(chunks, "iade", "c1")
    assert ok is False
    assert "HATA" in msg
