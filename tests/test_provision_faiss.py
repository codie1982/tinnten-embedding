"""
Adım 1 — company-create → per-company FAISS provision.
- POST /api/v10/company/<id>/provision-faiss: bayrak kapalı=no-op, açık=engine.ensure_index_persisted
- EmbeddingEngine.ensure_index_persisted: boş index oluşturur (idempotent)
"""
import os
import threading

import pytest


# ---------------------------------------------------------------------------
# Endpoint davranışı (mock'lu engine)
# ---------------------------------------------------------------------------
def test_provision_disabled_is_noop(client, mocker):
    mocker.patch("app.PER_COMPANY_FAISS_ENABLED", False)
    spy = mocker.patch("app.get_company_chunk_engine")
    r = client.post("/api/v10/company/abc123/provision-faiss")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["skipped"] == "feature_disabled"
    spy.assert_not_called()  # bayrak kapalı → engine'e hiç dokunma


def test_provision_enabled_creates_index(client, mocker):
    mocker.patch("app.PER_COMPANY_FAISS_ENABLED", True)
    fake_engine = mocker.Mock()
    fake_engine.ensure_index_persisted.return_value = {
        "created": True, "path": "company/abc123.index", "dimension": 768, "ntotal": 0,
    }
    getter = mocker.patch("app.get_company_chunk_engine", return_value=fake_engine)
    r = client.post("/api/v10/company/abc123/provision-faiss")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["companyId"] == "abc123" and body["created"] is True
    getter.assert_called_once_with("abc123")
    fake_engine.ensure_index_persisted.assert_called_once()


def test_provision_blank_company_id_400(client, mocker):
    mocker.patch("app.PER_COMPANY_FAISS_ENABLED", True)
    r = client.post("/api/v10/company/%20/provision-faiss")  # whitespace → strip → boş
    assert r.status_code == 400


def test_provision_engine_error_500(client, mocker):
    mocker.patch("app.PER_COMPANY_FAISS_ENABLED", True)
    fake_engine = mocker.Mock()
    fake_engine.ensure_index_persisted.side_effect = RuntimeError("disk full")
    mocker.patch("app.get_company_chunk_engine", return_value=fake_engine)
    r = client.post("/api/v10/company/abc123/provision-faiss")
    assert r.status_code == 500
    assert r.get_json()["ok"] is False


# ---------------------------------------------------------------------------
# EmbeddingEngine.ensure_index_persisted — gerçek faiss ile boş index yazımı
# ---------------------------------------------------------------------------
def _bare_engine(tmp_path, mocker, dim=8):
    from services.embedding_engine import EmbeddingEngine
    eng = EmbeddingEngine.__new__(EmbeddingEngine)  # __init__/model yüklemesini atla
    eng._lock = threading.RLock()
    eng._index = None
    eng._dimension = None
    eng._index_mtime = None
    eng.index_path = str(tmp_path / "company" / "c1.index")
    eng.max_index_dimension = 100000
    eng.auto_reset_on_dim_mismatch = False
    eng.model_name = "test-model"
    mocker.patch.object(EmbeddingEngine, "model_dimension", return_value=dim)
    return eng


def test_ensure_index_persisted_creates_empty(tmp_path, mocker):
    eng = _bare_engine(tmp_path, mocker)
    res = eng.ensure_index_persisted()
    assert res["created"] is True
    assert res["ntotal"] == 0
    assert res["dimension"] == 8
    assert os.path.exists(eng.index_path)  # dosya + company/ dizini oluştu


def test_ensure_index_persisted_idempotent(tmp_path, mocker):
    eng = _bare_engine(tmp_path, mocker)
    first = eng.ensure_index_persisted()
    assert first["created"] is True
    second = eng.ensure_index_persisted()  # dosya artık var → dokunma
    assert second["created"] is False
    assert second["ntotal"] == 0
