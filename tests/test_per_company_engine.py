"""
Per-company FAISS altyapısı testleri.

F1a: EmbeddingEngine model paylaşımı — aynı model_name'i kullanan tüm engine'ler
(per-company index'ler dâhil) TEK SentenceTransformer paylaşmalı ki 256 firma =
256 model yüklemesi (OOM) olmasın.
"""
from unittest.mock import patch, MagicMock


def _make_engines(model_names_paths):
    """Verilen (model_name, index_path) çiftleri için EmbeddingEngine üretir.
    SentenceTransformer + faiss lock dışı her şey mock'lu; sadece model paylaşımı
    davranışını izole ederiz."""
    from services.embedding_engine import EmbeddingEngine

    EmbeddingEngine._model_cache.clear()
    engines = [
        EmbeddingEngine(model_name=mn, index_path=ip) for mn, ip in model_names_paths
    ]
    return engines


def test_same_model_name_shares_single_model_instance():
    with patch("services.embedding_engine.SentenceTransformer") as MockST:
        # Her çağrı ayrı bir sahte model döndürsün ki paylaşımı kimlik ile ölçelim.
        MockST.side_effect = lambda name: MagicMock(name=f"model::{name}")

        e_a, e_b, e_c = _make_engines(
            [
                ("chunk-model", "/tmp/company/a.index"),
                ("chunk-model", "/tmp/company/b.index"),  # aynı model, farklı index
                ("other-model", "/tmp/company/c.index"),  # farklı model
            ]
        )

        # Aynı model_name → aynı model instance (paylaşılıyor)
        assert e_a._model is e_b._model
        # Farklı model_name → ayrı instance
        assert e_a._model is not e_c._model
        # Model yalnızca model_name başına bir kez yüklenmeli (2 farklı isim)
        assert MockST.call_count == 2


def test_chunk_search_routes_to_company_engine(client, mocker):
    """company_id filtresi varken arama, o firmanın engine'ine yönlendirilmeli."""
    import numpy as np

    mock_engine = mocker.Mock()
    mock_engine.encode.return_value = np.zeros((1, 768), dtype=np.float32)
    mock_engine.search.return_value = (np.array([[0.1]]), np.array([[10]]))
    routed = mocker.patch("app.get_company_chunk_engine", return_value=mock_engine)
    mocker.patch(
        "app.chunk_store.get_chunks_by_faiss_ids",
        return_value={
            10: {"doc_id": "d1", "text": "t", "chunk_id": "c10", "metadata": {"companyId": "CID"}},
        },
    )
    mocker.patch(
        "app.chunk_store.get_documents_by_ids", return_value={"d1": {"status": "active"}}
    )

    resp = client.post(
        "/api/v10/content/search",
        json={"text": "q", "k": 1, "filter": {"company_id": "CID"}},
    )
    assert resp.status_code == 200
    # Firma engine'i CID ile çözüldü ve arama onunla yapıldı.
    routed.assert_any_call("CID")
    mock_engine.search.assert_called_once()


def test_company_index_path_isolation_and_fallback():
    """Path çözümü: firma → company/<id>.index; boş → global; sanitize edilir."""
    import app

    p_cid = app._company_chunk_index_path("6a2b1893be2ff2f3a426c218")
    assert p_cid.endswith("company/6a2b1893be2ff2f3a426c218.index")
    # company_id yok → global chunk index
    assert app._company_chunk_index_path("") == app.CHUNK_INDEX_PATH
    assert app._company_chunk_index_path(None) == app.CHUNK_INDEX_PATH
    # path traversal denemesi temizlenir
    assert ".." not in app._company_chunk_index_path("../../etc/passwd")


def test_flag_off_falls_back_to_global_engine(mocker):
    """PER_COMPANY_FAISS_ENABLED kapalıyken company_id olsa bile global engine."""
    import app

    mocker.patch.object(app, "PER_COMPANY_FAISS_ENABLED", False)
    global_engine = mocker.Mock(name="global")
    mocker.patch("app.get_chunk_engine", return_value=global_engine)
    assert app.get_company_chunk_engine("6a2b1893be2ff2f3a426c218") is global_engine


def test_flag_on_uses_company_engine(mocker):
    """PER_COMPANY_FAISS_ENABLED açıkken company_id → firma engine; boş → global."""
    import app

    mocker.patch.object(app, "PER_COMPANY_FAISS_ENABLED", True)
    company_engine = mocker.Mock(name="company")
    global_engine = mocker.Mock(name="global")
    mocker.patch("app._get_company_chunk_engine_cached", return_value=company_engine)
    mocker.patch("app.get_chunk_engine", return_value=global_engine)
    assert app.get_company_chunk_engine("CID") is company_engine
    assert app.get_company_chunk_engine("") is global_engine


def test_remove_domain_is_scoped_to_company_engine(client, mocker):
    """remove/domain: firmanın o domain'e ait faiss'lerini KENDİ engine'inden siler,
    chunk'ları temizler; başka firmaya dokunmaz."""
    mock_engine = mocker.Mock()
    routed = mocker.patch("app.get_company_chunk_engine", return_value=mock_engine)
    idx_mock = mocker.patch(
        "app.chunk_store.get_chunk_index_by_company_domain",
        return_value={"faiss_ids": [10, 11, 12], "doc_ids": ["d1", "d2"]},
    )
    del_mock = mocker.patch(
        "app.chunk_store.delete_chunks_by_company_domain", return_value=3
    )
    mocker.patch("app.chunk_store.update_document_status")
    mocker.patch("app._deactivate_index_state")

    resp = client.post(
        "/api/v10/content/index/remove/domain",
        json={"companyId": "CID", "domain": "www.Example.com", "mode": "hard"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    routed.assert_any_call("CID")
    mock_engine.remove_ids.assert_called_once_with([10, 11, 12])
    del_mock.assert_called_once()
    # (company_id, normalize edilmiş domain) ile sorgulandı
    assert idx_mock.call_args.args[0] == "CID"
    assert data["faissRemoved"] == 3
    assert data["chunksDeleted"] == 3
    assert data["docs"] == 2


def test_remove_domain_requires_company_and_domain(client):
    r1 = client.post("/api/v10/content/index/remove/domain", json={"companyId": "CID"})
    assert r1.status_code == 400
    r2 = client.post("/api/v10/content/index/remove/domain", json={"domain": "x.com"})
    assert r2.status_code == 400


def test_different_index_paths_are_independent_engines():
    with patch("services.embedding_engine.SentenceTransformer") as MockST:
        MockST.side_effect = lambda name: MagicMock(name="model")
        e_a, e_b = _make_engines(
            [("chunk-model", "/tmp/company/a.index"), ("chunk-model", "/tmp/company/b.index")]
        )
        # Farklı index → farklı engine (farklı index_path, farklı lock)
        assert e_a is not e_b
        assert e_a.index_path != e_b.index_path
        assert e_a._lock is not e_b._lock
