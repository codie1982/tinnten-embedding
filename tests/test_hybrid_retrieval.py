"""
FAZ 5 — hybrid retrieval (dense FAISS + lexical $text, RRF füzyonu) + opsiyonel reranker.

Plan kararı #9: mongomock `$text`'i desteklemez → füzyon/çeviri/rerank SAF fonksiyon
olarak test edilir; endpoint hybrid dalı chunk_store mock'lanarak doğrulanır.
"""
import numpy as np
import pytest

from services.mongo_store import MongoStore


# ---------------------------------------------------------------------------
# _translate_chunk_filters — app.py _passes_chunk_filters ile birebir semantik
# ---------------------------------------------------------------------------
def test_translate_filters_metadata_and_toplevel():
    q = MongoStore._translate_chunk_filters(
        {"company_id": "c1", "metadata.domain": "modailgi.com.tr"}
    )
    assert q == {"company_id": "c1", "metadata.domain": "modailgi.com.tr"}


def test_translate_filters_preserves_in_operator():
    q = MongoStore._translate_chunk_filters(
        {"metadata.domain": {"$in": ["a.com", "b.com"]}, "doc_id": {"$in": ["d1"]}}
    )
    assert q["metadata.domain"] == {"$in": ["a.com", "b.com"]}
    assert q["doc_id"] == {"$in": ["d1"]}


def test_translate_filters_empty():
    assert MongoStore._translate_chunk_filters(None) == {}
    assert MongoStore._translate_chunk_filters({}) == {}


# ---------------------------------------------------------------------------
# text_search_chunks — $text sorgu inşası + filtre daraltması + boş sorgu
# ---------------------------------------------------------------------------
def _store_with_fake_chunks(mocker):
    store = MongoStore.__new__(MongoStore)  # __init__/Mongo bağlantısını atla
    fake_chunks = mocker.MagicMock()
    cursor = mocker.MagicMock()
    cursor.sort.return_value = cursor
    cursor.limit.return_value = [{"chunk_id": "x", "faiss_id": 7}]
    fake_chunks.find.return_value = cursor
    store.chunks = fake_chunks
    return store, fake_chunks, cursor


def test_text_search_builds_text_query_with_filters(mocker):
    store, fake_chunks, cursor = _store_with_fake_chunks(mocker)
    out = store.text_search_chunks(
        "yorgan fiyatları", {"metadata.domain": {"$in": ["modailgi.com.tr"]}}, limit=12
    )
    assert out == [{"chunk_id": "x", "faiss_id": 7}]
    query_arg, projection_arg = fake_chunks.find.call_args[0]
    assert query_arg["$text"] == {"$search": "yorgan fiyatları"}
    assert query_arg["metadata.domain"] == {"$in": ["modailgi.com.tr"]}
    assert projection_arg == {"score": {"$meta": "textScore"}}
    cursor.limit.assert_called_once_with(12)


def test_text_search_empty_query_short_circuits(mocker):
    store, fake_chunks, _ = _store_with_fake_chunks(mocker)
    assert store.text_search_chunks("   ", {"company_id": "c1"}) == []
    fake_chunks.find.assert_not_called()


# ---------------------------------------------------------------------------
# _rrf_fuse — saf füzyon: skor toplamı, sıralama, dedupe, tie-break
# ---------------------------------------------------------------------------
def test_rrf_fuse_sums_overlapping_and_keeps_dense_record(app_with_mocks):
    import app

    dense = [
        {"chunk_id": "a", "id": 1, "doc_id": "d", "score": 0.9},
        {"chunk_id": "b", "id": 2, "doc_id": "d", "score": 0.8},
    ]
    lexical = [
        {"chunk_id": "a", "id": 1, "doc_id": "d", "score": 0.1},  # örtüşen
        {"chunk_id": "c", "id": 3, "doc_id": "d", "score": 0.2},
    ]
    fused = app._rrf_fuse(dense, lexical, k_const=60)
    keys = [r["chunk_id"] for r in fused]
    # a iki listede de rank0 → en yüksek skor, ilk sırada
    assert keys[0] == "a"
    # a için DENSE kaydı korunur (score 0.9, lexical 0.1 değil)
    assert fused[0]["score"] == 0.9
    # b ve c eşit skorda (ikisi de tek listede rank1) → ilk-görülen b önce
    assert keys[1] == "b" and keys[2] == "c"
    # rrf_score alanı eklenir ve a'nınki en büyük (6 haneye yuvarlanır → abs tolerans)
    assert fused[0]["rrf_score"] == pytest.approx(2.0 / 60.0, abs=1e-6)
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_rrf_fuse_falls_back_to_id_doc_key_when_no_chunk_id(app_with_mocks):
    import app

    dense = [{"id": 5, "doc_id": "d1", "score": 0.5}]
    lexical = [{"id": 5, "doc_id": "d1", "score": 0.4}]  # aynı (id,doc) → tek aday
    fused = app._rrf_fuse(dense, lexical, k_const=60)
    assert len(fused) == 1
    assert fused[0]["rrf_score"] == pytest.approx(2.0 / 60.0, abs=1e-6)


def test_rrf_fuse_empty_inputs(app_with_mocks):
    import app

    assert app._rrf_fuse([], []) == []
    only_dense = app._rrf_fuse([{"chunk_id": "a", "id": 1, "doc_id": "d"}], [])
    assert [r["chunk_id"] for r in only_dense] == ["a"]


# ---------------------------------------------------------------------------
# _maybe_rerank — gating + yeniden sıralama + güvenli düşüş
# ---------------------------------------------------------------------------
def test_maybe_rerank_disabled_is_passthrough(app_with_mocks, mocker):
    import app

    spy = mocker.patch("app._get_reranker")
    results = [{"chunk_id": "a", "text": "x"}]
    out, applied = app._maybe_rerank("q", results, enabled=False)
    assert out is results
    assert applied is False
    spy.assert_not_called()


def test_maybe_rerank_no_model_is_passthrough(app_with_mocks, mocker):
    import app

    mocker.patch("app._get_reranker", return_value=None)
    results = [{"chunk_id": "a", "text": "x"}, {"chunk_id": "b", "text": "y"}]
    out, applied = app._maybe_rerank("q", results)
    assert out == results
    # Model yokken sessizce hybrid sırası döner; `applied=False` bunu çağırana
    # bildirir (eval "hybrid+rerank" sanıp yalnız hybrid ölçmesin).
    assert applied is False


def test_maybe_rerank_reorders_head_by_score(app_with_mocks, mocker):
    import app

    fake = mocker.Mock()
    fake.predict.return_value = [0.1, 0.9, 0.5]
    mocker.patch("app._get_reranker", return_value=fake)
    mocker.patch("app.RERANK_TOP_N", 3)
    results = [
        {"chunk_id": "a", "text": "ta"},
        {"chunk_id": "b", "text": "tb"},
        {"chunk_id": "c", "text": "tc"},
    ]
    out, applied = app._maybe_rerank("q", results)
    assert [r["chunk_id"] for r in out] == ["b", "c", "a"]
    assert out[0]["rerank_score"] == 0.9
    assert applied is True


def test_maybe_rerank_predict_failure_falls_back(app_with_mocks, mocker):
    import app

    fake = mocker.Mock()
    fake.predict.side_effect = RuntimeError("boom")
    mocker.patch("app._get_reranker", return_value=fake)
    results = [{"chunk_id": "a", "text": "x"}]
    out, applied = app._maybe_rerank("q", results)
    assert out is results
    assert applied is False


def test_title_heading_match_beats_repeated_inline_link(app_with_mocks):
    import app

    rows = [
        {
            "doc_id": "sidebar-page",
            "score": 0.91,
            "text": "Populer: [Plaj Donusu Ilk 24 Saat Cildinizi Rahatlatacak](/x)",
            "metadata": {"url": "https://example.com/unrelated"},
        },
        {
            "doc_id": "target-page",
            "score": 0.64,
            "text": "# Plaj Donusu Ilk 24 Saat Cildinizi Rahatlatacak En Iyi Kremler",
            "metadata": {"url": "https://example.com/target"},
        },
    ]

    out = app._promote_title_matches(
        "Plaj Donusu Ilk 24 Saat Cildinizi Rahatlatacak", rows
    )
    assert out[0]["doc_id"] == "target-page"
    assert out[0]["title_match"] == 2


def test_title_promotion_deduplicates_document_chunks(app_with_mocks):
    import app

    rows = [
        {"doc_id": "same", "score": 0.9, "text": "hello", "metadata": {}},
        {"doc_id": "same", "score": 0.8, "text": "hello again", "metadata": {}},
        {"doc_id": "other", "score": 0.7, "text": "other", "metadata": {}},
    ]
    out = app._promote_title_matches("semantic query", rows)
    assert [row["doc_id"] for row in out] == ["same", "other"]


# ---------------------------------------------------------------------------
# Endpoint hybrid dalı — dense + lexical füzyonu, per-request gating
# ---------------------------------------------------------------------------
def _mock_dense(mocker):
    mock_vec = np.zeros((1, 768), dtype=np.float32)
    engine = mocker.Mock()
    engine.encode.return_value = mock_vec
    engine.search.return_value = (np.array([[0.9, 0.7]]), np.array([[10, 20]]))
    mocker.patch("app.get_chunk_engine", return_value=engine)
    return engine


def test_hybrid_fuses_dense_and_lexical(client, mocker):
    _mock_dense(mocker)
    mocker.patch(
        "app.chunk_store.get_chunks_by_faiss_ids",
        return_value={
            10: {"doc_id": "d1", "text": "dense10", "chunk_id": "c10", "metadata": {}},
            20: {"doc_id": "d2", "text": "dense20", "chunk_id": "c20", "metadata": {}},
        },
    )
    mocker.patch(
        "app.chunk_store.get_documents_by_ids",
        return_value={"d1": {"status": "active"}, "d2": {"status": "active"}, "d3": {"status": "active"}},
    )
    # Lexical yalnız chunk'ı: c10 (örtüşür) + c30 (yeni)
    lex = mocker.patch(
        "app.chunk_store.text_search_chunks",
        return_value=[
            {"doc_id": "d1", "text": "dense10", "chunk_id": "c10", "faiss_id": 10, "metadata": {}, "score": 3.2},
            {"doc_id": "d3", "text": "lex30", "chunk_id": "c30", "faiss_id": 30, "metadata": {}, "score": 2.1},
        ],
    )
    resp = client.post("/api/v10/vector/search", json={"text": "q", "k": 3, "hybrid": True})
    assert resp.status_code == 200
    results = resp.get_json()["results"]
    lex.assert_called_once()
    ids = [r["chunk_id"] for r in results]
    assert "c30" in ids  # lexical-only aday füzyona girdi
    assert all(r["retrieval"] == "hybrid" for r in results)
    # c10 hem dense hem lexical → en yüksek RRF, ilk sırada
    assert ids[0] == "c10"


def test_hybrid_false_stays_dense(client, mocker):
    _mock_dense(mocker)
    mocker.patch(
        "app.chunk_store.get_chunks_by_faiss_ids",
        return_value={
            10: {"doc_id": "d1", "text": "dense10", "chunk_id": "c10", "metadata": {}},
            20: {"doc_id": "d2", "text": "dense20", "chunk_id": "c20", "metadata": {}},
        },
    )
    mocker.patch(
        "app.chunk_store.get_documents_by_ids",
        return_value={"d1": {"status": "active"}, "d2": {"status": "active"}},
    )
    lex = mocker.patch("app.chunk_store.text_search_chunks", return_value=[])
    resp = client.post("/api/v10/vector/search", json={"text": "q", "k": 2, "hybrid": False})
    assert resp.status_code == 200
    body = resp.get_json()
    results = body["results"]
    lex.assert_not_called()  # hybrid kapalı → lexical hiç çağrılmaz
    assert [r["id"] for r in results] == [10, 20]
    # Dense dalı da artık damgalanır: eval'in "dense mi hybrid mi çalıştı"yı
    # tahmin etmek yerine KANITLAYABİLMESİ için (eskiden yalnız hybrid damgalıydı).
    assert all(r["retrieval"] == "dense" for r in results)
    meta = body["retrieval_meta"]
    assert meta["retrieval"] == "dense"
    assert meta["hybrid_applied"] is False
    assert meta["rerank_applied"] is False
    assert meta["lexical_count"] == 0


# ---------------------------------------------------------------------------
# retrieval_meta — "hangi yol GERÇEKTEN çalıştı" kanıtı (sessiz fallback yakalama)
# ---------------------------------------------------------------------------
def _mock_hybrid_stores(mocker):
    _mock_dense(mocker)
    mocker.patch(
        "app.chunk_store.get_chunks_by_faiss_ids",
        return_value={
            10: {"doc_id": "d1", "text": "dense10", "chunk_id": "c10", "metadata": {}},
            20: {"doc_id": "d2", "text": "dense20", "chunk_id": "c20", "metadata": {}},
        },
    )
    mocker.patch(
        "app.chunk_store.get_documents_by_ids",
        return_value={"d1": {"status": "active"}, "d2": {"status": "active"}},
    )
    mocker.patch("app.chunk_store.text_search_chunks", return_value=[])


def test_retrieval_meta_reports_rerank_not_applied_when_model_missing(client, mocker):
    """Reranker YOKken rerank:true istenirse yanıt sessizce hybrid döner.

    `rerank_applied=False` bunu ele verir — eval aksi halde "hybrid+rerank"
    ölçtüğünü sanıp gerçekte yalnız hybrid ölçerdi.
    """
    _mock_hybrid_stores(mocker)
    mocker.patch("app._get_reranker", return_value=None)
    resp = client.post(
        "/api/v10/vector/search", json={"text": "q", "k": 2, "hybrid": True, "rerank": True}
    )
    assert resp.status_code == 200
    meta = resp.get_json()["retrieval_meta"]
    assert meta["hybrid_applied"] is True
    assert meta["rerank_requested"] is True
    assert meta["rerank_applied"] is False  # ← sessiz fallback görünür oldu


def test_retrieval_meta_reports_rerank_applied_when_model_runs(client, mocker):
    _mock_hybrid_stores(mocker)
    fake = mocker.Mock()
    fake.predict.return_value = [0.9, 0.1]
    mocker.patch("app._get_reranker", return_value=fake)
    resp = client.post(
        "/api/v10/vector/search", json={"text": "q", "k": 2, "hybrid": True, "rerank": True}
    )
    assert resp.status_code == 200
    meta = resp.get_json()["retrieval_meta"]
    assert meta["rerank_applied"] is True


def test_retrieval_meta_rerank_applied_false_when_predict_fails(client, mocker):
    _mock_hybrid_stores(mocker)
    fake = mocker.Mock()
    fake.predict.side_effect = RuntimeError("boom")
    mocker.patch("app._get_reranker", return_value=fake)
    resp = client.post(
        "/api/v10/vector/search", json={"text": "q", "k": 2, "hybrid": True, "rerank": True}
    )
    assert resp.status_code == 200
    meta = resp.get_json()["retrieval_meta"]
    assert meta["rerank_applied"] is False


# ---------------------------------------------------------------------------
# /api/v10/config — capability doğrulaması, YAN ETKİSİZ olmalı
# ---------------------------------------------------------------------------
def test_config_endpoint_reports_capability(client):
    resp = client.get("/api/v10/config")
    assert resp.status_code == 200
    body = resp.get_json()
    for key in (
        "hybrid_search_enabled",
        "rerank_top_n",
        "rrf_k_constant",
        "reranker_configured",
        "reranker_loaded",
        "reranker_load_error",
    ):
        assert key in body


def test_config_endpoint_does_not_load_reranker(client, mocker):
    """`/config` ASLA `_get_reranker()` çağırmamalı.

    Aksi halde read-only bir capability sorgusu ~2GB model indirmeyi tetiklerdi.
    """
    spy = mocker.patch("app._get_reranker")
    resp = client.get("/api/v10/config")
    assert resp.status_code == 200
    spy.assert_not_called()


# ---------------------------------------------------------------------------
# Faz 4 — reranker warmup: boot'ta yüklenir ama başarısızlık boot'u DÜŞÜRMEZ
# ---------------------------------------------------------------------------
def _reset_warmup(mocker):
    """Warmup singleton'ını sıfırla ve ağır adımları (store/engine) mock'la."""
    import app

    mocker.patch.object(app, "_startup_warmup_done", False)
    mocker.patch.object(app.store, "warmup_model")
    mocker.patch("app.get_chunk_engine")
    return app


def test_warmup_loads_reranker_when_configured(app_with_mocks, mocker):
    app = _reset_warmup(mocker)
    mocker.patch.object(app, "RERANKER_MODEL", "some/model")
    get_reranker = mocker.patch("app._get_reranker", return_value=mocker.Mock())
    app.warmup_embedding_models()
    get_reranker.assert_called_once()  # boot'ta yüklendi


def test_warmup_skips_reranker_when_not_configured(app_with_mocks, mocker):
    app = _reset_warmup(mocker)
    mocker.patch.object(app, "RERANKER_MODEL", "")
    get_reranker = mocker.patch("app._get_reranker")
    app.warmup_embedding_models()
    get_reranker.assert_not_called()  # kapaliyken no-op


def test_warmup_survives_reranker_load_failure(app_with_mocks, mocker):
    """Reranker yüklenemezse boot DÜŞMEZ — servis reranker'sız ayağa kalkar."""
    app = _reset_warmup(mocker)
    mocker.patch.object(app, "RERANKER_MODEL", "bad/model")
    mocker.patch("app._get_reranker", return_value=None)  # yükleme başarısız
    mocker.patch.object(app, "_RERANKER_LOAD_ERROR", "offline")
    # Patlamamalı:
    app.warmup_embedding_models()
    assert app._startup_warmup_done is True
