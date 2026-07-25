"""
İndeks sağlığı ölçüm testleri.

CMS'teki drift göstergesi bu iki davranışa dayanır:
  1) `remove_ids` GERÇEKTEN silinen vektör sayısını döner (talep edileni değil).
     Aksi halde "silindi" raporu, hiçbir şey silinmediğinde bile başarı gösterir.
  2) `count()` diskten reload eder — ingest worker AYRI bir process olduğu için
     reload yapılmazsa bayat bir sayı okunur ve drift sessizce yanlış çıkar.
"""
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


DIM = 8


def _engine(path):
    """Gerçek FAISS + sahte model ile bir engine kurar (model yalnızca boyut verir)."""
    from services.embedding_engine import EmbeddingEngine

    EmbeddingEngine._model_cache.clear()
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = DIM
    with patch("services.embedding_engine.SentenceTransformer", return_value=model):
        return EmbeddingEngine(model_name="test-model", index_path=path)


def _vecs(n):
    arr = np.random.rand(n, DIM).astype("float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def test_remove_ids_returns_actual_removed_count():
    with tempfile.TemporaryDirectory() as tmp:
        eng = _engine(os.path.join(tmp, "a.index"))
        eng.add_embeddings(_vecs(3), [10, 11, 12])

        # 2'si var, 1'i (99) yok → gerçek silinen 2 olmalı, talep edilen 3 değil.
        assert eng.remove_ids([10, 11, 99]) == 2
        assert eng.count() == 1


def test_remove_ids_on_missing_index_is_zero_not_requested_count():
    with tempfile.TemporaryDirectory() as tmp:
        eng = _engine(os.path.join(tmp, "yok.index"))
        # Index dosyası hiç yok → hiçbir şey silinemez.
        assert eng.remove_ids([1, 2, 3]) == 0


def test_count_reloads_writes_from_another_process():
    """
    İki ayrı engine nesnesi = iki ayrı process'i taklit eder (API vs ingest worker).
    B yazdıktan sonra A'nın count()'u güncel değeri görmelidir.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "shared.index")
        a = _engine(path)
        b = _engine(path)

        a.add_embeddings(_vecs(2), [1, 2])
        assert a.count() == 2

        b.add_embeddings(_vecs(3), [3, 4, 5])
        # A kendi belleğinde 2 tutuyor; reload etmezse bayat değeri döner.
        assert a.count() == 5


def test_company_filter_matches_both_storage_locations():
    """
    `company_id` chunk'ta iki yerde olabiliyor (üst seviye VEYA metadata.companyId).
    Yalnız üst seviyeye bakmak, pre-embed/legacy yollarla yazılmış chunk'ları her
    firma filtreli aramada görünmez kılıyordu — global fallback'in işe yaraması da
    buna bağlı.
    """
    from app import _passes_chunk_filters

    flt = {"company_id": "CID"}
    assert _passes_chunk_filters({"company_id": "CID"}, flt)
    assert _passes_chunk_filters({"metadata": {"companyId": "CID"}}, flt)
    assert _passes_chunk_filters({"company_id": None, "metadata": {"companyId": "CID"}}, flt)
    # Başka firmaya sızmamalı
    assert not _passes_chunk_filters({"metadata": {"companyId": "OTHER"}}, flt)
    assert not _passes_chunk_filters({"metadata": {}}, flt)


def _mock_search_engine(mocker, hits=True, raises=False):
    eng = mocker.Mock()
    eng.index_path = "/tmp/x.index"
    eng.encode.return_value = np.zeros((1, 768), dtype=np.float32)
    if raises:
        eng.search.side_effect = RuntimeError("FAISS index is empty.")
    else:
        eng.search.return_value = (
            np.array([[0.9]]), np.array([[10]]) if hits else np.array([[-1]])
        )
    return eng


def _patch_chunk_lookups(mocker):
    mocker.patch(
        "app.chunk_store.get_chunks_by_faiss_ids",
        return_value={10: {"doc_id": "d1", "text": "t", "chunk_id": "c10", "metadata": {}}},
    )
    mocker.patch("app.chunk_store.get_documents_by_ids", return_value={"d1": {"status": "active"}})


def test_dual_read_falls_back_when_primary_index_missing(client, mocker):
    """
    Birincil index boş/yoksa arama SESSİZCE boş dönmemeli; fallback devreye girip
    hangi indeksin hizmet verdiğini `retrieval_meta.index_scope` ile bildirmeli.

    Per-company SABİT açık olduğu için birincil = firma index'i, fallback = global.
    Bu, migration henüz koşmamış firmayı temsil eder: vektörler hâlâ global'de.
    """
    mocker.patch("app.get_company_chunk_engine", return_value=_mock_search_engine(mocker, raises=True))
    fallback = _mock_search_engine(mocker, hits=True)
    mocker.patch("app.get_chunk_engine", return_value=fallback)
    _patch_chunk_lookups(mocker)

    resp = client.post(
        "/api/v10/content/search",
        json={"text": "q", "k": 1, "filter": {"company_id": "CID"}},
    )
    assert resp.status_code == 200
    meta = resp.get_json()["retrieval_meta"]
    assert meta["index_scope"] == "global_fallback"
    assert meta["fallback_reason"] == "index_missing_or_empty"
    fallback.search.assert_called_once()


def test_primary_hit_does_not_touch_fallback(client, mocker):
    """Birincil isabet verirse fallback motoru hiç KURULMAMALI (tembel plan)."""
    mocker.patch("app.get_company_chunk_engine", return_value=_mock_search_engine(mocker, hits=True))
    global_engine = mocker.patch("app.get_chunk_engine")
    _patch_chunk_lookups(mocker)

    resp = client.post(
        "/api/v10/content/search",
        json={"text": "q", "k": 1, "filter": {"company_id": "CID"}},
    )
    assert resp.status_code == 200
    assert resp.get_json()["retrieval_meta"]["index_scope"] == "company"
    global_engine.assert_not_called()


def test_metadata_company_filter_routes_to_company_index(client, mocker):
    """
    `metadata.companyId` formu da firma index'ine YÖNLENMELİ.

    Eskiden routing yalnız top-level anahtarlara bakıyordu; bu formu gönderen
    çağıran global'e düşer, hata almaz, sadece 0 sonuç görürdü — sessiz kesinti.
    """
    company_engine = _mock_search_engine(mocker, hits=True)
    resolver = mocker.patch("app.get_company_chunk_engine", return_value=company_engine)
    _patch_chunk_lookups(mocker)

    resp = client.post(
        "/api/v10/content/search",
        json={"text": "q", "k": 1, "filter": {"metadata.companyId": "CID"}},
    )
    assert resp.status_code == 200
    assert resp.get_json()["retrieval_meta"]["index_scope"] == "company"
    resolver.assert_called_with("CID")


def test_multi_company_filter_does_not_route_to_single_index(client, mocker):
    """
    Çok firmalı `$in` tek bir firma index'ine yönlendirilemez — global'den okunmalı.
    Aksi halde ilk firmanın index'i seçilir ve diğerlerinin sonuçları sessizce düşer.
    """
    mocker.patch("app.get_chunk_engine", return_value=_mock_search_engine(mocker, hits=True))
    resolver = mocker.patch("app.get_company_chunk_engine")
    _patch_chunk_lookups(mocker)

    resp = client.post(
        "/api/v10/content/search",
        json={"text": "q", "k": 1, "filter": {"companyId": {"$in": ["A", "B"]}}},
    )
    assert resp.status_code == 200
    assert resp.get_json()["retrieval_meta"]["index_scope"] == "global"
    resolver.assert_not_called()


def test_all_indexes_unavailable_reports_reason(client, mocker):
    """Hiçbir index arama yapamazsa boş sonuç DÖNER ama sebebi görünür olur."""
    mocker.patch("app.get_company_chunk_engine", return_value=_mock_search_engine(mocker, raises=True))
    mocker.patch("app.get_chunk_engine", return_value=_mock_search_engine(mocker, raises=True))

    resp = client.post(
        "/api/v10/content/search",
        json={"text": "q", "k": 1, "filter": {"company_id": "CID"}},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["results"] == []
    assert body["retrieval_meta"]["index_scope"] is None
    assert body["retrieval_meta"]["fallback_reason"] == "index_missing_or_empty"


@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_remove_response_separates_requested_from_removed(client, mocker, mode):
    """
    `faissRemoved` bir ÖLÇÜMdür; `faissRequested` ile arasındaki fark görünür olmalı.
    Yetim kalanlar `faissOrphaned` — "pending" değil, çünkü onları toplayacak bir
    janitor yazılamaz (chunk satırları aynı istekte siliniyor).
    """
    mock_engine = mocker.Mock()
    mock_engine.remove_ids.return_value = 1  # 3 talep edildi, 1 silinebildi
    mocker.patch("app.get_company_chunk_engine", return_value=mock_engine)
    mocker.patch(
        "app.chunk_store.get_chunks_by_doc",
        return_value=[{"faiss_id": 10}, {"faiss_id": 11}, {"faiss_id": 12}],
    )
    mocker.patch("app.chunk_store.delete_chunks_by_doc", return_value=3)
    mocker.patch("app.chunk_store.update_document_status")
    mocker.patch("app._deactivate_index_state")

    resp = client.post(
        "/api/v10/content/index/remove",
        json={"documentId": "d1", "companyId": "CID", "mode": mode},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["faissRequested"] == 3
    if mode == "hard":
        assert data["faissRemoved"] == 1
        assert data["faissOrphaned"] == 2
    else:
        # soft: FAISS'e hiç dokunulmaz → hepsi yetim kalır
        assert data["faissRemoved"] == 0
        assert data["faissOrphaned"] == 3


def test_remove_without_company_id_still_targets_company_index(client, mocker):
    """
    `companyId` OPSİYONEL ve tinnten-server her zaman göndermiyor.

    Firma chunk'ın KENDİSİNDEN çözülmezse silme global engine'e düşer ve sessiz
    bir no-op olur: chunk metadata gider, vektör firma index'inde SONSUZA DEK
    kalır (kimliği kaybolduğu için sonradan toplanamaz).
    """
    mock_engine = mocker.Mock()
    mock_engine.remove_ids.return_value = 2
    resolver = mocker.patch("app.get_company_chunk_engine", return_value=mock_engine)
    mocker.patch(
        "app.chunk_store.get_chunks_by_doc",
        return_value=[
            {"faiss_id": 10, "metadata": {"companyId": "CID"}},
            {"faiss_id": 11, "company_id": "CID"},
        ],
    )
    mocker.patch("app.chunk_store.delete_chunks_by_doc", return_value=2)
    mocker.patch("app.chunk_store.update_document_status")
    mocker.patch("app._deactivate_index_state")

    # companyId BİLEREK gönderilmiyor.
    resp = client.post(
        "/api/v10/content/index/remove",
        json={"documentId": "d1", "mode": "hard"},
    )
    assert resp.status_code == 200
    # Global'e değil, chunk'tan çözülen firmanın index'ine gitmeli.
    resolver.assert_called_once_with("CID")
    mock_engine.remove_ids.assert_called_once_with([10, 11])
    assert resp.get_json()["faissRemovedByCompany"] == {"CID": 2}
