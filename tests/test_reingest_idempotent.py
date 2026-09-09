"""
FAZ 5 — idempotent re-ingest: versiyonlu swap + lease'li CAS lock.

Neden önemli: `_chunk_and_embed` eskiden yalnız EKLİYORDU (eskiyi hiç silmezdi).
Fetcher her değişen sayfayı stabil `information_page_doc_id` ile yeniden
indexlediği için bu, CANLI bir duplicate chunk + yetim vektör kaynağıydı.

Burada kanıtlanan sözleşme:
  * aynı doküman iki kez index'lenince chunk sayısı SABİT kalır
  * FAISS `ntotal` sabit kalır (yetim vektör yok)
  * Mongo↔FAISS id'leri birebir örtüşür
  * adım ortasında hata → ESKİ sürüm bütün ve aktif kalır (telafi)
  * lease dolmadan ikinci job kilidi ALAMAZ; dolunca devralır
  * eski job, yeni job'ın durumunu ezemez
  * redelivery (aynı job_id) idempotenttir
"""
import mongomock
import numpy as np
import pytest
from bson import ObjectId

from services.mongo_store import MongoStore


# ---------------------------------------------------------------------------
# Sahte FAISS engine — ntotal/id kümesini gerçekçi biçimde izler
# ---------------------------------------------------------------------------
class FakeEngine:
    def __init__(self):
        self.ids: set[int] = set()
        self.add_calls = 0
        self.fail_add = False

    def add_embeddings(self, embeddings, ids):
        if self.fail_add:
            raise RuntimeError("FAISS add patladi")
        self.add_calls += 1
        self.ids |= {int(i) for i in ids}

    def encode(self, texts, *, batch_size):
        return np.ones((len(texts), 4), dtype=np.float32)

    def remove_ids(self, ids):
        self.ids -= {int(i) for i in ids}
        return len(list(ids))

    @property
    def ntotal(self) -> int:
        return len(self.ids)


def _store(mocker) -> MongoStore:
    """Gerçek MongoStore mantığı + mongomock koleksiyonları."""
    client = mongomock.MongoClient()
    store = MongoStore.__new__(MongoStore)
    store.document_db = client["tinnten"]
    store.chunk_db = client["tinnten-embedding"]
    store.documents = store.chunk_db["embedding_documents"]
    store.chunks = store.chunk_db["embedding_chunks"]
    store.counters = store.chunk_db["counters"]
    return store


# ---------------------------------------------------------------------------
# Sürüm-seçici repository — swap'ın temeli
# ---------------------------------------------------------------------------
def test_delete_by_version_leaves_other_version(mocker):
    store = _store(mocker)
    store.insert_chunks([
        {"doc_id": "d1", "faiss_id": 1, "ingest_version": "v1", "text": "eski"},
        {"doc_id": "d1", "faiss_id": 2, "ingest_version": "v2", "text": "yeni"},
    ])
    removed = store.delete_chunks_by_doc_version("d1", "v1")
    assert removed == 1
    kalan = store.get_chunks_by_doc("d1")
    assert [c["ingest_version"] for c in kalan] == ["v2"]


def test_delete_except_version_also_clears_legacy_unversioned(mocker):
    """Sürümsüz (legacy) chunk'lar da temizlenmeli — yoksa ilk versiyonlu
    ingest eski kayıtları geride bırakır ve duplicate görünür."""
    store = _store(mocker)
    store.insert_chunks([
        {"doc_id": "d1", "faiss_id": 1, "text": "legacy surumsuz"},          # ingest_version YOK
        {"doc_id": "d1", "faiss_id": 2, "ingest_version": "v_old", "text": "eski"},
        {"doc_id": "d1", "faiss_id": 3, "ingest_version": "v_new", "text": "yeni"},
    ])
    removed = store.delete_chunks_by_doc_except_version("d1", "v_new")
    assert removed == 2
    kalan = store.get_chunks_by_doc("d1")
    assert [c["faiss_id"] for c in kalan] == [3]


def test_delete_by_doc_is_version_blind_by_design(mocker):
    """`delete_chunks_by_doc` HER sürümü siler — bu yüzden swap'ta kullanılamaz.

    Bu davranış tam kaldırma (remove endpoint) için doğrudur; testi, birinin
    yanlışlıkla swap'a sokmasına karşı sözleşmeyi sabitler.
    """
    store = _store(mocker)
    store.insert_chunks([
        {"doc_id": "d1", "faiss_id": 1, "ingest_version": "v1"},
        {"doc_id": "d1", "faiss_id": 2, "ingest_version": "v2"},
    ])
    assert store.delete_chunks_by_doc("d1") == 2
    assert store.get_chunks_by_doc("d1") == []


def test_faiss_ids_except_version(mocker):
    store = _store(mocker)
    store.insert_chunks([
        {"doc_id": "d1", "faiss_id": 10, "ingest_version": "v1"},
        {"doc_id": "d1", "faiss_id": 11, "ingest_version": "v1"},
        {"doc_id": "d1", "faiss_id": 20, "ingest_version": "v2"},
    ])
    assert sorted(store.get_faiss_ids_by_doc_except_version("d1", "v2")) == [10, 11]


def test_set_active_ingest_version(mocker):
    store = _store(mocker)
    store.documents.insert_one({"doc_id": "d1", "status": "ready"})
    store.set_active_ingest_version("d1", "v9")
    assert store.documents.find_one({"doc_id": "d1"})["active_ingest_version"] == "v9"


def test_insert_chunks_adds_normalized_content_fingerprint(mocker):
    store = _store(mocker)
    store.insert_chunks([
        {"doc_id": "d1", "faiss_id": 1, "text": "  AYNI\n CONTENT  "},
        {"doc_id": "d1", "faiss_id": 2, "text": "ayni content"},
    ])
    rows = list(store.chunks.find({"doc_id": "d1"}).sort("faiss_id", 1))
    assert rows[0]["content_fingerprint"]
    assert rows[0]["content_fingerprint"] == rows[1]["content_fingerprint"]


def test_chunk_audit_page_is_bounded_and_keeps_previous_chunk(mocker):
    store = _store(mocker)
    store.insert_chunks([
        {"doc_id": "d1", "chunk_index": index, "faiss_id": index, "text": f"chunk {index}"}
        for index in range(45)
    ])

    result = store.get_chunks_page_by_doc("d1", page=2, limit=20)

    assert [row["chunk_index"] for row in result["chunks"]] == list(range(20, 40))
    assert result["previous_chunk"]["chunk_index"] == 19
    assert result["total"] == 45
    assert result["document_total"] == 45
    assert result["page"] == 2
    assert result["limit"] == 20
    assert result["total_pages"] == 3


def test_chunk_audit_page_searches_all_chunks_and_caps_page_size(mocker):
    store = _store(mocker)
    store.insert_chunks([
        {"doc_id": "d1", "chunk_index": index, "faiss_id": index, "text": f"metin {index}"}
        for index in range(130)
    ])
    store.insert_chunks([
        {"doc_id": "other", "chunk_index": 0, "faiss_id": 999, "text": "metin 42"},
    ])

    result = store.get_chunks_page_by_doc("d1", page=99, limit=500, query="metin 42")

    assert [row["chunk_index"] for row in result["chunks"]] == [42]
    assert result["previous_chunk"] is None
    assert result["total"] == 1
    assert result["document_total"] == 130
    assert result["page"] == 1
    assert result["limit"] == 100
    assert result["total_pages"] == 1


def test_duplicate_health_counts_only_active_document_versions(mocker):
    store = _store(mocker)
    store.documents.insert_many([
        {"doc_id": "d1", "status": "ready", "active_ingest_version": "v2"},
        {"doc_id": "d2", "status": "ready"},
        {"doc_id": "d3", "status": "disabled"},
    ])
    store.insert_chunks([
        {"doc_id": "d1", "company_id": "CID", "faiss_id": 1, "ingest_version": "v1", "text": "eski tekrar"},
        {"doc_id": "d1", "company_id": "CID", "faiss_id": 2, "ingest_version": "v2", "text": "Aynı içerik"},
        {"doc_id": "d1", "company_id": "CID", "faiss_id": 3, "ingest_version": "v2", "text": "aynı   içerik"},
        {"doc_id": "d2", "company_id": "CID", "faiss_id": 4, "text": "benzersiz"},
        {"doc_id": "d3", "company_id": "CID", "faiss_id": 5, "text": "aynı içerik"},
    ])

    result = store.active_chunk_duplicate_stats_by_company("CID", 100)

    assert result == {
        "scanned": 3,
        "unique": 2,
        "duplicates": 1,
        "ratio": 0.333333,
        "sampled": False,
        "activeTotal": 3,
    }


def test_embedding_profile_reports_mixed_models_and_legacy_unknowns(mocker):
    store = _store(mocker)
    store.documents.insert_many([
        {"doc_id": "d1", "status": "ready", "active_ingest_version": "v2"},
        {"doc_id": "d2", "status": "ready"},
        {"doc_id": "disabled", "status": "disabled"},
    ])
    store.insert_chunks([
        {"doc_id": "d1", "company_id": "CID", "faiss_id": 1, "ingest_version": "v1", "text": "old", "embedding_model": "old-model", "embedding_dimension": 384},
        {"doc_id": "d1", "company_id": "CID", "faiss_id": 2, "ingest_version": "v2", "text": "new", "embedding_model": "new-model", "embedding_dimension": 768},
        {"doc_id": "d2", "company_id": "CID", "faiss_id": 3, "text": "other", "embedding_model": "other-model", "embedding_dimension": 768},
        {"doc_id": "d2", "company_id": "CID", "faiss_id": 4, "text": "legacy"},
        {"doc_id": "disabled", "company_id": "CID", "faiss_id": 5, "text": "ignore", "embedding_model": "old-model", "embedding_dimension": 384},
    ])

    result = store.active_chunk_embedding_profile_by_company("CID", 100)

    assert result == {
        "scanned": 3,
        "sampled": False,
        "unknown": 1,
        "unknownRatio": 0.333333,
        "models": [
            {"model": "new-model", "count": 1},
            {"model": "other-model", "count": 1},
        ],
        "dimensions": [{"dimension": 768, "count": 2}],
    }


# ---------------------------------------------------------------------------
# Swap sözleşmesi — worker'ın _swap_in_chunks mantığını birebir uygular
# ---------------------------------------------------------------------------
def _swap(store, engine, doc_id, texts, *, fail_insert=False):
    """`IngestWorker._swap_in_chunks` ile aynı sıra: ekle → swap → temizle."""
    import uuid as _uuid

    version = _uuid.uuid4().hex
    faiss_ids = store.reserve_faiss_ids(len(texts))
    engine.add_embeddings(np.zeros((len(texts), 4), dtype=np.float32), faiss_ids)
    try:
        if fail_insert:
            raise RuntimeError("Mongo insert patladi")
        store.insert_chunks([
            {"doc_id": doc_id, "faiss_id": int(f), "ingest_version": version,
             "chunk_index": i, "text": t}
            for i, (t, f) in enumerate(zip(texts, faiss_ids))
        ])
        store.set_active_ingest_version(doc_id, version)
    except Exception:
        engine.remove_ids(faiss_ids)                       # telafi
        store.delete_chunks_by_doc_version(doc_id, version)
        raise
    old = store.get_faiss_ids_by_doc_except_version(doc_id, version)
    if old:
        engine.remove_ids(old)
    store.delete_chunks_by_doc_except_version(doc_id, version)
    return version


def test_reingest_twice_keeps_chunk_count_and_ntotal_stable(mocker):
    """Çekirdek regresyon: canlı duplicate bug'ı."""
    store = _store(mocker)
    store.documents.insert_one({"doc_id": "d1", "status": "ready"})
    engine = FakeEngine()

    _swap(store, engine, "d1", ["a", "b", "c"])
    assert len(store.get_chunks_by_doc("d1")) == 3
    assert engine.ntotal == 3

    _swap(store, engine, "d1", ["a", "b", "c"])            # AYNI icerik tekrar
    assert len(store.get_chunks_by_doc("d1")) == 3, "duplicate chunk olustu"
    assert engine.ntotal == 3, "yetim vektor kaldi"


def test_reingest_with_fewer_chunks_shrinks(mocker):
    store = _store(mocker)
    store.documents.insert_one({"doc_id": "d1", "status": "ready"})
    engine = FakeEngine()
    _swap(store, engine, "d1", ["a", "b", "c", "d"])
    assert engine.ntotal == 4
    _swap(store, engine, "d1", ["a"])                      # icerik kisaldi
    assert len(store.get_chunks_by_doc("d1")) == 1
    assert engine.ntotal == 1


def test_mongo_and_faiss_ids_match_after_reingest(mocker):
    store = _store(mocker)
    store.documents.insert_one({"doc_id": "d1", "status": "ready"})
    engine = FakeEngine()
    _swap(store, engine, "d1", ["a", "b"])
    _swap(store, engine, "d1", ["x", "y", "z"])
    mongo_ids = {c["faiss_id"] for c in store.get_chunks_by_doc("d1")}
    assert mongo_ids == engine.ids, "Mongo ile FAISS id kumeleri ayristi"


def test_failure_mid_swap_leaves_old_version_intact(mocker):
    """Fault injection: Mongo insert adımı patlarsa eski sürüm bozulmamalı."""
    store = _store(mocker)
    store.documents.insert_one({"doc_id": "d1", "status": "ready"})
    engine = FakeEngine()

    v1 = _swap(store, engine, "d1", ["saglam1", "saglam2"])
    onceki_ids = set(engine.ids)

    with pytest.raises(RuntimeError, match="Mongo insert patladi"):
        _swap(store, engine, "d1", ["yeni"], fail_insert=True)

    # Eski sürüm hâlâ aktif, bütün ve tek
    doc = store.documents.find_one({"doc_id": "d1"})
    assert doc["active_ingest_version"] == v1
    kalan = store.get_chunks_by_doc("d1")
    assert len(kalan) == 2
    assert all(c["ingest_version"] == v1 for c in kalan)
    assert engine.ids == onceki_ids, "telafi calismadi, yetim vektor kaldi"


def test_failure_on_faiss_add_writes_nothing(mocker):
    store = _store(mocker)
    store.documents.insert_one({"doc_id": "d1", "status": "ready"})
    engine = FakeEngine()
    v1 = _swap(store, engine, "d1", ["saglam"])
    engine.fail_add = True
    with pytest.raises(RuntimeError, match="FAISS add patladi"):
        _swap(store, engine, "d1", ["yeni"])
    assert store.documents.find_one({"doc_id": "d1"})["active_ingest_version"] == v1
    assert len(store.get_chunks_by_doc("d1")) == 1


def test_ownership_loss_before_activation_compensates_new_version(mocker):
    """Lease kaybeden worker görünürlüğü çeviremez ve yeni yazılarını temizler."""
    from workers.ingest_worker import IngestWorker, RetryableIngestLockError

    store = _store(mocker)
    store.create_document(doc_id="d1", job_id="job-A", status="ready")
    engine = FakeEngine()
    old_version = _swap(store, engine, "d1", ["eski"])
    old_ids = set(engine.ids)

    worker = IngestWorker.__new__(IngestWorker)
    worker._get_store = mocker.MagicMock(return_value=store)

    def build_docs(ids, version):
        return [
            {
                "doc_id": "d1",
                "faiss_id": int(ids[0]),
                "ingest_version": version,
                "chunk_index": 0,
                "text": "yeni",
            }
        ]

    with pytest.raises(RetryableIngestLockError, match="ownership lost"):
        worker._swap_in_chunks(
            doc_id="d1",
            engine=engine,
            embeddings=np.ones((1, 4), dtype=np.float32),
            build_chunk_docs=build_docs,
            chunk_count=1,
            ownership_guard=lambda: False,
            expected_job_id="job-A",
        )

    doc = store.get_document("d1")
    assert doc["active_ingest_version"] == old_version
    assert engine.ids == old_ids
    assert [row["text"] for row in store.get_chunks_by_doc("d1")] == ["eski"]


def test_search_only_sees_active_version_during_swap_window(app_with_mocks, mocker):
    """`_assemble_chunk_result` aktif olmayan sürümü elemeli.

    Aksi halde 'önce ekle' penceresinde aynı içerik iki kez döner.
    """
    import app

    doc_status_map = {"d1": {"status": "active", "active_ingest_version": "v2"}}
    eski = {"doc_id": "d1", "chunk_id": "c1", "text": "eski", "ingest_version": "v1", "metadata": {}}
    yeni = {"doc_id": "d1", "chunk_id": "c2", "text": "yeni", "ingest_version": "v2", "metadata": {}}

    assert app._assemble_chunk_result(eski, 1, 0.9, {}, 0, doc_status_map, {}) is None
    row = app._assemble_chunk_result(yeni, 2, 0.9, {}, 0, doc_status_map, {})
    assert row is not None and row["chunk_id"] == "c2"


def test_search_unaffected_when_doc_has_no_active_version(app_with_mocks, mocker):
    """Legacy doküman (sürüm etiketi yok) → süzme YAPILMAZ, geriye uyumlu."""
    import app

    doc_status_map = {"d1": {"status": "active"}}   # active_ingest_version YOK
    chunk = {"doc_id": "d1", "chunk_id": "c1", "text": "legacy", "metadata": {}}
    row = app._assemble_chunk_result(chunk, 1, 0.9, {}, 0, doc_status_map, {})
    assert row is not None and row["chunk_id"] == "c1"


def test_parent_child_context_expands_only_active_revision(app_with_mocks, mocker):
    """Child hit gains neighbor context, but never from a stale document version."""
    import app

    doc_status_map = {"d1": {"status": "active", "active_ingest_version": "v2"}}
    chunks = [
        {"doc_id": "d1", "chunk_id": "old", "chunk_index": 0, "text": "STALE", "char_start": 0, "char_end": 5, "ingest_version": "v1", "metadata": {}},
        {"doc_id": "d1", "chunk_id": "a", "chunk_index": 0, "text": "Alpha ", "char_start": 0, "char_end": 6, "ingest_version": "v2", "metadata": {}},
        {"doc_id": "d1", "chunk_id": "b", "chunk_index": 1, "text": "Beta", "char_start": 6, "char_end": 10, "ingest_version": "v2", "metadata": {}},
    ]
    mocker.patch.object(app.chunk_store, "get_chunks_by_doc", return_value=chunks)

    row = app._assemble_chunk_result(chunks[2], 2, 0.9, {}, 1, doc_status_map, {})

    assert row["match_text"] == "Beta"
    assert row["text"] == "Alpha Beta"
    assert row["context_expanded"] is True
    assert row["context_chunk_ids"] == ["a", "b"]
    assert "STALE" not in row["text"]


# ---------------------------------------------------------------------------
# Lease'li CAS lock
# ---------------------------------------------------------------------------
def _content_store(mocker):
    from services.content_store import ContentDocumentStore

    client = mongomock.MongoClient()
    store = ContentDocumentStore.__new__(ContentDocumentStore)
    store.db = client["tinnten"]
    store.documents = store.db["contentdocuments"]
    store.logs = store.db["contentdocumentlogs"]
    return store


def test_get_documents_without_company_supports_canonical_and_legacy_ids(mocker):
    """Personal-library lookup must not depend on a company filter variable."""
    cs = _content_store(mocker)
    legacy_id = ObjectId()
    cs.documents.insert_many(
        [
            {"companyId": "c1", "documentId": "canonical"},
            {"_id": legacy_id, "title": "legacy"},
        ]
    )

    documents = cs.get_documents(None, ["canonical", str(legacy_id)])

    assert set(documents) == {"canonical", str(legacy_id)}


def test_legacy_id_lookup_remains_company_scoped(mocker):
    cs = _content_store(mocker)
    company_id = ObjectId()
    other_company_id = ObjectId()
    document_id = ObjectId()
    cs.documents.insert_one({"_id": document_id, "companyid": company_id})

    assert cs.get_document(str(company_id), str(document_id)) is not None
    assert cs.get_document(str(other_company_id), str(document_id)) is None


def test_upsert_preserves_node_source_enum_and_stores_ingest_payload_under_index(mocker):
    cs = _content_store(mocker)
    company_id = ObjectId()
    user_id = ObjectId()
    document_id = ObjectId()
    cs.documents.insert_one(
        {
            "_id": document_id,
            "companyid": company_id,
            "userid": user_id,
            "name": "Dosya",
            "type": "pdf",
            "source": "upload",
            "uploadid": "UP1",
            "index": {"state": "not_indexed"},
        }
    )

    updated = cs.upsert_document_with_source(
        company_id=str(company_id),
        document_id=str(document_id),
        source={"type": "upload", "uploadId": "UP1"},
        metadata={"filename": "rapor.pdf"},
        options={"chunkSize": 900, "chunkOverlap": 120, "minChars": 80},
        job_id="job-A",
        user_id=str(user_id),
    )

    assert updated["source"] == "upload"
    assert updated["index"]["source"] == {"type": "upload", "uploadId": "UP1"}
    assert updated["index"]["state"] == "queued"


def test_delayed_enqueue_cannot_replace_a_newer_job(mocker):
    from services.content_store import IndexJobConflictError

    cs = _content_store(mocker)
    company_id = ObjectId()
    document_id = ObjectId()
    cs.documents.insert_one(
        {
            "_id": document_id,
            "companyid": company_id,
            "source": "upload",
            "index": {"jobId": "job-new", "state": "queued"},
        }
    )

    with pytest.raises(IndexJobConflictError) as caught:
        cs.upsert_document_with_source(
            company_id=str(company_id),
            document_id=str(document_id),
            source={"type": "upload", "uploadId": "UP1"},
            metadata={"filename": "report.pdf"},
            options={"chunkSize": 900, "chunkOverlap": 120, "minChars": 80},
            job_id="job-old-delayed",
            expected_job_id="job-before-both",
        )

    assert caught.value.current_job_id == "job-new"
    stored = cs.documents.find_one({"_id": document_id})
    assert stored["index"] == {"jobId": "job-new", "state": "queued"}


def test_monotonic_attempt_rejects_older_and_equal_different_jobs(mocker):
    from services.content_store import IndexJobConflictError

    cs = _content_store(mocker)
    company_id = ObjectId()
    document_id = ObjectId()
    cs.documents.insert_one(
        {
            "_id": document_id,
            "companyid": company_id,
            "source": "upload",
            "index": {"attempt": 4, "jobId": "job-4", "state": "queued"},
        }
    )
    base = {
        "company_id": str(company_id),
        "document_id": str(document_id),
        "source": {"type": "upload", "uploadId": "UP1"},
        "metadata": {"filename": "report.pdf"},
        "options": {"chunkSize": 900, "chunkOverlap": 120, "minChars": 80},
    }

    with pytest.raises(IndexJobConflictError):
        cs.upsert_document_with_source(**base, job_id="job-3", attempt=3)
    with pytest.raises(IndexJobConflictError):
        cs.upsert_document_with_source(**base, job_id="other-job-4", attempt=4)
    with pytest.raises(IndexJobConflictError):
        cs.upsert_document_with_source(**base, job_id="legacy-job", attempt=None)

    same = cs.upsert_document_with_source(**base, job_id="job-4", attempt=4)
    assert same["index"]["attempt"] == 4
    assert same["index"]["jobId"] == "job-4"

    newer = cs.upsert_document_with_source(**base, job_id="job-5", attempt=5)
    assert newer["index"]["attempt"] == 5
    assert newer["index"]["jobId"] == "job-5"


def test_worker_audit_log_uses_node_objectid_contract(mocker):
    cs = _content_store(mocker)
    company_id = ObjectId()
    user_id = ObjectId()
    document_id = ObjectId()
    cs.documents.insert_one(
        {
            "_id": document_id,
            "companyid": company_id,
            "userid": user_id,
        }
    )

    cs.append_log_entry(
        company_id=str(company_id),
        document_id=str(document_id),
        job_id="job-A",
        level="error",
        message="Embedding failed",
        state="failed",
        user_id=None,
        trigger="manual",
        details={"stage": "embedding"},
    )

    row = cs.logs.find_one({"documentId": document_id})
    assert row["companyid"] == company_id
    assert row["userid"] == user_id
    assert row["event"] == "index.failed"
    assert row["meta"]["jobId"] == "job-A"
    assert row["meta"]["details"] == {"stage": "embedding"}
    assert "level" not in row


def test_lock_acquired_when_free(mocker):
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    doc = cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-A")
    assert doc is not None
    assert doc["index"]["lock"]["jobId"] == "job-A"
    assert doc["index"]["state"] == "indexing"


def test_second_job_cannot_steal_live_lock(mocker):
    """Eşzamanlılık: lease dolmadan ikinci job kilidi ALAMAZ."""
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    assert cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-A") is not None
    assert cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-B") is None


def test_same_job_cannot_reacquire_live_lock(mocker):
    """Aynı job'ın eşzamanlı redelivery'si de canlı sahibi geçemez."""
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    assert cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-A") is not None
    assert cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-A") is None


def test_expired_lease_is_taken_over(mocker):
    """Worker ölürse kilit kalıcı olmamalı."""
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    cs.try_acquire_ingest_lock(
        company_id="c1", document_id="d1", job_id="olu-job", lease_seconds=-1
    )
    cs.documents.update_one(
        {"companyId": "c1", "documentId": "d1"},
        {"$set": {"index.jobId": "job-B", "index.state": "queued"}},
    )
    doc = cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-B")
    assert doc is not None
    assert doc["index"]["lock"]["jobId"] == "job-B"


def test_expired_lease_does_not_let_stale_job_replace_new_current_job(mocker):
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    cs.try_acquire_ingest_lock(
        company_id="c1", document_id="d1", job_id="job-A", lease_seconds=-1
    )
    cs.documents.update_one(
        {"companyId": "c1", "documentId": "d1"},
        {"$set": {"index.jobId": "job-B", "index.state": "queued"}},
    )

    assert cs.try_acquire_ingest_lock(
        company_id="c1", document_id="d1", job_id="job-A"
    ) is None
    stored = cs.documents.find_one({"documentId": "d1"})
    assert stored["index"]["jobId"] == "job-B"
    assert stored["index"]["state"] == "queued"


def test_release_only_by_owner(mocker):
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-A")
    assert cs.release_ingest_lock(company_id="c1", document_id="d1", job_id="job-B") is False
    assert cs.documents.find_one({"documentId": "d1"})["index"]["lock"]["jobId"] == "job-A"
    assert cs.release_ingest_lock(
        company_id="c1", document_id="d1", job_id="job-A", state="ready"
    ) is True
    doc = cs.documents.find_one({"documentId": "d1"})
    assert doc["index"]["lock"] is None
    assert doc["index"]["state"] == "indexed"


def test_stale_job_cannot_overwrite_new_jobs_state(mocker):
    """Lease'i dolup işi devralınan ESKİ job geri dönüp durumu ezememeli."""
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    cs.try_acquire_ingest_lock(
        company_id="c1", document_id="d1", job_id="eski-job", lease_seconds=-1
    )
    cs.documents.update_one(
        {"companyId": "c1", "documentId": "d1"},
        {"$set": {"index.jobId": "yeni-job", "index.state": "queued"}},
    )
    cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="yeni-job")

    # Eski job "bitti" demeye calisiyor → REDDEDILMELI
    assert cs.release_ingest_lock(
        company_id="c1", document_id="d1", job_id="eski-job", state="failed"
    ) is False
    doc = cs.documents.find_one({"documentId": "d1"})
    assert doc["index"]["lock"]["jobId"] == "yeni-job"
    assert doc["index"]["state"] == "indexing", "eski job yeni job'in durumunu ezdi"


def test_renew_lease_only_by_owner(mocker):
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-A")
    assert cs.renew_ingest_lock(company_id="c1", document_id="d1", job_id="job-B") is False
    assert cs.renew_ingest_lock(company_id="c1", document_id="d1", job_id="job-A") is True


def test_lock_works_without_company_id_personal_docs(mocker):
    cs = _content_store(mocker)
    cs.documents.insert_one({"documentId": "d1"})
    doc = cs.try_acquire_ingest_lock(company_id=None, document_id="d1", job_id="job-A")
    assert doc is not None and doc["index"]["lock"]["jobId"] == "job-A"


def test_lock_renew_and_release_work_for_mongoose_legacy_identity(mocker):
    cs = _content_store(mocker)
    company_id = ObjectId()
    document_id = ObjectId()
    cs.documents.insert_one(
        {
            "_id": document_id,
            "companyid": company_id,
            "index": {"jobId": "job-A", "state": "queued"},
        }
    )

    acquired = cs.try_acquire_ingest_lock(
        company_id=str(company_id), document_id=str(document_id), job_id="job-A"
    )
    assert acquired is not None
    assert cs.renew_ingest_lock(
        company_id=str(company_id), document_id=str(document_id), job_id="job-A"
    ) is True
    assert cs.release_ingest_lock(
        company_id=str(company_id),
        document_id=str(document_id),
        job_id="job-A",
        state="completed",
    ) is True
    stored = cs.documents.find_one({"_id": document_id})
    assert stored["index"]["lock"] is None
    assert stored["index"]["state"] == "indexed"


def test_lock_works_for_personal_mongoose_legacy_id(mocker):
    cs = _content_store(mocker)
    document_id = ObjectId()
    cs.documents.insert_one({"_id": document_id})

    acquired = cs.try_acquire_ingest_lock(
        company_id=None, document_id=str(document_id), job_id="job-A"
    )

    assert acquired is not None
    assert acquired["index"]["lock"]["jobId"] == "job-A"


def test_index_state_update_uses_job_cas_on_legacy_identity(mocker):
    cs = _content_store(mocker)
    company_id = ObjectId()
    document_id = ObjectId()
    cs.documents.insert_one(
        {
            "_id": document_id,
            "companyid": company_id,
            "index": {"jobId": "job-new", "state": "queued"},
        }
    )

    stale = cs.update_index_fields(
        company_id=str(company_id),
        document_id=str(document_id),
        job_id="job-old",
        state="completed",
    )
    assert stale is None
    assert cs.documents.find_one({"_id": document_id})["index"]["state"] == "queued"

    current = cs.update_index_fields(
        company_id=str(company_id),
        document_id=str(document_id),
        job_id="job-new",
        state="completed",
    )
    assert current is not None
    assert current["index"]["state"] == "indexed"


def test_stale_release_only_clears_old_lease_after_new_job_is_queued(mocker):
    cs = _content_store(mocker)
    cs.documents.insert_one(
        {"companyId": "c1", "documentId": "d1", "index": {"jobId": "job-A"}}
    )
    assert cs.try_acquire_ingest_lock(
        company_id="c1", document_id="d1", job_id="job-A"
    ) is not None

    # Enqueue job B while A still owns the lease.
    marker = "new-job-timestamp"
    cs.documents.update_one(
        {"companyId": "c1", "documentId": "d1"},
        {
            "$set": {
                "index.jobId": "job-B",
                "index.state": "queued",
                "index.lastRunAt": marker,
            }
        },
    )

    assert cs.release_ingest_lock(
        company_id="c1",
        document_id="d1",
        job_id="job-A",
        state="completed",
    ) is False
    stored = cs.documents.find_one({"companyId": "c1", "documentId": "d1"})
    assert stored["index"]["lock"] is None
    assert stored["index"]["jobId"] == "job-B"
    assert stored["index"]["state"] == "queued"
    assert stored["index"]["lastRunAt"] == marker
    assert "finishedAt" not in stored["index"]


# ---------------------------------------------------------------------------
# Worker sarmalayıcısı — kilidin GERÇEKTEN bağlı olduğunu kanıtlar
# ---------------------------------------------------------------------------
def _worker(mocker, content_store):
    from workers.ingest_worker import IngestWorker

    worker = IngestWorker.__new__(IngestWorker)
    worker.ingest_lease_seconds = 900
    mocker.patch.object(IngestWorker, "_get_content_store", return_value=content_store)
    return worker


def _ctx(job_id="job-A"):
    from workers.ingest_worker import DocumentJobContext

    ctx = DocumentJobContext.__new__(DocumentJobContext)
    ctx.company_id = "c1"
    ctx.document_id = "d1"
    ctx.job_id = job_id
    return ctx


def test_worker_requeues_when_another_job_holds_lock(mocker):
    """Sarmalayıcı kilidi alamazsa asıl iş çalışmaz ve mesaj retry olur."""
    from workers.ingest_worker import IngestWorker, RetryableIngestLockError

    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="baska-job")

    worker = _worker(mocker, cs)
    inner = mocker.patch.object(IngestWorker, "_process_single_document_locked")
    with pytest.raises(RetryableIngestLockError, match="lock busy"):
        worker._process_single_document({}, _ctx("job-A"))
    inner.assert_not_called()


def test_worker_releases_lock_after_success(mocker):
    from workers.ingest_worker import IngestWorker

    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    worker = _worker(mocker, cs)
    mocker.patch.object(IngestWorker, "_process_single_document_locked")
    worker._process_single_document({}, _ctx("job-A"))
    assert cs.documents.find_one({"documentId": "d1"})["index"]["lock"] is None


def test_worker_releases_lock_even_when_work_raises(mocker):
    """Asıl iş patlasa bile kilit bırakılmalı — yoksa doküman kalıcı kilitli kalır."""
    from workers.ingest_worker import IngestWorker

    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    worker = _worker(mocker, cs)
    mocker.patch.object(
        IngestWorker, "_process_single_document_locked", side_effect=RuntimeError("ingest patladi")
    )
    with pytest.raises(RuntimeError, match="ingest patladi"):
        worker._process_single_document({}, _ctx("job-A"))
    assert cs.documents.find_one({"documentId": "d1"})["index"]["lock"] is None


def test_worker_renews_lease_and_checks_owner_during_long_ingest(mocker):
    import time
    from workers.ingest_worker import IngestWorker

    content_store = mocker.MagicMock()
    content_store.try_acquire_ingest_lock.return_value = {"documentId": "d1"}
    content_store.renew_ingest_lock.return_value = True
    content_store.owns_ingest_lock.return_value = True
    worker = _worker(mocker, content_store)
    worker.ingest_heartbeat_seconds = 0.01

    def work(document, context, *, ownership_guard):
        time.sleep(0.04)
        assert ownership_guard() is True

    mocker.patch.object(IngestWorker, "_process_single_document_locked", side_effect=work)

    worker._process_single_document({}, _ctx("job-A"))

    assert content_store.renew_ingest_lock.call_count >= 1
    content_store.owns_ingest_lock.assert_called_once()
    content_store.release_ingest_lock.assert_called_once()


def test_worker_requeues_when_lock_backend_fails(mocker):
    """Kilit altyapısı patlarsa kilitsiz devam edilmez."""
    from workers.ingest_worker import IngestWorker, RetryableIngestLockError

    cs = _content_store(mocker)
    mocker.patch.object(
        cs, "try_acquire_ingest_lock", side_effect=RuntimeError("mongo down")
    )
    worker = _worker(mocker, cs)
    inner = mocker.patch.object(IngestWorker, "_process_single_document_locked")
    with pytest.raises(RetryableIngestLockError, match="backend unavailable"):
        worker._process_single_document({}, _ctx("job-A"))
    inner.assert_not_called()


def test_modern_content_ingest_creates_ready_embedding_parent_and_active_chunks(mocker):
    from workers.ingest_worker import DocumentJobContext, IngestWorker

    store = _store(mocker)
    engine = FakeEngine()
    worker = IngestWorker.__new__(IngestWorker)
    worker.chunk_size = 900
    worker.chunk_overlap = 120
    worker.batch_size = 8
    worker.email_events = mocker.MagicMock()
    worker._get_store = mocker.MagicMock(return_value=store)
    worker._engine_for_company = mocker.MagicMock(return_value=engine)
    worker._resolve_document_source = mocker.MagicMock(
        return_value={
            "source": "upload",
            "doc_type": "pdf",
            "upload_id": "UP1",
            "metadata": {"filename": "rapor.pdf"},
        }
    )
    worker._load_document_content = mocker.MagicMock(
        return_value=("Gerçek ve aranabilir PDF metni", {"filename": "rapor.pdf"})
    )
    worker._safe_update_index_state = mocker.MagicMock(return_value=True)
    worker._log_document_event = mocker.MagicMock()
    worker._notify_index_failure = mocker.MagicMock()
    worker._log_worker_error = mocker.MagicMock()
    worker._mark_file_source_index_failed = mocker.MagicMock()
    worker._mark_file_source_index_completed = mocker.MagicMock()
    context = DocumentJobContext(
        company_id="C1",
        document_id="D1",
        job_id="J1",
        user_id="U1",
        trigger="upload_scan_clean",
        options={
            "chunkSize": 900,
            "chunkOverlap": 120,
            "minChars": 80,
            "cleanup": True,
            "ocr": False,
            "langDetect": False,
        },
    )

    worker._process_single_document_locked(
        {"title": "Rapor"}, context, ownership_guard=lambda: True
    )

    parent = store.get_document("D1")
    assert parent is not None
    assert parent["status"] == "ready"
    assert parent["job_id"] == "J1"
    assert parent["company_id"] == "C1"
    chunks = store.get_chunks_by_doc("D1")
    assert len(chunks) == 1
    assert chunks[0]["ingest_version"] == parent["active_ingest_version"]
    assert chunks[0]["company_id"] == "C1"


def _content_message_worker(mocker, document):
    from workers.ingest_worker import IngestWorker

    worker = IngestWorker.__new__(IngestWorker)
    worker.chunk_size = 1000
    worker.chunk_overlap = 100
    content_store = mocker.MagicMock()
    content_store.get_documents.return_value = {"d1": document}
    mocker.patch.object(worker, "_get_content_store", return_value=content_store)
    process = mocker.patch.object(worker, "_process_single_document")
    return worker, process


def test_worker_discards_message_when_document_has_a_newer_job(mocker):
    worker, process = _content_message_worker(
        mocker,
        {"documentId": "d1", "index": {"jobId": "job-new", "options": {}}},
    )

    worker._process_content_index_message(
        {"companyId": "c1", "documentIds": ["d1"], "jobId": "job-old"}
    )

    process.assert_not_called()


def test_worker_discards_unversioned_or_stale_attempt_message_for_modern_job(mocker):
    document = {
        "documentId": "d1",
        "index": {"jobId": "job-current", "attempt": 8, "options": {}},
    }
    worker, process = _content_message_worker(mocker, document)

    worker._process_content_index_message(
        {"companyId": "c1", "documentIds": ["d1"], "jobId": "job-current"}
    )
    worker._process_content_index_message(
        {
            "companyId": "c1",
            "documentIds": ["d1"],
            "jobId": "job-current",
            "attempt": 7,
        }
    )

    process.assert_not_called()


def test_worker_propagates_matching_attempt_into_job_context(mocker):
    worker, process = _content_message_worker(
        mocker,
        {
            "documentId": "d1",
            "index": {"jobId": "job-current", "attempt": 8, "options": {}},
        },
    )

    worker._process_content_index_message(
        {
            "companyId": "c1",
            "documentIds": ["d1"],
            "jobId": "job-current",
            "attempt": 8,
        }
    )

    assert process.call_args.args[1].attempt == 8


def test_worker_uses_payload_job_for_legacy_document_without_job_id(mocker):
    worker, process = _content_message_worker(
        mocker,
        {"documentId": "d1", "index": {"options": {}}},
    )

    worker._process_content_index_message(
        {"companyId": "c1", "documentIds": ["d1"], "jobId": "job-message"}
    )

    context = process.call_args.args[1]
    assert context.job_id == "job-message"
