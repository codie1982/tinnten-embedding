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


def test_lock_acquired_when_free(mocker):
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    doc = cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-A")
    assert doc is not None
    assert doc["index"]["lock"]["jobId"] == "job-A"
    assert doc["index"]["state"] == "processing"


def test_second_job_cannot_steal_live_lock(mocker):
    """Eşzamanlılık: lease dolmadan ikinci job kilidi ALAMAZ."""
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    assert cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-A") is not None
    assert cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-B") is None


def test_same_job_reacquires_lock_redelivery_idempotent(mocker):
    """RabbitMQ redelivery aynı job_id ile gelir → kendi kilidini yeniden alır."""
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    assert cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-A") is not None
    assert cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-A") is not None


def test_expired_lease_is_taken_over(mocker):
    """Worker ölürse kilit kalıcı olmamalı."""
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    cs.try_acquire_ingest_lock(
        company_id="c1", document_id="d1", job_id="olu-job", lease_seconds=-1
    )
    doc = cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="job-B")
    assert doc is not None
    assert doc["index"]["lock"]["jobId"] == "job-B"


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
    assert doc["index"]["state"] == "ready"


def test_stale_job_cannot_overwrite_new_jobs_state(mocker):
    """Lease'i dolup işi devralınan ESKİ job geri dönüp durumu ezememeli."""
    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    cs.try_acquire_ingest_lock(
        company_id="c1", document_id="d1", job_id="eski-job", lease_seconds=-1
    )
    cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="yeni-job")

    # Eski job "bitti" demeye calisiyor → REDDEDILMELI
    assert cs.release_ingest_lock(
        company_id="c1", document_id="d1", job_id="eski-job", state="failed"
    ) is False
    doc = cs.documents.find_one({"documentId": "d1"})
    assert doc["index"]["lock"]["jobId"] == "yeni-job"
    assert doc["index"]["state"] == "processing", "eski job yeni job'in durumunu ezdi"


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


def test_worker_skips_when_another_job_holds_lock(mocker):
    """Sarmalayıcı kilidi alamazsa asıl iş HİÇ çalışmamalı."""
    from workers.ingest_worker import IngestWorker

    cs = _content_store(mocker)
    cs.documents.insert_one({"companyId": "c1", "documentId": "d1"})
    cs.try_acquire_ingest_lock(company_id="c1", document_id="d1", job_id="baska-job")

    worker = _worker(mocker, cs)
    inner = mocker.patch.object(IngestWorker, "_process_single_document_locked")
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


def test_worker_proceeds_when_lock_backend_fails(mocker):
    """Kilit altyapısı patlarsa ingest DURMAZ — swap zaten tek başına idempotent."""
    from workers.ingest_worker import IngestWorker

    cs = _content_store(mocker)
    mocker.patch.object(
        cs, "try_acquire_ingest_lock", side_effect=RuntimeError("mongo down")
    )
    worker = _worker(mocker, cs)
    inner = mocker.patch.object(IngestWorker, "_process_single_document_locked")
    worker._process_single_document({}, _ctx("job-A"))
    inner.assert_called_once()
