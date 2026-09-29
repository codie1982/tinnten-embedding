"""
FAZ 2c — index-state callback body sözleşmesi.
Per-sayfa (fetcher_page/initial) doc'larda callback body'si `metadata.domain` +
`stats.domainChunks` taşımalı ki server website ENTRY'sini domain ile bulup
badge'i güncellesin. Diğer source'larda bu alanlar OLMAMALI.
"""
import queue
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import mongomock

from services.tinnten_server_client import TinntenServerClient


def _client():
    with patch("services.tinnten_server_client.get_keycloak_service") as kc:
        kc.return_value = MagicMock(get_service_token=lambda: "tok")
        c = TinntenServerClient()
        c.base_url = "http://server:5001"
        return c


def _capture_patch_body(client, **kwargs):
    with patch("services.tinnten_server_client.requests.patch") as pat:
        pat.return_value = MagicMock(status_code=200)
        client.update_document_index_state("b55eaea10f281ccd07647aa5", "completed", **kwargs)
        return pat.call_args.kwargs["json"]


def test_fetcher_page_callback_carries_domain_and_domain_chunks():
    body = _capture_patch_body(
        _client(),
        company_id="C1",
        domain="example.com",
        source="fetcher_page",
        page_url="https://example.com/pricing",
        page_title="Pricing",
        page_source_subscription_id="sub-1",
        domain_chunks=42,
    )
    assert body["state"] == "indexed"  # completed → indexed
    assert body["metadata"] == {
        "domain": "example.com",
        "source": "fetcher_page",
        "documentId": "b55eaea10f281ccd07647aa5",
        "url": "https://example.com/pricing",
        "title": "Pricing",
        "sourceSubscriptionId": "sub-1",
    }
    assert body["stats"]["domainChunks"] == 42
    assert body["companyid"] == "C1"


def test_non_fetcher_source_omits_domain_metadata():
    body = _capture_patch_body(
        _client(),
        company_id="C1",
        stats={"chunkCount": 5},
        # domain/source verilmedi (ör. upload/library)
    )
    assert "metadata" not in body
    assert "domainChunks" not in body.get("stats", {})


def test_callback_carries_the_ingest_job_id_for_precise_operation_matching():
    body = _capture_patch_body(_client(), company_id="C1", job_id="space-index:op-42")
    assert body["jobId"] == "space-index:op-42"


def test_callback_carries_monotonic_attempt():
    body = _capture_patch_body(_client(), company_id="C1", job_id="job-7", attempt=7)
    assert body["attempt"] == 7


def test_callback_preserves_rich_worker_stats_and_legacy_aliases():
    body = _capture_patch_body(
        _client(),
        company_id="C1",
        stats={
            "chunkCount": 3,
            "tokenCount": 44,
            "charCount": 321,
            "meaningfulCharCount": 300,
            "chunkMode": "automatic",
            "chunkStrategy": "auto",
            "resolvedChunkStrategy": "recursive",
        },
    )

    assert body["stats"] == {
        "chunkCount": 3,
        "tokenCount": 44,
        "charCount": 321,
        "meaningfulCharCount": 300,
        "chunkMode": "automatic",
        "chunkStrategy": "auto",
        "resolvedChunkStrategy": "recursive",
        "chunks": 3,
        "tokens": 44,
    }


def test_processing_callback_is_mapped_to_canonical_indexing():
    client = _client()
    with patch("services.tinnten_server_client.requests.patch") as request_patch:
        request_patch.return_value = MagicMock(status_code=200)
        assert client.update_document_index_state("D1", "processing") is True

    assert request_patch.call_args.kwargs["json"]["state"] == "indexing"


def test_domain_without_fetcher_source_is_ignored():
    # domain var ama source fetcher_page değil → metadata eklenmez (güvenli)
    body = _capture_patch_body(
        _client(), company_id="C1", domain="example.com", source="upload", domain_chunks=9
    )
    assert "metadata" not in body


# ── 0-chunk tamamlanma: callback domain HINT ile gitmeli ─────────────────────
# chunks=0 biten ingest'te kayıtlı chunk yok → chunk-tabanlı domain çözümü boş
# kalır ve callback metadata'sız giderdi (server website entry'sini bulamaz,
# abonelik embedding-state köprüsü hiç tetiklenmez). Hint (content-load
# metadata'sındaki domain) bu boşluğu kapatır.

def _worker_for_state_callback(chunk_docs):
    from workers.ingest_worker import IngestWorker

    w = IngestWorker.__new__(IngestWorker)
    store = MagicMock()
    store.get_chunks_by_doc.return_value = list(chunk_docs)
    store.get_document.return_value = None
    store.count_active_chunks_by_company_domain.return_value = len(chunk_docs)
    w.store = store
    w.content_store = MagicMock()
    # Bildirim artık arka plan thread'inden gidiyor (doküman yolunu bekletmemek
    # için); `__init__` atlandığından kuyruk/kilit burada kurulur.
    w._callback_queue = queue.Queue(maxsize=100)
    w._callback_thread = None
    w._callback_lock = threading.RLock()
    return w


def _run_safe_update(worker, **kwargs):
    from workers import ingest_worker as iw

    ctx = iw.DocumentJobContext(
        company_id="C1",
        document_id="D1",
        job_id="J1",
        user_id=None,
        trigger="fetcher_page",
        options={},
    )
    with patch.object(iw, "get_tinnten_server_client") as get_cli:
        worker._safe_update_index_state(
            ctx, state="completed", stats={"chunkCount": 0}, error=None,
            callback_domain=kwargs.get("callback_domain"),
            callback_source=kwargs.get("callback_source"),
        )
        # Bildirim asenkron: patch hâlâ aktifken teslimin bitmesini bekle.
        worker._callback_queue.join()
        return get_cli.return_value.update_document_index_state.call_args.kwargs


def test_zero_chunk_completed_callback_carries_domain_from_hint():
    w = _worker_for_state_callback(chunk_docs=[])
    kwargs = _run_safe_update(
        w, callback_domain="grntsoftware.com", callback_source="fetcher_page"
    )
    assert kwargs["domain"] == "grntsoftware.com"
    assert kwargs["source"] == "fetcher_page"
    assert kwargs["domain_chunks"] == 0  # 0-chunk'ta da aggregate sayım gönderilir
    assert kwargs["job_id"] == "J1"


def test_zero_chunk_without_hint_keeps_legacy_gap():
    # Hint yoksa eski davranış: chunk yok → domain çözülemez (regresyon değil).
    w = _worker_for_state_callback(chunk_docs=[])
    kwargs = _run_safe_update(w)
    assert kwargs["domain"] is None


def test_chunk_fallback_still_resolves_domain_without_hint():
    w = _worker_for_state_callback(
        chunk_docs=[{"metadata": {"domain": "x.com", "source": "fetcher_page"}}]
    )
    kwargs = _run_safe_update(w)
    assert kwargs["domain"] == "x.com"
    assert kwargs["source"] == "fetcher_page"


def test_callback_prefers_persisted_page_url_and_title_metadata():
    w = _worker_for_state_callback(chunk_docs=[])
    w.content_store.get_document.return_value = {
        "index": {"jobId": "J1"},
        "title": "Stored title",
        "metadata": {
            "domain": "x.com",
            "source": "fetcher_page",
            "url": "https://x.com/article",
            "title": "Exact article title",
        },
    }
    kwargs = _run_safe_update(w)
    assert kwargs["page_url"] == "https://x.com/article"
    assert kwargs["page_title"] == "Exact article title"


def test_rejected_job_cas_does_not_enqueue_terminal_callback():
    from workers import ingest_worker as iw

    w = _worker_for_state_callback(chunk_docs=[])
    w.content_store.update_index_fields.return_value = None
    ctx = iw.DocumentJobContext(
        company_id="C1",
        document_id="D1",
        job_id="job-stale",
        user_id=None,
        trigger="manual",
        options={},
    )

    with patch.object(iw, "get_tinnten_server_client") as get_cli:
        w._safe_update_index_state(
            ctx, state="completed", stats={"chunkCount": 1}, error=None
        )

    assert w._callback_queue.empty()
    get_cli.assert_not_called()


def test_index_state_backend_failure_returns_false_and_does_not_callback():
    from workers import ingest_worker as iw

    w = _worker_for_state_callback(chunk_docs=[])
    w.content_store.update_index_fields.side_effect = RuntimeError("mongo down")
    ctx = iw.DocumentJobContext(
        company_id="C1",
        document_id="D1",
        job_id="job-current",
        user_id=None,
        trigger="manual",
        options={},
    )

    with patch.object(iw, "get_tinnten_server_client") as get_cli:
        applied = w._safe_update_index_state(
            ctx, state="completed", stats={"chunkCount": 1}, error=None
        )

    assert applied is False
    assert w._callback_queue.empty()
    get_cli.assert_not_called()


def test_queued_callback_is_discarded_when_a_newer_job_is_current():
    from workers import ingest_worker as iw

    w = _worker_for_state_callback(chunk_docs=[])
    w.content_store.get_document.return_value = {"index": {"jobId": "job-new"}}
    ctx = iw.DocumentJobContext(
        company_id="C1",
        document_id="D1",
        job_id="job-old",
        user_id=None,
        trigger="manual",
        options={},
    )

    with patch.object(iw, "get_tinnten_server_client") as get_cli:
        w._deliver_index_state_callback(
            context=ctx,
            state="completed",
            stats={"chunkCount": 1},
            error_msg=None,
            callback_domain=None,
            callback_source=None,
        )

    get_cli.assert_not_called()


def test_callback_soft_failure_is_retried(mocker):
    from workers import ingest_worker as iw

    w = _worker_for_state_callback(chunk_docs=[])
    w.content_store.get_document.return_value = {"index": {"jobId": "J1"}}
    ctx = iw.DocumentJobContext(
        company_id="C1",
        document_id="D1",
        job_id="J1",
        user_id=None,
        trigger="manual",
        options={},
    )

    with patch.object(iw, "get_tinnten_server_client") as get_cli, patch.dict(
        "os.environ",
        {"INDEX_STATE_CALLBACK_ATTEMPTS": "3", "INDEX_STATE_CALLBACK_RETRY_SECONDS": "0"},
    ):
        get_cli.return_value.update_document_index_state.side_effect = [False, False, True]
        w._deliver_index_state_callback(
            context=ctx,
            state="completed",
            stats={"chunkCount": 1},
            error_msg=None,
            callback_domain=None,
            callback_source=None,
        )

    assert get_cli.return_value.update_document_index_state.call_count == 3


def _durable_content_store():
    from services.content_store import ContentDocumentStore

    store = ContentDocumentStore.__new__(ContentDocumentStore)
    store.db = mongomock.MongoClient()["tinnten"]
    store.documents = store.db["contentdocuments"]
    store.logs = store.db["contentdocumentlogs"]
    return store


def _pending_record(event_id="event-1"):
    now = datetime.now(timezone.utc)
    return {
        "eventId": event_id,
        "companyId": "C1",
        "documentId": "D1",
        "jobId": "J1",
        "userId": None,
        "trigger": "manual",
        "options": {},
        "attempt": 1,
        "state": "completed",
        "stats": {"chunkCount": 1},
        "errorMsg": None,
        "callbackDomain": None,
        "callbackSource": None,
        "createdAt": now,
        "nextAttemptAt": now,
        "attempts": 0,
        "leaseOwner": None,
        "leaseUntil": None,
    }


def test_terminal_state_and_callback_intent_are_written_atomically():
    from workers import ingest_worker as iw

    worker = _worker_for_state_callback(chunk_docs=[])
    worker.callback_outbox_enabled = True
    worker._submit_index_state_callback = MagicMock()
    context = iw.DocumentJobContext(
        company_id="C1",
        document_id="D1",
        job_id="J1",
        user_id=None,
        trigger="manual",
        options={},
        attempt=4,
    )

    assert worker._safe_update_index_state(
        context,
        state="quota_blocked",
        stats={"chunkCount": 0},
        error="index quota exceeded",
    ) is True

    update = worker.content_store.update_index_fields.call_args.kwargs
    pending = update["extra_updates"]["index.callbackPending"]
    assert pending["state"] == "quota_blocked"
    assert pending["jobId"] == "J1"
    assert pending["attempt"] == 4
    assert pending["errorMsg"] == "index quota exceeded"
    worker._submit_index_state_callback.assert_called_once()
    assert worker._submit_index_state_callback.call_args.kwargs["_event_id"] == pending["eventId"]


def test_callback_outbox_claim_retry_and_completion_are_lease_safe():
    store = _durable_content_store()
    store.documents.insert_one(
        {
            "companyId": "C1",
            "documentId": "D1",
            "index": {"jobId": "J1", "callbackPending": _pending_record()},
        }
    )
    at = datetime.now(timezone.utc) + timedelta(seconds=1)

    claimed = store.claim_index_callback(
        company_id="C1",
        document_id="D1",
        event_id="event-1",
        lease_owner="worker-a",
        lease_seconds=30,
        now=at,
    )
    assert claimed["eventId"] == "event-1"
    assert store.claim_index_callback(
        company_id="C1",
        document_id="D1",
        event_id="event-1",
        lease_owner="worker-b",
        lease_seconds=30,
        now=at,
    ) is None

    retry_at = at + timedelta(seconds=10)
    assert store.retry_index_callback(
        company_id="C1",
        document_id="D1",
        event_id="event-1",
        lease_owner="worker-a",
        next_attempt_at=retry_at,
        error="server unavailable",
    ) is True
    assert store.claim_index_callback(
        lease_owner="worker-b", lease_seconds=30, now=retry_at - timedelta(seconds=1)
    ) is None
    claimed_again = store.claim_index_callback(
        lease_owner="worker-b", lease_seconds=30, now=retry_at
    )
    assert claimed_again["attempts"] == 1
    assert store.complete_index_callback(
        company_id="C1",
        document_id="D1",
        event_id="event-1",
        lease_owner="worker-b",
    ) is True
    stored = store.documents.find_one({"companyId": "C1", "documentId": "D1"})
    assert "callbackPending" not in stored["index"]


def test_startup_replay_delivers_crash_left_callback():
    from workers.ingest_worker import IngestWorker

    store = _durable_content_store()
    store.documents.insert_one(
        {
            "companyId": "C1",
            "documentId": "D1",
            "index": {"jobId": "J1", "callbackPending": _pending_record()},
        }
    )
    worker = IngestWorker.__new__(IngestWorker)
    worker.content_store = store
    worker._callback_lease_owner = "replay-worker"
    worker._deliver_index_state_callback = MagicMock(return_value=True)

    worker._replay_one_index_state_callback()

    worker._deliver_index_state_callback.assert_called_once()
    stored = store.documents.find_one({"companyId": "C1", "documentId": "D1"})
    assert "callbackPending" not in stored["index"]
