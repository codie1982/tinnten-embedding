"""
FAZ 2c — index-state callback body sözleşmesi.
Per-sayfa (fetcher_page/initial) doc'larda callback body'si `metadata.domain` +
`stats.domainChunks` taşımalı ki server website ENTRY'sini domain ile bulup
badge'i güncellesin. Diğer source'larda bu alanlar OLMAMALI.
"""
from unittest.mock import MagicMock, patch

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
        domain_chunks=42,
    )
    assert body["state"] == "indexed"  # completed → indexed
    assert body["metadata"] == {"domain": "example.com", "source": "fetcher_page"}
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
    store.chunks.count_documents.return_value = len(chunk_docs)
    w.store = store
    w.content_store = MagicMock()
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
        return get_cli.return_value.update_document_index_state.call_args.kwargs


def test_zero_chunk_completed_callback_carries_domain_from_hint():
    w = _worker_for_state_callback(chunk_docs=[])
    kwargs = _run_safe_update(
        w, callback_domain="grntsoftware.com", callback_source="fetcher_page"
    )
    assert kwargs["domain"] == "grntsoftware.com"
    assert kwargs["source"] == "fetcher_page"
    assert kwargs["domain_chunks"] == 0  # 0-chunk'ta da aggregate sayım gönderilir


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
