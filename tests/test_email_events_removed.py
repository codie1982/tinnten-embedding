"""
İndexleme e-postaları KALDIRILDI — EmbeddingEmailEvents kalıcı no-op.

Doğrulama: EMBEDDING_EMAIL_EVENTS_ENABLED açık OLSA BİLE hiçbir send_* metodu
yayın yapmaz (publisher=None, _enabled=False), False döner, crash olmaz.
"""
import os


def test_embedding_email_events_hard_disabled(monkeypatch):
    # Bayrağı AÇIK ayarla — hard-disable ondan bağımsız olmalı.
    monkeypatch.setenv("EMBEDDING_EMAIL_EVENTS_ENABLED", "1")
    from services.email_queue_events import EmbeddingEmailEvents

    ev = EmbeddingEmailEvents()
    # Env açık olsa da kalıcı kapalı + hiçbir bağlantı nesnesi kurulmadı.
    assert ev._enabled is False
    assert ev.publisher is None
    assert ev.server_client is None

    # Tüm indexleme send_* metodları no-op (False, yan-etkisiz).
    assert ev.send_index_started(
        company_id="c", document_id="d", job_id="j", source="fetcher_page", trigger="t"
    ) is False
    assert ev.send_index_completed(
        company_id="c", document_id="d", job_id="j", source="fetcher_page",
        stats={"chunkCount": 5}, finished_at=None
    ) is False
    assert ev.send_index_failed(
        company_id="c", document_id="d", job_id="j", source="fetcher_page",
        stage="chunking", reason="boom"
    ) is False
