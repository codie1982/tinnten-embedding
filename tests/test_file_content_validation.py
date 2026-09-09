"""Parser-backed file ingest must not report zero-content files as indexed."""
import io
from unittest.mock import MagicMock

import numpy as np
import pytest

from services.content_validation import (
    NonIndexableFileContentError,
    validate_file_chunk_result,
    validate_file_extraction,
)
from services.document_loader import DocumentContent, DocumentLoader, DocumentParseError
from workers.ingest_worker import DocumentJobContext, IngestWorker


def test_empty_pdf_requires_ocr_before_embedding():
    with pytest.raises(NonIndexableFileContentError) as caught:
        validate_file_extraction(
            source="upload",
            text=" \n\t ",
            metadata={"filename": "scan.pdf", "contentType": "application/pdf"},
            min_chars=80,
            ocr_enabled=False,
        )

    assert caught.value.reason == "ocr_required"
    assert caught.value.metrics.extracted_chars == 4


def test_requesting_ocr_without_an_ocr_result_still_requires_ocr():
    with pytest.raises(NonIndexableFileContentError) as caught:
        validate_file_extraction(
            source="upload",
            text="",
            metadata={"filename": "empty.pdf"},
            min_chars=80,
            ocr_enabled=True,
        )

    assert caught.value.reason == "ocr_required"


def test_empty_pdf_after_confirmed_ocr_is_no_extractable_text():
    with pytest.raises(NonIndexableFileContentError) as caught:
        validate_file_extraction(
            source="upload",
            text="",
            metadata={"filename": "empty.pdf", "parseResult": {"ocrApplied": True}},
            min_chars=80,
            ocr_enabled=True,
        )

    assert caught.value.reason == "no_extractable_text"


@pytest.mark.parametrize(
    ("filename", "text"),
    [
        ("empty.docx", ""),
        ("empty.xlsx", "[sheet] Sheet1"),
        ("empty.pptx", "[slide] 1\n[slide] 2"),
    ],
)
def test_empty_office_files_have_no_extractable_text(filename, text):
    with pytest.raises(NonIndexableFileContentError) as caught:
        validate_file_extraction(
            source="upload",
            text=text,
            metadata={"filename": filename},
            min_chars=80,
            ocr_enabled=False,
        )

    assert caught.value.reason == "no_extractable_text"
    assert caught.value.metrics.meaningful_chars == 0


def test_short_real_file_text_is_valid_even_below_chunk_tail_limit():
    metrics = validate_file_extraction(
        source="upload",
        text="Kısa ama gerçek içerik.",
        metadata={"filename": "note.txt"},
        min_chars=80,
        ocr_enabled=False,
    )

    assert 0 < metrics.meaningful_chars < metrics.min_chars


@pytest.mark.parametrize("text", ["---", "... !!!", "\u200b\u200d\ufeff"])
def test_punctuation_and_zero_width_only_are_not_extractable(text):
    with pytest.raises(NonIndexableFileContentError, match="no_extractable_text"):
        validate_file_extraction(
            source="upload",
            text=text,
            metadata={"filename": "empty.txt"},
            min_chars=80,
            ocr_enabled=False,
        )


def test_unicode_letters_and_numbers_are_meaningful():
    metrics = validate_file_extraction(
        source="upload",
        text="Türkçe 中文 １２３",
        metadata={"filename": "unicode.txt"},
        min_chars=80,
        ocr_enabled=False,
    )

    assert metrics.meaningful_chars > 0


def test_scaffold_like_text_is_only_stripped_for_structural_formats():
    text_metrics = validate_file_extraction(
        source="upload",
        text="[slide] gerçek başlık",
        metadata={"filename": "note.txt"},
        min_chars=80,
        ocr_enabled=False,
    )
    assert text_metrics.meaningful_chars > 0

    with pytest.raises(NonIndexableFileContentError, match="no_extractable_text"):
        validate_file_extraction(
            source="upload",
            text="[slide] gerçek başlık",
            metadata={"detectedExt": "pptx", "filename": "opaque.bin"},
            min_chars=80,
            ocr_enabled=False,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"detectedExt": "pdf", "filename": "/api/v10/files/raw"},
        {"storageKey": "private/uploads/scanned.pdf?version=2"},
        {"indexPreflight": {"parser": "pdf"}, "filename": "opaque"},
    ],
)
def test_pdf_detection_uses_authoritative_and_legacy_metadata(metadata):
    with pytest.raises(NonIndexableFileContentError, match="ocr_required"):
        validate_file_extraction(
            source="upload",
            text="",
            metadata=metadata,
            min_chars=80,
            ocr_enabled=False,
        )


def test_empty_fetcher_content_is_outside_file_validation_policy():
    metrics = validate_file_extraction(
        source="fetcher_crawl",
        text="",
        metadata={"domain": "example.com"},
        min_chars=80,
        ocr_enabled=False,
    )

    assert metrics.meaningful_chars == 0


def test_empty_local_file_is_blocked_like_an_upload():
    with pytest.raises(NonIndexableFileContentError) as caught:
        validate_file_extraction(
            source="local_file",
            text="\n\t",
            metadata={"filename": "empty.txt"},
            min_chars=80,
            ocr_enabled=False,
        )

    assert caught.value.reason == "no_extractable_text"


def test_nonempty_file_with_defensive_zero_chunk_result_is_insufficient_text():
    metrics = validate_file_extraction(
        source="upload",
        text="gerçek içerik",
        metadata={"filename": "note.txt"},
        min_chars=80,
        ocr_enabled=False,
    )

    with pytest.raises(NonIndexableFileContentError) as caught:
        validate_file_chunk_result(
            source="upload",
            chunk_count=0,
            metrics=metrics,
            metadata={"filename": "note.txt"},
        )

    assert caught.value.reason == "insufficient_text"


def _chunk_worker():
    worker = IngestWorker.__new__(IngestWorker)
    worker.chunk_size = 900
    worker.chunk_overlap = 120
    worker.batch_size = 8
    engine = MagicMock()
    engine.model_name = "test-model"
    engine.encode.return_value = np.ones((1, 4), dtype=np.float32)
    worker._engine_for_company = MagicMock(return_value=engine)
    worker._swap_in_chunks = MagicMock(return_value="version-1")
    return worker, engine


def test_short_upload_uses_one_chunk_fallback():
    worker, engine = _chunk_worker()

    stats = worker._chunk_and_embed(
        doc_id="D1",
        company_id="C1",
        doc_type="document",
        source="upload",
        metadata={"filename": "short.txt"},
        text="Kısa gerçek metin",
        options={"chunkSize": 900, "chunkOverlap": 120, "minChars": 80},
    )

    assert stats["chunkCount"] == 1
    assert stats["minChars"] == 80
    assert stats["effectiveMinChars"] == 1
    assert engine.encode.call_args.args[0] == ["Kısa gerçek metin"]


def test_short_web_text_keeps_existing_zero_chunk_behaviour():
    worker, engine = _chunk_worker()

    stats = worker._chunk_and_embed(
        doc_id="D1",
        company_id="C1",
        doc_type="web",
        source="text",
        metadata={"domain": "example.com"},
        text="short page",
        options={"chunkSize": 900, "chunkOverlap": 120, "minChars": 80},
    )

    assert stats["chunkCount"] == 0
    engine.encode.assert_not_called()


def _processing_worker(*, source, text, metadata):
    worker = IngestWorker.__new__(IngestWorker)
    worker.chunk_size = 900
    worker.chunk_overlap = 120
    worker.email_events = MagicMock()
    worker._resolve_document_source = MagicMock(
        return_value={"source": source, "doc_type": "document", "metadata": metadata}
    )
    worker._load_document_content = MagicMock(return_value=(text, metadata))
    worker._safe_update_index_state = MagicMock()
    worker._log_document_event = MagicMock()
    worker._notify_index_failure = MagicMock()
    worker._log_worker_error = MagicMock()
    worker._mark_file_source_index_failed = MagicMock()
    worker._mark_file_source_index_completed = MagicMock()
    embedding_store = MagicMock()
    worker._get_store = MagicMock(return_value=embedding_store)
    worker._chunk_and_embed = MagicMock(
        return_value={"chunkCount": 0, "tokenCount": 0, "charCount": 0}
    )
    worker._embedding_store = embedding_store
    return worker


def _context(trigger="manual"):
    return DocumentJobContext(
        company_id="C1",
        document_id="D1",
        job_id="J1",
        user_id="U1",
        trigger=trigger,
        options={
            "chunkSize": 900,
            "chunkOverlap": 120,
            "minChars": 80,
            "cleanup": True,
            "ocr": False,
            "langDetect": False,
        },
    )


def test_empty_upload_finishes_failed_and_never_calls_chunker():
    worker = _processing_worker(
        source="upload",
        text="",
        metadata={"filename": "scan.pdf", "contentType": "application/pdf"},
    )

    with pytest.raises(NonIndexableFileContentError, match="ocr_required"):
        worker._process_single_document_locked({}, _context())

    worker._chunk_and_embed.assert_not_called()
    failure_call = next(
        call
        for call in worker._safe_update_index_state.call_args_list
        if call.kwargs.get("state") == "failed"
    )
    assert failure_call.kwargs["error"] == "ocr_required"
    assert failure_call.kwargs["stats"]["failureReason"] == "ocr_required"
    assert failure_call.kwargs["stats"]["extractedCharCount"] == 0


def test_empty_fetcher_result_can_keep_completed_callback_state():
    worker = _processing_worker(
        source="fetcher_crawl",
        text="",
        metadata={"domain": "example.com", "source": "fetcher_page"},
    )

    worker._process_single_document_locked({}, _context(trigger="fetcher_page"))

    states = [call.kwargs.get("state") for call in worker._safe_update_index_state.call_args_list]
    assert "failed" not in states
    assert states[-1] == "completed"


def test_processing_transition_uses_server_canonical_indexing_state():
    worker = _processing_worker(
        source="upload",
        text="Gerçek dosya içeriği",
        metadata={"filename": "note.txt"},
    )
    worker._chunk_and_embed.return_value = {
        "chunkCount": 1,
        "tokenCount": 4,
        "charCount": 24,
    }

    worker._process_single_document_locked({}, _context())

    assert worker._safe_update_index_state.call_args_list[0].kwargs["state"] == "indexing"


def test_worker_requeues_when_initial_indexing_state_cannot_be_persisted():
    from workers.ingest_worker import RetryableIngestLockError

    worker = _processing_worker(
        source="upload",
        text="Gerçek dosya içeriği",
        metadata={"filename": "note.txt"},
    )
    worker._safe_update_index_state.return_value = False

    with pytest.raises(RetryableIngestLockError, match="indexing state ownership lost"):
        worker._process_single_document_locked({}, _context())

    worker._load_document_content.assert_not_called()


def test_validation_failure_is_requeued_when_terminal_state_write_is_uncertain():
    from workers.ingest_worker import RetryableIngestLockError

    worker = _processing_worker(
        source="upload",
        text="",
        metadata={"filename": "scan.pdf", "contentType": "application/pdf"},
    )
    worker._safe_update_index_state.side_effect = [True, False]

    with pytest.raises(RetryableIngestLockError, match="validation failure state"):
        worker._process_single_document_locked({}, _context())

    worker._mark_file_source_index_failed.assert_not_called()


def test_upload_is_completed_only_after_embedding_terminal_state():
    worker = _processing_worker(
        source="upload",
        text="Kısa ama gerçek dosya içeriği",
        metadata={"filename": "note.txt"},
    )
    worker._resolve_document_source.return_value["upload_id"] = "UP1"
    worker._chunk_and_embed.return_value = {
        "chunkCount": 1,
        "tokenCount": 5,
        "charCount": 29,
    }

    context = _context(trigger="upload_scan_clean")
    worker._process_single_document_locked({}, context)

    worker._mark_file_source_index_completed.assert_called_once_with(
        resolved_source=worker._resolve_document_source.return_value,
        document_id="D1",
        context=context,
    )
    ready_call = worker._embedding_store.update_document_status.call_args_list[-1]
    assert ready_call.kwargs["status"] == "ready"
    assert ready_call.kwargs["expected_job_id"] == "J1"


def test_downstream_embedding_failure_marks_upload_failed():
    worker = _processing_worker(
        source="upload",
        text="Gerçek dosya içeriği",
        metadata={"filename": "note.txt"},
    )
    worker._resolve_document_source.return_value["upload_id"] = "UP1"
    worker._chunk_and_embed.side_effect = RuntimeError("faiss unavailable")

    with pytest.raises(RuntimeError, match="faiss unavailable"):
        worker._process_single_document_locked(
            {}, _context(trigger="upload_scan_clean")
        )

    failed_call = worker._mark_file_source_index_failed.call_args
    assert failed_call.kwargs["document_id"] == "D1"
    assert "faiss unavailable" in failed_call.kwargs["reason"]


def test_upload_parse_success_remains_in_progress_until_chunk_commit(mocker):
    worker = IngestWorker.__new__(IngestWorker)
    upload_store = MagicMock()
    upload_store.get_upload_by_id.return_value = {"file": {"key": "opaque-key"}}
    upload_store.get_file_by_upload_id.return_value = {
        "key": "private/uuid-object",
        "originalname": "scan.pdf",
    }
    loader = MagicMock()
    loader.fetch_text.return_value = DocumentContent(
        bucket="bucket",
        key="private/uuid-object",
        filename="uuid-object",
        content_type="application/pdf",
        text="PDF metni",
    )
    worker._get_upload_store = MagicMock(return_value=upload_store)
    worker._get_loader = MagicMock(return_value=loader)
    worker._resolve_s3_location = MagicMock(
        return_value=("bucket", "private/uuid-object", "scan.pdf")
    )
    mocker.patch.dict(
        "os.environ",
        {"UPLOAD_DOWNLOAD_RETRIES": "1", "UPLOAD_DOWNLOAD_RETRY_DELAY_SECONDS": "0"},
    )

    text, metadata = worker._load_upload_text(
        "UP1", {}, _context(trigger="upload_scan_clean")
    )

    assert text == "PDF metni"
    assert metadata["filename"] == "scan.pdf"
    assert metadata["extension"] == "pdf"
    assert metadata["storageKey"] == "private/uuid-object"
    statuses = [call.kwargs["index_status"] for call in upload_store.update_upload_status.call_args_list]
    assert statuses == ["in_progress", "in_progress"]


def test_upload_location_combines_opaque_key_with_files_original_name(mocker):
    worker = IngestWorker.__new__(IngestWorker)
    mocker.patch.dict("os.environ", {"AWS_S3_BUCKET": "uploads"})

    bucket, key, filename = worker._resolve_s3_location(
        {"file": {"key": "private/6a9fa02330716aa6c8db85e3"}},
        {
            "key": "private/6a9fa02330716aa6c8db85e3",
            "originalname": "tinten-light-theme-platform-analizi.pdf",
        },
    )

    assert (bucket, key) == ("uploads", "private/6a9fa02330716aa6c8db85e3")
    assert filename == "tinten-light-theme-platform-analizi.pdf"


def test_document_loader_uses_original_filename_hint_for_opaque_object_key():
    loader = DocumentLoader.__new__(DocumentLoader)
    loader.bucket = "uploads"
    loader.client = MagicMock()
    loader.client.get_object.return_value = {
        "Body": io.BytesIO(b"pdf bytes"),
        "ContentType": "application/octet-stream",
    }
    loader._extract_text = MagicMock(return_value="parsed")

    document = loader.fetch_text(
        "private/6a9fa02330716aa6c8db85e3",
        filename_hint="report.pdf",
    )

    loader._extract_text.assert_called_once_with(
        b"pdf bytes", "report.pdf", "application/octet-stream"
    )
    assert document.filename == "report.pdf"


def test_legacy_doc_is_rejected_without_a_guaranteed_parser():
    loader = DocumentLoader.__new__(DocumentLoader)

    with pytest.raises(DocumentParseError, match="Unsupported file extension: .doc"):
        loader._extract_text(b"legacy word payload", "report.doc", "application/msword")


def test_manual_upload_reindex_does_not_mutate_shared_upload_lifecycle(mocker):
    worker = IngestWorker.__new__(IngestWorker)
    upload_store = MagicMock()
    upload_store.get_upload_by_id.return_value = {"file": {"key": "opaque-key"}}
    upload_store.get_file_by_upload_id.return_value = {
        "key": "private/uuid-object",
        "originalname": "report.pdf",
    }
    loader = MagicMock()
    loader.fetch_text.return_value = DocumentContent(
        bucket="bucket",
        key="private/uuid-object",
        filename="report.pdf",
        content_type="application/pdf",
        text="Aranabilir PDF metni",
    )
    worker._get_upload_store = MagicMock(return_value=upload_store)
    worker._get_loader = MagicMock(return_value=loader)
    worker._resolve_s3_location = MagicMock(
        return_value=("bucket", "private/uuid-object", "report.pdf")
    )
    mocker.patch.dict(
        "os.environ",
        {"UPLOAD_DOWNLOAD_RETRIES": "1", "UPLOAD_DOWNLOAD_RETRY_DELAY_SECONDS": "0"},
    )

    worker._load_upload_text("UP1", {}, _context(trigger="manual_update"))
    worker._mark_file_source_index_completed(
        resolved_source={"source": "upload", "upload_id": "UP1"},
        document_id="D1",
        context=_context(trigger="manual_update"),
    )
    worker._mark_file_source_index_failed(
        resolved_source={"source": "upload", "upload_id": "UP1"},
        document_id="D1",
        reason="manual reindex failed",
        context=_context(trigger="manual_update"),
    )

    upload_store.update_upload_status.assert_not_called()


def _embedded_chunks_worker():
    worker = IngestWorker.__new__(IngestWorker)
    worker.chunk_size = 900
    worker.chunk_overlap = 120
    worker.index_path = "/tmp/test.index"
    store = MagicMock()
    worker._get_store = MagicMock(return_value=store)
    worker._get_existing_ready_docs = MagicMock(return_value={})
    worker._get_upload_store = MagicMock()
    worker._log_worker_error = MagicMock()
    worker.email_events = MagicMock()
    return worker, store


def test_empty_preembedded_upload_is_failed_not_ready():
    worker, store = _embedded_chunks_worker()

    with pytest.raises(NonIndexableFileContentError, match="ocr_required"):
        worker._process_embedded_chunks(
            {
                "payload_type": "embedded_chunks",
                "doc_id": "D1",
                "doc_type": "document",
                "source": "upload",
                "metadata": {"filename": "scan.pdf"},
                "chunks": [],
                "embeddings": [],
            }
        )

    terminal = store.update_document_status.call_args_list[-1]
    assert terminal.kwargs == {
        "status": "failed",
        "chunk_count": 0,
        "error": "ocr_required",
    }


def test_empty_preembedded_web_payload_keeps_ready_zero_behaviour():
    worker, store = _embedded_chunks_worker()

    worker._process_embedded_chunks(
        {
            "payload_type": "embedded_chunks",
            "doc_id": "D1",
            "doc_type": "web",
            "source": "web",
            "chunks": [],
            "embeddings": [],
        }
    )

    terminal = store.update_document_status.call_args_list[-1]
    assert terminal.kwargs == {"status": "ready", "chunk_count": 0}


def test_blank_preembedded_upload_chunk_is_rejected_before_vector_swap():
    worker, store = _embedded_chunks_worker()
    worker._swap_in_chunks = MagicMock()

    with pytest.raises(NonIndexableFileContentError, match="no_extractable_text"):
        worker._process_embedded_chunks(
            {
                "payload_type": "embedded_chunks",
                "doc_id": "D1",
                "doc_type": "document",
                "source": "upload",
                "metadata": {"filename": "blank.txt"},
                "chunks": [{"text": "  ---  "}],
                "embeddings": [[0.1, 0.2]],
            }
        )

    worker._swap_in_chunks.assert_not_called()
    assert store.update_document_status.call_args_list[-1].kwargs["status"] == "failed"


def test_processing_preembedded_document_is_not_treated_as_terminal():
    worker = IngestWorker.__new__(IngestWorker)
    store = MagicMock()
    store.get_documents_by_ids.return_value = {
        "D1": {"doc_id": "D1", "status": "processing"}
    }
    worker._get_store = MagicMock(return_value=store)

    assert worker._get_existing_ready_docs(["D1"]) == {}


def test_preembedded_downstream_failure_marks_upload_failed():
    worker, store = _embedded_chunks_worker()
    store.update_document_status.return_value = True

    worker._report_message_failure(
        {
            "payload_type": "embedded_chunks",
            "doc_id": "D1",
            "jobId": "J1",
            "source": "upload",
            "metadata": {"uploadId": "UP1", "filename": "report.pdf"},
        },
        RuntimeError("faiss unavailable"),
    )

    worker._get_upload_store.return_value.update_upload_status.assert_called_once_with(
        "UP1",
        index_status="failed",
        is_file_opened=True,
        file_open_error="faiss unavailable",
    )
    assert store.update_document_status.call_args.kwargs["expected_job_id"] == "J1"


def test_manual_preembedded_reindex_does_not_mutate_shared_upload_lifecycle():
    worker, store = _embedded_chunks_worker()
    store.update_document_status.return_value = True

    worker._report_message_failure(
        {
            "payload_type": "embedded_chunks",
            "doc_id": "D1",
            "jobId": "J2",
            "trigger": "manual_update",
            "source": "upload",
            "metadata": {"uploadId": "UP1", "filename": "report.pdf"},
        },
        RuntimeError("manual reindex failed"),
    )

    worker._get_upload_store.return_value.update_upload_status.assert_not_called()


@pytest.mark.parametrize("extension", ["xml", "yaml", "yml"])
def test_text_like_contract_formats_are_parsed(extension):
    loader = DocumentLoader.__new__(DocumentLoader)
    payload = "başlık: değer\nöğe: 1".encode("utf-8")

    assert loader._extract_text(payload, f"config.{extension}", None) == payload.decode("utf-8")
