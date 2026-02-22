"""
RabbitMQ worker that consumes ingest jobs, chunking and embedding them into FAISS + MongoDB.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import gzip
import io
from urllib.parse import unquote, urlparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pika
import requests
import uuid
import numpy as np

# Ensure project-root imports work when executed as `python workers/ingest_worker.py`.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

from init.rabbit_connection import connect_rabbit_with_retry
from services.chunker import chunk_text
from services.mongo_store import MongoStore
from services.content_store import ContentDocumentStore, normalize_index_options
from services.upload_store import UploadStore, UploadNotFoundError
from services.fetcher_store import FetcherStore
from init.aws import get_s3_client


DEFAULT_PRIMARY_QUEUE_NAME = "content_indexing_queue"
LEGACY_QUEUE_NAME = "embedding.ingest"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_CONTENT_DOC_LOOKUP_RETRIES = 3
DEFAULT_CONTENT_DOC_LOOKUP_DELAY_SECONDS = 2.0
DEFAULT_UPLOAD_DOWNLOAD_RETRIES = 5
DEFAULT_UPLOAD_DOWNLOAD_RETRY_DELAY_SECONDS = 10.0
DEFAULT_FETCHER_PAGE_LIMIT = 50
DEFAULT_FETCHER_MAX_TEXT_CHARS = 1_500_000
DEFAULT_FETCHER_MEDIA_PER_PAGE = 20
DEFAULT_FETCHER_MEDIA_TEXT_CHARS = 2_000


def log(msg: str, *args: object) -> None:
    if args:
        try:
            msg = msg % args
        except Exception:
            msg = " ".join([str(msg), *[str(a) for a in args]])
    print(f"[ingest-worker] {msg}", flush=True)


@dataclass(slots=True)
class DocumentJobContext:
    company_id: str
    document_id: str
    job_id: str
    user_id: Optional[str]
    trigger: Optional[str]
    options: Dict[str, Any]


UPDATE_UNSET = object()


class IngestWorker:
    def __init__(self) -> None:
        queue_env_primary = (os.getenv("CONTENT_INDEX_QUEUE_NAME") or "").strip()
        queue_env_legacy = (os.getenv("EMBED_QUEUE_NAME") or "").strip()
        queues = [DEFAULT_PRIMARY_QUEUE_NAME]
        for raw in (queue_env_primary, queue_env_legacy):
            if not raw:
                continue
            queues.extend([q.strip() for q in raw.split(",") if q.strip()])

        # Optional legacy queue support for backward compatibility.
        legacy_flag = (os.getenv("ENABLE_LEGACY_EMBED_QUEUE") or "").strip().lower()
        legacy_enabled = legacy_flag in {"1", "true", "yes"}
        if legacy_enabled and LEGACY_QUEUE_NAME not in queues:
            queues.append(LEGACY_QUEUE_NAME)

        # Ensure the list is not empty and deduplicated.
        if not queues:
            queues = [DEFAULT_PRIMARY_QUEUE_NAME]
        seen = set()
        self.queue_names: List[str] = []
        for name in queues:
            if name not in seen:
                seen.add(name)
                self.queue_names.append(name)
        self.primary_queue = self.queue_names[0]

        self.chunk_size = int(os.getenv("EMBED_CHUNK_SIZE") or DEFAULT_CHUNK_SIZE)
        self.chunk_overlap = int(os.getenv("EMBED_CHUNK_OVERLAP") or DEFAULT_CHUNK_OVERLAP)
        self.batch_size = int(os.getenv("EMBED_BATCH_SIZE") or DEFAULT_BATCH_SIZE)
        default_chunk_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        self.model_name = os.getenv("CHUNK_MODEL_NAME") or os.getenv("MODEL_NAME") or default_chunk_model
        self.index_path = os.getenv("CHUNK_INDEX_PATH") or os.getenv("FAISS_INDEX_PATH") or "faiss.index"

        # Lazily initialised dependencies to keep startup fast and avoid blocking
        # on network connections/model downloads before consuming messages.
        self.store: MongoStore | None = None
        self._store_lock = threading.RLock()
        self.content_store: ContentDocumentStore | None = None
        self._content_store_lock = threading.RLock()
        self.upload_store: UploadStore | None = None
        self._upload_store_lock = threading.RLock()
        self.fetcher_store: FetcherStore | None = None
        self._fetcher_store_lock = threading.RLock()
        self.loader: DocumentLoader | None = None
        self._loader_lock = threading.RLock()
        self.engine: EmbeddingEngine | None = None
        self._engine_lock = threading.RLock()
        self._engine_cache: Dict[str, EmbeddingEngine] = {}
        self.connection: pika.BlockingConnection | None = None
        self.channel: pika.adapters.blocking_connection.BlockingChannel | None = None
        self._stop_event = threading.Event()

    def _get_store(self) -> MongoStore:
        if self.store is not None:
            return self.store
        with self._store_lock:
            if self.store is None:
                log("Initialising MongoStore...")
                self.store = MongoStore()
            return self.store

    def _get_content_store(self) -> ContentDocumentStore:
        if self.content_store is not None:
            return self.content_store
        with self._content_store_lock:
            if self.content_store is None:
                log("Initialising ContentDocumentStore...")
                self.content_store = ContentDocumentStore()
            return self.content_store

    def _get_upload_store(self) -> UploadStore:
        if self.upload_store is not None:
            return self.upload_store
        with self._upload_store_lock:
            if self.upload_store is None:
                log("Initialising UploadStore...")
                self.upload_store = UploadStore()
            return self.upload_store

    def _get_fetcher_store(self) -> FetcherStore:
        if self.fetcher_store is not None:
            return self.fetcher_store
        with self._fetcher_store_lock:
            if self.fetcher_store is None:
                log("Initialising FetcherStore...")
                self.fetcher_store = FetcherStore()
            return self.fetcher_store

    def _get_loader(self) -> DocumentLoader:
        if self.loader is not None:
            return self.loader
        with self._loader_lock:
            if self.loader is None:
                log("Initialising DocumentLoader...")
                from services.document_loader import DocumentLoader  # local import for faster startup

                self.loader = DocumentLoader()
            return self.loader

    def _get_engine(self) -> EmbeddingEngine:
        """
        Lazily initialise the embedding engine so the worker can start consuming messages
        without waiting for model downloads at process start.
        """
        if self.engine is not None:
            return self.engine
        with self._engine_lock:
            if self.engine is None:
                log("Initialising embedding engine (model=%s index_path=%s)...", self.model_name, self.index_path)
                from services.embedding_engine import EmbeddingEngine  # local import for faster startup

                self.engine = EmbeddingEngine(
                    model_name=self.model_name,
                    index_path=self.index_path,
                    auto_reset_on_error=False,
                    auto_reset_on_dim_mismatch=False,
                )
            return self.engine

    def _get_engine_for_index(self, index_path: Optional[str]) -> EmbeddingEngine:
        if not index_path or index_path == self.index_path:
            return self._get_engine()
        with self._engine_lock:
            cached = self._engine_cache.get(index_path)
            if cached is not None:
                return cached
            log("Initialising embedding engine (model=%s index_path=%s)...", self.model_name, index_path)
            from services.embedding_engine import EmbeddingEngine  # local import for faster startup

            engine = EmbeddingEngine(
                model_name=self.model_name,
                index_path=index_path,
                auto_reset_on_error=False,
                auto_reset_on_dim_mismatch=False,
            )
            self._engine_cache[index_path] = engine
            return engine

    # ------------------------------------------------------------------
    def start(self) -> None:
        log("Starting ingest worker...")
        log(
            "Config: queues=%s rabbit=%s:%s vhost=%s index_path=%s model=%s",
            self.queue_names,
            os.getenv("RABBITMQ_HOST") or "rabbitmq",
            os.getenv("RABBITMQ_PORT") or "5672",
            os.getenv("RABBITMQ_VHOST") or "/",
            self.index_path,
            self.model_name,
        )
        self.connection = connect_rabbit_with_retry()
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=1)
        for queue_name in self.queue_names:
            self.channel.queue_declare(queue=queue_name, durable=True, auto_delete=False, exclusive=False)
            self.channel.basic_consume(queue=queue_name, on_message_callback=self._handle_message, auto_ack=False)
        log(f"Awaiting messages on queues: {', '.join(self.queue_names)}")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        log("Stopping worker...")
        self._stop_event.set()
        try:
            if self.channel and self.channel.is_open:
                self.channel.stop_consuming()
        except Exception:
            pass
        try:
            if self.connection and self.connection.is_open:
                self.connection.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _handle_message(
        self,
        channel: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes,
    ) -> None:
        delivery_tag = method.delivery_tag
        payload: Dict[str, Any] = {}
        try:
            payload = json.loads(body.decode("utf-8"))
            self._process_payload(payload)
            channel.basic_ack(delivery_tag=delivery_tag)
        except Exception as exc:  # noqa: BLE001
            log(f"Error processing message: {exc}")
            # attempt to mark document(s) as failed if possible
            try:
                if isinstance(payload, dict):
                    doc_id = payload.get("doc_id")
                    if doc_id:
                        self._get_store().update_document_status(doc_id, status="failed", error=str(exc))
                    else:
                        document_ids = self._coalesce(payload, "documentIds", "document_ids", "documentids")
                        company_id = self._coalesce(payload, "companyId", "company_id", "companyid")
                        if company_id and document_ids:
                            ids = document_ids if isinstance(document_ids, Sequence) else [document_ids]
                            for document_id in ids:
                                self._safe_update_index_state(
                                    DocumentJobContext(
                                        company_id=company_id,
                                        document_id=str(document_id),
                                        job_id=str(payload.get("jobId") or payload.get("job_id") or uuid.uuid4()),
                                        user_id=self._coalesce(payload, "userId", "userid", "user_id"),
                                        trigger=self._coalesce(payload, "trigger"),
                                        options={},
                                    ),
                                    state="failed",
                                    error=str(exc),
                                )
            except Exception:
                pass
            channel.basic_nack(delivery_tag=delivery_tag, requeue=False)

    # ------------------------------------------------------------------
    def _process_payload(self, payload: Dict[str, Any]) -> None:
        if self._is_embedded_chunks_message(payload):
            log("Received embedded-chunks payload")
            self._process_embedded_chunks(payload)
        elif self._is_content_index_message(payload):
            log(f"Received content-index message")
            self._process_content_index_message(payload)
        else:
            log("Received legacy ingest payload")
            self._process_legacy_payload(payload)

    def _is_embedded_chunks_message(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        return payload.get("payload_type") == "embedded_chunks"

    def _is_content_index_message(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        return any(key in payload for key in ("documentIds", "document_ids", "documentids"))

    def _get_existing_ready_docs(self, doc_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        if not doc_ids:
            return {}
        docs = self._get_store().get_documents_by_ids(doc_ids)
        ready_docs: Dict[str, Dict[str, Any]] = {}
        for doc_id, doc in docs.items():
            status = str(doc.get("status") or "").lower()
            if status in {"ready", "processing"}:
                ready_docs[doc_id] = doc
        return ready_docs

    def _process_content_index_message(self, payload: Dict[str, Any]) -> None:
        company_id = self._coalesce(payload, "companyId", "company_id", "companyid")
        if not company_id:
            raise ValueError("payload missing companyId")

        document_ids_raw = self._coalesce(payload, "documentIds", "document_ids", "documentids")
        if document_ids_raw is None:
            raise ValueError("payload missing documentIds")

        if isinstance(document_ids_raw, str):
            document_ids = [document_ids_raw]
        elif isinstance(document_ids_raw, Sequence):
            document_ids = [str(doc_id) for doc_id in document_ids_raw]
        else:
            raise TypeError("documentIds must be a list of identifiers")

        user_id = self._coalesce(payload, "userId", "user_id", "userid")
        job_id = self._coalesce(payload, "jobId", "job_id")
        trigger = self._coalesce(payload, "trigger") or "manual"
        options_override = payload.get("options") or {}

        log(f"Processing company_id={company_id} documents={document_ids} job_id={job_id}")

        lookup_retries = int(os.getenv("CONTENT_DOC_LOOKUP_RETRIES") or DEFAULT_CONTENT_DOC_LOOKUP_RETRIES)
        lookup_delay = float(os.getenv("CONTENT_DOC_LOOKUP_DELAY_SECONDS") or DEFAULT_CONTENT_DOC_LOOKUP_DELAY_SECONDS)
        attempts = max(1, lookup_retries)

        documents: Dict[str, Dict[str, Any]] = {}
        missing_docs: List[str] = []
        for attempt in range(1, attempts + 1):
            documents = self._get_content_store().get_documents(company_id, document_ids)
            missing_docs = [doc_id for doc_id in document_ids if doc_id not in documents]
            if not missing_docs:
                break
            if attempt < attempts and lookup_delay > 0:
                log(
                    f"Documents missing from contentdocuments (attempt {attempt}/{attempts}); "
                    f"retrying in {lookup_delay}s: {missing_docs}"
                )
                time.sleep(lookup_delay)

        if missing_docs:
            for doc_id in missing_docs:
                ctx = DocumentJobContext(
                    company_id=company_id,
                    document_id=doc_id,
                    job_id=job_id or str(uuid.uuid4()),
                    user_id=user_id,
                    trigger=trigger,
                    options={},
                )
                msg = f"Document {doc_id} not found in contentdocuments collection."
                log(msg)
                self._log_document_event(ctx, level="error", message=msg, state="failed")
                self._safe_update_index_state(ctx, state="failed", error="document not found")

        for doc_id, document in documents.items():
            doc_index = document.get("index") if isinstance(document, dict) else None
            doc_job_id = (
                self._coalesce(doc_index or {}, "jobId", "job_id")
                or job_id
                or str(uuid.uuid4())
            )
            combined_options = {}
            if isinstance(doc_index, dict) and isinstance(doc_index.get("options"), dict):
                combined_options.update(doc_index["options"])
            combined_options.update(options_override)
            normalised_options = normalize_index_options(
                combined_options,
                default_chunk_size=self.chunk_size,
                default_chunk_overlap=self.chunk_overlap,
            )
            context = DocumentJobContext(
                company_id=company_id,
                document_id=doc_id,
                job_id=str(doc_job_id),
                user_id=user_id,
                trigger=trigger,
                options=normalised_options,
            )
            try:
                self._process_single_document(document, context)
            except Exception as exc:  # noqa: BLE001
                log(f"Failed to process document {doc_id}: {exc}")

    def _process_embedded_chunks(self, payload: Dict[str, Any]) -> None:
        doc_id = payload.get("doc_id")
        if not doc_id:
            raise ValueError("payload missing 'doc_id'")

        existing_docs = self._get_existing_ready_docs([doc_id])
        if doc_id in existing_docs:
            status = str(existing_docs[doc_id].get("status") or "").lower()
            log(f"Skipping embedded chunks doc_id={doc_id} because status={status}")
            return

        chunks = payload.get("chunks") or []
        embeddings = payload.get("embeddings") or []
        if not isinstance(chunks, list) or not isinstance(embeddings, list):
            raise TypeError("payload 'chunks' and 'embeddings' must be lists")

        doc_type = (payload.get("doc_type") or "web").lower()
        source = payload.get("source") or "web"
        metadata = payload.get("metadata") or {}
        options = normalize_index_options(
            payload.get("options") or {},
            default_chunk_size=self.chunk_size,
            default_chunk_overlap=self.chunk_overlap,
            default_min_chars=80,
        )
        index_path = payload.get("index_path") or self.index_path

        store = self._get_store()

        # Idempotency handled later via replace_embeddings

        store.create_document(doc_id=doc_id, doc_type=doc_type, source=source, metadata=metadata)
        store.update_document_status(doc_id, status="processing", error=None)

        if not chunks:
            store.update_document_status(doc_id, status="ready", chunk_count=0)
            return

        embeddings_arr = np.asarray(embeddings, dtype=np.float32)
        if embeddings_arr.ndim == 1:
            embeddings_arr = embeddings_arr.reshape(1, -1)
        if embeddings_arr.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if embeddings_arr.shape[0] != len(chunks):
            raise ValueError("embeddings length must match chunks length")

        store = self._get_store()
        faiss_ids = store.reserve_faiss_ids(len(chunks))
        
        # Idempotency: check for existing chunks
        existing_chunks = store.get_chunks_by_doc(doc_id)
        old_faiss_ids = []
        if existing_chunks:
            old_faiss_ids = [c["faiss_id"] for c in existing_chunks if "faiss_id" in c]
            deleted_count = store.delete_chunks_by_doc(doc_id)
            log(f"Removed {deleted_count} old chunks for doc_id={doc_id}")

        engine = self._get_engine_for_index(index_path)
        
        if old_faiss_ids:
            # Optimize: use replace_embeddings to save to disk only once
            engine.replace_embeddings(old_faiss_ids, embeddings_arr, faiss_ids)
            log(f"Replaced {len(old_faiss_ids)} old vectors with {len(faiss_ids)} new vectors for doc_id={doc_id}")
        else:
            engine.add_embeddings(embeddings_arr, faiss_ids)

        now = datetime.now(timezone.utc)
        chunk_docs: List[Dict[str, Any]] = []
        for idx, (chunk, faiss_id) in enumerate(zip(chunks, faiss_ids)):
            chunk_text = chunk.get("text") or ""
            chunk_index = chunk.get("chunk_index")
            if chunk_index is None:
                chunk_index = idx
            chunk_metadata = dict(metadata)
            if isinstance(chunk.get("metadata"), dict):
                chunk_metadata.update(chunk["metadata"])
            chunk_metadata.setdefault("chunk_index", chunk_index)
            chunk_metadata.setdefault("source", source)
            chunk_docs.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "company_id": None,
                    "chunk_index": int(chunk_index),
                    "faiss_id": int(faiss_id),
                    "text": chunk_text,
                    "char_start": chunk.get("char_start"),
                    "char_end": chunk.get("char_end"),
                    "doc_type": doc_type,
                    "source": source,
                    "metadata": chunk_metadata,
                    "options": dict(options),
                    "created_at": now,
                }
            )

        store.insert_chunks(chunk_docs)
        store.update_document_status(doc_id, status="ready", chunk_count=len(chunk_docs))
        log(f"Processed embedded chunks doc_id={doc_id} chunks={len(chunk_docs)}")

    # ------------------------------------------------------------------
    def _process_single_document(self, document: Dict[str, Any], context: DocumentJobContext) -> None:
        started_at = datetime.now(timezone.utc)
        log(f"Begin document doc_id={context.document_id} company_id={context.company_id} job_id={context.job_id}")
        options = context.options
        base_stats = self._initial_stats(options)
        self._safe_update_index_state(
            context,
            state="processing",
            stats=base_stats,
            error=None,
            extra={"index.startedAt": started_at},
        )
        self._log_document_event(
            context,
            level="info",
            message="Indexing job started.",
            state="processing",
            details={"options": options},
        )

        resolved_source = self._resolve_document_source(document, options)
        try:
            text, metadata = self._load_document_content(document, resolved_source, context)
        except Exception as exc:  # noqa: BLE001
            finished_at = datetime.now(timezone.utc)
            error_msg = f"Failed to load document content: {exc}"
            self._safe_update_index_state(
                context,
                state="failed",
                stats=base_stats,
                error=error_msg,
                extra={"index.finishedAt": finished_at},
            )
            self._log_document_event(context, level="error", message=error_msg, state="failed")
            raise

        self._log_document_event(
            context,
            level="info",
            message="Document content loaded.",
            state="processing",
            details={"source": resolved_source.get("source"), "metadataKeys": list(metadata.keys())},
        )

        if options.get("cleanup"):
            text = self._cleanup_text(text)

        if options.get("langDetect"):
            detected_lang = self._detect_language(text)
            if detected_lang:
                metadata = dict(metadata)
                metadata.setdefault("language", detected_lang)
                log(f"Detected language for doc_id={context.document_id}: {detected_lang}")

        try:
            stats = self._chunk_and_embed(
                doc_id=context.document_id,
                company_id=context.company_id,
                doc_type=resolved_source["doc_type"],
                source=resolved_source["source"],
                metadata=metadata,
                text=text,
                options=options,
            )
        except Exception as exc:  # noqa: BLE001
            finished_at = datetime.now(timezone.utc)
            error_msg = f"Embedding failed: {exc}"
            self._safe_update_index_state(
                context,
                state="failed",
                stats=base_stats,
                error=error_msg,
                extra={"index.finishedAt": finished_at},
            )
            self._log_document_event(context, level="error", message=error_msg, state="failed")
            raise

        finished_at = datetime.now(timezone.utc)
        self._log_document_event(
            context,
            level="info",
            message=f"Embedding completed with {stats['chunkCount']} chunks.",
            state="processing",
            details={"chunks": stats["chunkCount"], "tokenCount": stats["tokenCount"], "charCount": stats["charCount"]},
        )
        log(
            f"Completed doc_id={context.document_id} chunks={stats['chunkCount']} "
            f"tokens={stats['tokenCount']} chars={stats['charCount']}"
        )
        stats_with_flags = dict(stats)
        stats_with_flags.update(
            {
                "cleanup": bool(options.get("cleanup")),
                "ocr": bool(options.get("ocr")),
                "langDetect": bool(options.get("langDetect")),
                "scope": options.get("scope"),
                "minChars": options.get("minChars"),
            }
        )
        self._safe_update_index_state(
            context,
            state="completed",
            stats=stats_with_flags,
            error=None,
            extra={"index.finishedAt": finished_at},
        )
        self._log_document_event(
            context,
            level="info",
            message="Indexing job completed.",
            state="completed",
            details={"stats": stats_with_flags},
        )

    # ------------------------------------------------------------------
    def _load_document_content(
        self,
        document: Dict[str, Any],
        resolved_source: Dict[str, Any],
        context: DocumentJobContext,
    ) -> Tuple[str, Dict[str, Any]]:
        metadata = dict(resolved_source.get("metadata") or {})
        metadata.setdefault("documentId", context.document_id)
        metadata.setdefault("companyId", context.company_id)
        metadata.setdefault("source", resolved_source["source"])
        metadata.setdefault("jobId", context.job_id)
        if context.trigger:
            metadata.setdefault("trigger", context.trigger)
        if context.user_id:
            metadata.setdefault("userId", context.user_id)

        source_type = resolved_source["source"]
        if source_type == "upload":
            upload_id = resolved_source.get("upload_id")
            if not upload_id:
                raise ValueError("upload source missing upload_id")
            return self._load_upload_text(upload_id, metadata, context)

        if source_type in {"fetcher_crawl", "crawler", "fetcher"}:
            return self._load_fetcher_crawl_text(resolved_source, metadata, context)

        if source_type in {"import_url", "url", "web"}:
            url = resolved_source.get("url")
            if not url:
                raise ValueError("url source missing url")
            text, content_type = self._fetch_url_text(url)
            metadata.setdefault("url", url)
            if content_type:
                metadata.setdefault("contentType", content_type)
            return text, metadata

        text = resolved_source.get("text")
        if not text:
            raise ValueError("text source missing content")
        return str(text), metadata

    def _load_upload_text(
        self,
        upload_id: str,
        metadata: Dict[str, Any],
        context: DocumentJobContext,
    ) -> Tuple[str, Dict[str, Any]]:
        from services.document_loader import DocumentDownloadError, DocumentParseError

        try:
            upload_doc = self._get_upload_store().get_upload_by_id(upload_id)
        except UploadNotFoundError as exc:
            raise RuntimeError(f"upload record not found for uploadId={upload_id}") from exc

        file_doc = self._get_upload_store().get_file_by_upload_id(upload_id)
        bucket, key, filename = self._resolve_s3_location(upload_doc, file_doc)

        self._get_upload_store().update_upload_status(
            upload_id,
            index_status="in_progress",
            is_file_opened=False,
            file_open_error=None,
            embedding_doc_id=context.document_id,
        )

        download_retries = int(os.getenv("UPLOAD_DOWNLOAD_RETRIES") or DEFAULT_UPLOAD_DOWNLOAD_RETRIES)
        download_delay = float(
            os.getenv("UPLOAD_DOWNLOAD_RETRY_DELAY_SECONDS") or DEFAULT_UPLOAD_DOWNLOAD_RETRY_DELAY_SECONDS
        )
        attempts = max(1, download_retries)

        last_exc: Exception | None = None
        document = None
        for attempt in range(1, attempts + 1):
            if download_delay > 0:
                log(
                    "Upload download wait %ss (attempt %s/%s) uploadId=%s key=%s",
                    download_delay,
                    attempt,
                    attempts,
                    upload_id,
                    key,
                )
                time.sleep(download_delay)
            try:
                document = self._get_loader().fetch_text(key, bucket=bucket)
                last_exc = None
                break
            except DocumentParseError as exc:
                last_exc = exc
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < attempts:
                    log("Upload download failed (attempt %s/%s): %s", attempt, attempts, exc)
                    continue
                break

        if document is None:
            error_text = str(last_exc) if last_exc is not None else "unknown download error"
            self._get_upload_store().update_upload_status(
                upload_id,
                index_status="failed",
                is_file_opened=False,
                file_open_error=error_text,
            )
            if last_exc is not None:
                raise last_exc
            raise DocumentDownloadError(error_text)

        combined_metadata = dict(metadata)
        combined_metadata.setdefault("uploadId", upload_id)
        combined_metadata.setdefault("bucket", document.bucket)
        combined_metadata.setdefault("key", document.key)
        combined_metadata.setdefault("filename", document.filename or filename)
        if document.content_type:
            combined_metadata.setdefault("contentType", document.content_type)

        self._get_upload_store().update_upload_status(
            upload_id,
            index_status="completed",
            is_file_opened=True,
            file_open_error=None,
        )
        return document.text, combined_metadata

    def _load_fetcher_crawl_text(
        self,
        resolved_source: Dict[str, Any],
        metadata: Dict[str, Any],
        context: DocumentJobContext,
    ) -> Tuple[str, Dict[str, Any]]:
        domain = str(
            resolved_source.get("domain")
            or resolved_source.get("domain_id")
            or metadata.get("domain")
            or ""
        ).strip().lower()
        if not domain:
            raise ValueError("fetcher_crawl source missing domain")

        fetcher_store = self._get_fetcher_store()
        domain_doc = fetcher_store.get_domain(domain)
        if not domain_doc:
            raise RuntimeError(f"fetcher domain not found: {domain}")

        domain_company = (
            domain_doc.get("companyId")
            or domain_doc.get("companyid")
            or (domain_doc.get("config") or {}).get("companyId")
            or (domain_doc.get("config") or {}).get("companyid")
        )
        if not domain_company:
            raise RuntimeError(f"fetcher domain has no companyId: {domain}")
        if str(domain_company) != str(context.company_id):
            raise RuntimeError(
                f"fetcher domain company mismatch: domain={domain} expected={context.company_id} got={domain_company}"
            )

        site_profile = domain_doc.get("site_profile") if isinstance(domain_doc.get("site_profile"), dict) else {}
        source_hint = str(site_profile.get("source") or "").strip().lower()
        blocked_sources = {
            "system",
            "discovered",
            "crawler_discovered",
            "sitemap_discovered",
            "auto_discovered",
        }
        if source_hint in blocked_sources:
            raise RuntimeError(f"fetcher domain source is not user-provided: {source_hint}")

        try:
            page_limit = int(
                resolved_source.get("pageLimit")
                or resolved_source.get("limit")
                or resolved_source.get("maxPages")
                or DEFAULT_FETCHER_PAGE_LIMIT
            )
        except Exception:
            page_limit = DEFAULT_FETCHER_PAGE_LIMIT
        page_limit = max(1, min(page_limit, 500))

        fetched_from = self._parse_optional_datetime(
            resolved_source.get("fetchedFrom")
            or resolved_source.get("from")
            or resolved_source.get("startDate")
        )
        fetched_to = self._parse_optional_datetime(
            resolved_source.get("fetchedTo")
            or resolved_source.get("to")
            or resolved_source.get("endDate")
        )
        requested_storage = self._normalize_storage_preferences(
            resolved_source.get("storagePreference")
            or resolved_source.get("preferStorage")
            or resolved_source.get("storage")
        )
        requested_page_urls = self._normalize_fetcher_page_urls(
            resolved_source.get("pageUrls")
            or resolved_source.get("page_urls")
            or resolved_source.get("urls")
            or resolved_source.get("pages")
            or resolved_source.get("crawlPages")
        )
        if requested_page_urls:
            page_limit = max(page_limit, len(requested_page_urls))

        results = fetcher_store.list_crawl_results(
            domain=domain,
            limit=page_limit,
            fetched_from=fetched_from,
            fetched_to=fetched_to,
            urls=requested_page_urls,
        )
        if not results:
            if requested_page_urls:
                raise RuntimeError(
                    f"no crawl_results found for domain={domain} and requested URLs"
                )
            raise RuntimeError(f"no crawl_results found for domain={domain}")

        logs_by_url = fetcher_store.latest_logs_by_urls([str(row.get("url") or "").strip() for row in results])
        media_by_url = fetcher_store.list_crawl_media_by_parent_urls(
            [str(row.get("url") or "").strip() for row in results],
            limit_per_url=int(os.getenv("FETCHER_MEDIA_LIMIT_PER_URL") or DEFAULT_FETCHER_MEDIA_PER_PAGE),
        )
        s3_bucket = (
            os.getenv("FETCHER_S3_BUCKET_RAW")
            or os.getenv("S3_BUCKET_RAW")
            or os.getenv("AWS_S3_BUCKET")
            or ""
        ).strip()
        max_chars = int(os.getenv("FETCHER_CRAWL_MAX_TEXT_CHARS") or DEFAULT_FETCHER_MAX_TEXT_CHARS)

        selected_blocks: List[str] = []
        selected_urls: List[str] = []
        total_chars = 0
        for row in results:
            page_url = str(row.get("url") or "").strip()
            if not page_url:
                continue
            row_storage = requested_storage or self._normalize_storage_preferences(
                ((row.get("content_preferences") or {}).get("storage") if isinstance(row.get("content_preferences"), dict) else None)
            )
            if not row_storage:
                row_storage = ["db", "s3", "disk"]
            media_entries = media_by_url.get(page_url) or []

            text_value = self._resolve_fetcher_page_text(
                row=row,
                log_entry=logs_by_url.get(page_url),
                media_entries=media_entries,
                storage_preferences=row_storage,
                s3_bucket=s3_bucket,
            )
            if not text_value:
                continue

            title = str(row.get("title") or "").strip()
            header = f"[url] {page_url}"
            if title:
                header = f"{header}\n[title] {title}"
            block = f"{header}\n{text_value.strip()}"
            if not block.strip():
                continue

            if total_chars + len(block) > max_chars:
                if total_chars == 0:
                    block = block[:max_chars]
                else:
                    break
            selected_blocks.append(block)
            selected_urls.append(page_url)
            total_chars += len(block)
            if total_chars >= max_chars:
                break

        if not selected_blocks:
            raise RuntimeError(
                f"crawl content unavailable for domain={domain} (storage={requested_storage or ['db','s3','disk']})"
            )

        combined_metadata = dict(metadata)
        combined_metadata.setdefault("source", "fetcher_crawl")
        combined_metadata.setdefault("fetcherDomain", domain)
        combined_metadata.setdefault("fetcherPageCount", len(selected_blocks))
        combined_metadata.setdefault("fetcherTextChars", total_chars)
        if requested_page_urls:
            combined_metadata.setdefault("requestedPageCount", len(requested_page_urls))
        if selected_urls:
            combined_metadata.setdefault("firstUrl", selected_urls[0])

        return "\n\n---\n\n".join(selected_blocks), combined_metadata

    def _resolve_fetcher_page_text(
        self,
        *,
        row: Dict[str, Any],
        log_entry: Optional[Dict[str, Any]],
        media_entries: Optional[List[Dict[str, Any]]],
        storage_preferences: List[str],
        s3_bucket: str,
    ) -> str:
        markdown = str(row.get("markdown") or "").strip()
        html = str(row.get("html") or "").strip()
        media_text = self._serialize_fetcher_media(media_entries)

        for target in storage_preferences:
            if target == "db":
                if markdown:
                    return self._join_fetcher_text_with_media(markdown, media_text)
                if html:
                    return self._join_fetcher_text_with_media(html, media_text)
            elif target == "s3":
                text_value = self._load_fetcher_content_from_s3(log_entry, s3_bucket)
                if text_value:
                    return self._join_fetcher_text_with_media(text_value, media_text)
            elif target == "disk":
                text_value = self._load_fetcher_content_from_disk(log_entry)
                if text_value:
                    return self._join_fetcher_text_with_media(text_value, media_text)

        # Final fallback when preferences are missing/invalid.
        base_text = markdown or html
        return self._join_fetcher_text_with_media(base_text, media_text)

    def _load_fetcher_content_from_s3(self, log_entry: Optional[Dict[str, Any]], s3_bucket: str) -> str:
        if not log_entry or not s3_bucket:
            return ""
        s3_client = get_s3_client()

        json_key = str(log_entry.get("s3_key_json") or "").strip()
        if json_key:
            payload = self._read_s3_text(s3_client, s3_bucket, json_key)
            if payload:
                try:
                    data = json.loads(payload)
                    if isinstance(data, dict):
                        markdown = str(data.get("markdown") or "").strip()
                        html = str(data.get("html") or "").strip()
                        if markdown:
                            return markdown
                        if html:
                            return html
                except Exception:
                    pass

        html_key = str(log_entry.get("s3_key") or "").strip()
        if html_key:
            return self._read_s3_text(s3_client, s3_bucket, html_key)
        return ""

    def _read_s3_text(self, s3_client, bucket: str, key: str) -> str:
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            raw_bytes = response["Body"].read()
            encoding = str(response.get("ContentEncoding") or "").strip().lower()
            should_unzip = encoding == "gzip" or str(key).endswith(".gz")
            if should_unzip:
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes), mode="rb") as gz:
                        raw_bytes = gz.read()
                except Exception:
                    return ""
            return raw_bytes.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    @staticmethod
    def _load_fetcher_content_from_disk(log_entry: Optional[Dict[str, Any]]) -> str:
        if not log_entry:
            return ""
        disk_paths = log_entry.get("disk_paths")
        if not isinstance(disk_paths, list):
            return ""
        for path in disk_paths:
            path_str = str(path or "").strip()
            if not path_str:
                continue
            if not os.path.exists(path_str):
                continue
            try:
                with open(path_str, "r", encoding="utf-8", errors="replace") as file:
                    text = file.read().strip()
                if text:
                    return text
            except Exception:
                continue
        return ""

    @staticmethod
    def _join_fetcher_text_with_media(base_text: str, media_text: str) -> str:
        base = str(base_text or "").strip()
        media = str(media_text or "").strip()
        if base and media:
            return f"{base}\n\n[media]\n{media}"
        return base or media

    @staticmethod
    def _serialize_fetcher_media(entries: Optional[List[Dict[str, Any]]]) -> str:
        if not entries:
            return ""
        max_chars = int(os.getenv("FETCHER_MEDIA_MAX_TEXT_CHARS") or DEFAULT_FETCHER_MEDIA_TEXT_CHARS)
        chunks: List[str] = []
        total_chars = 0

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            media_type = str(entry.get("type") or "").strip()
            src = str(entry.get("src") or "").strip()
            alt = str(entry.get("alt") or "").strip()
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            title = str(metadata.get("title") or metadata.get("name") or "").strip()
            caption = str(metadata.get("caption") or "").strip()
            line_parts = []
            if media_type:
                line_parts.append(f"type={media_type}")
            if src:
                line_parts.append(f"src={src}")
            if alt:
                line_parts.append(f"alt={alt}")
            if title:
                line_parts.append(f"title={title}")
            if caption:
                line_parts.append(f"caption={caption}")
            if not line_parts:
                continue
            line = " | ".join(line_parts)
            if total_chars + len(line) + 1 > max_chars:
                break
            chunks.append(line)
            total_chars += len(line) + 1
        return "\n".join(chunks)

    @staticmethod
    def _parse_optional_datetime(raw_value: Any) -> Optional[datetime]:
        if raw_value is None:
            return None
        if isinstance(raw_value, datetime):
            return raw_value
        value = str(raw_value).strip()
        if not value:
            return None
        try:
            # Accept both `...Z` and timezone-aware ISO8601.
            value = value.replace("Z", "+00:00")
            return datetime.fromisoformat(value)
        except Exception:
            return None

    @staticmethod
    def _normalize_storage_preferences(raw_value: Any) -> List[str]:
        allowed = {"db", "s3", "disk"}
        values: List[str] = []
        if raw_value is None:
            return values
        if isinstance(raw_value, str):
            chunks = [v.strip().lower() for v in raw_value.split(",")]
        elif isinstance(raw_value, list):
            chunks = [str(v).strip().lower() for v in raw_value]
        else:
            return values
        for item in chunks:
            if item and item in allowed and item not in values:
                values.append(item)
        return values

    @staticmethod
    def _normalize_fetcher_page_urls(raw_value: Any) -> List[str]:
        if raw_value is None:
            return []
        values: List[Any]
        if isinstance(raw_value, str):
            candidate = raw_value.strip()
            if not candidate:
                return []
            if candidate.startswith("["):
                try:
                    parsed = json.loads(candidate)
                    values = parsed if isinstance(parsed, list) else [candidate]
                except Exception:
                    values = [candidate]
            else:
                values = [part.strip() for part in candidate.split(",") if part.strip()]
        elif isinstance(raw_value, dict):
            values = [raw_value]
        elif isinstance(raw_value, list):
            values = raw_value
        else:
            return []

        normalized: List[str] = []
        seen = set()
        for item in values:
            if isinstance(item, dict):
                value = (
                    item.get("url")
                    or item.get("href")
                    or item.get("link")
                    or item.get("pageUrl")
                    or item.get("page_url")
                )
            else:
                value = item
            text = str(value or "").strip()
            if not text:
                continue
            if "#" in text:
                text = text.split("#", 1)[0].strip()
            if text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    # ------------------------------------------------------------------
    def _chunk_and_embed(
        self,
        *,
        doc_id: str,
        company_id: str,
        doc_type: str,
        source: str,
        metadata: Dict[str, Any],
        text: str,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        chunk_size = int(options.get("chunkSize") or self.chunk_size)
        chunk_overlap = int(options.get("chunkOverlap") or self.chunk_overlap)
        min_chars = int(options.get("minChars") or 80)

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            min_chars=min_chars,
        )
        if not chunks:
            return {
                "chunkCount": 0,
                "tokenCount": 0,
                "charCount": 0,
                "chunkSize": chunk_size,
                "chunkOverlap": chunk_overlap,
                "minChars": min_chars,
            }

        faiss_ids = self._get_store().reserve_faiss_ids(len(chunks))
        engine = self._get_engine()
        embeddings = engine.encode([chunk.text for chunk in chunks], batch_size=self.batch_size)
        engine.add_embeddings(embeddings, faiss_ids)

        now = datetime.now(timezone.utc)
        chunk_docs: List[Dict[str, Any]] = []
        total_chars = 0
        total_tokens = 0
        for chunk, faiss_id in zip(chunks, faiss_ids):
            total_chars += len(chunk.text)
            total_tokens += self._estimate_token_count(chunk.text)
            chunk_metadata = dict(metadata)
            chunk_docs.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "company_id": company_id,
                    "chunk_index": chunk.index,
                    "faiss_id": int(faiss_id),
                    "text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "doc_type": doc_type,
                    "source": source,
                    "metadata": chunk_metadata,
                    "options": dict(options),
                    "created_at": now,
                }
            )
        self._get_store().insert_chunks(chunk_docs)
        return {
            "chunkCount": len(chunk_docs),
            "tokenCount": total_tokens,
            "charCount": total_chars,
            "chunkSize": chunk_size,
            "chunkOverlap": chunk_overlap,
            "minChars": min_chars,
        }

    # ------------------------------------------------------------------
    def _process_legacy_payload(self, payload: Dict[str, Any]) -> None:
        doc_id = payload.get("doc_id")
        if not doc_id:
            raise ValueError("payload missing 'doc_id'")

        ingest_type = (payload.get("ingest_type") or "web").lower()
        chunk_size = int(payload.get("chunk_size") or self.chunk_size)
        chunk_overlap = int(payload.get("chunk_overlap") or self.chunk_overlap)
        min_chars = int(payload.get("min_chars") or 80)

        if ingest_type == "upload":
            upload_id = payload.get("upload_id")
            if not upload_id:
                raise ValueError("payload missing 'upload_id'")
            metadata = payload.get("metadata") or {}
            self._process_upload_ingest_legacy(
                doc_id=doc_id,
                upload_id=upload_id,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chars=min_chars,
            )
        else:
            text = payload.get("text")
            if not text or not isinstance(text, str):
                raise ValueError("payload missing 'text'")
            doc_type = (payload.get("doc_type") or "web").lower()
            source = payload.get("source") or "web"
            metadata = payload.get("metadata") or {}
            self._ingest_text_legacy(
                doc_id=doc_id,
                doc_type=doc_type,
                source=source,
                metadata=metadata,
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chars=min_chars,
            )

    def _ingest_text_legacy(
        self,
        *,
        doc_id: str,
        doc_type: str,
        source: str,
        metadata: Dict[str, Any],
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        min_chars: int,
    ) -> None:
        self._get_store().update_document_status(doc_id, status="processing", error=None)

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            min_chars=min_chars,
        )
        if not chunks:
            self._get_store().update_document_status(doc_id, status="ready", chunk_count=0)
            return

        faiss_ids = self._get_store().reserve_faiss_ids(len(chunks))
        engine = self._get_engine()
        embeddings = engine.encode([chunk.text for chunk in chunks], batch_size=self.batch_size)
        engine.add_embeddings(embeddings, faiss_ids)

        now = datetime.now(timezone.utc)
        chunk_docs: List[Dict[str, Any]] = []
        for chunk, faiss_id in zip(chunks, faiss_ids):
            chunk_docs.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "chunk_index": chunk.index,
                    "faiss_id": int(faiss_id),
                    "text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "doc_type": doc_type,
                    "source": source,
                    "metadata": metadata,
                    "created_at": now,
                }
            )
        self._get_store().insert_chunks(chunk_docs)
        self._get_store().update_document_status(doc_id, status="ready", chunk_count=len(chunk_docs))
        log(f"Processed doc_id={doc_id} chunks={len(chunk_docs)}")

    def _process_upload_ingest_legacy(
        self,
        *,
        doc_id: str,
        upload_id: str,
        metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int,
        min_chars: int,
    ) -> None:
        from services.document_loader import DocumentDownloadError, DocumentParseError

        try:
            upload_doc = self._get_upload_store().get_upload_by_id(upload_id)
        except UploadNotFoundError as exc:
            self._get_store().update_document_status(doc_id, status="failed", error=str(exc))
            raise

        file_doc = self._get_upload_store().get_file_by_upload_id(upload_id)
        bucket, key, filename = self._resolve_s3_location(upload_doc, file_doc)

        self._get_upload_store().update_upload_status(
            upload_id,
            index_status="in_progress",
            is_file_opened=False,
            file_open_error=None,
            embedding_doc_id=doc_id,
        )

        try:
            document = self._get_loader().fetch_text(key, bucket=bucket)
        except (DocumentDownloadError, DocumentParseError) as exc:
            self._get_store().update_document_status(doc_id, status="failed", error=str(exc))
            self._get_upload_store().update_upload_status(
                upload_id,
                index_status="failed",
                is_file_opened=False,
                file_open_error=str(exc),
            )
            raise

        combined_metadata = dict(metadata)
        combined_metadata.setdefault("upload_id", upload_id)
        combined_metadata.setdefault("bucket", document.bucket)
        combined_metadata.setdefault("key", document.key)
        combined_metadata.setdefault("filename", document.filename or filename)
        if document.content_type:
            combined_metadata.setdefault("content_type", document.content_type)

        try:
            self._ingest_text_legacy(
                doc_id=doc_id,
                doc_type="document",
                source="upload",
                metadata=combined_metadata,
                text=document.text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chars=min_chars,
            )
        except Exception as exc:  # noqa: BLE001
            self._get_upload_store().update_upload_status(
                upload_id,
                index_status="failed",
                is_file_opened=True,
                file_open_error=str(exc),
            )
            raise

        self._get_upload_store().update_upload_status(
            upload_id,
            index_status="completed",
            is_file_opened=True,
            file_open_error=None,
        )

    # ------------------------------------------------------------------
    def _resolve_document_source(self, document: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        metadata = self._extract_metadata(document)

        source_info: Dict[str, Any] = {}
        for key in ("source", "ingest", "payload"):
            value = document.get(key)
            if isinstance(value, dict):
                source_info.update(value)
        # allow index.source to override
        index = document.get("index")
        if isinstance(index, dict) and isinstance(index.get("source"), dict):
            source_info.update(index["source"])

        source_type = (
            source_info.get("type")
            or source_info.get("source")
            or document.get("source")
            or options.get("source")
            or "text"
        )
        source_type = str(source_type).lower()

        doc_type = (
            document.get("docType")
            or document.get("type")
            or source_info.get("docType")
            or options.get("scope")
            or "document"
        )
        doc_type = str(doc_type).lower()

        upload_id = (
            source_info.get("uploadId")
            or source_info.get("upload_id")
            or source_info.get("uploadid")
            or document.get("uploadId")
            or document.get("upload_id")
            or document.get("uploadid")
            or metadata.get("uploadId")
            or metadata.get("upload_id")
            or metadata.get("uploadid")
        )

        payload_section = document.get("payload") if isinstance(document.get("payload"), dict) else {}

        url = (
            source_info.get("url")
            or payload_section.get("url")
            or document.get("url")
            or metadata.get("url")
        )

        text = (
            source_info.get("text")
            or payload_section.get("text")
            or document.get("text")
            or document.get("rawText")
            or document.get("body")
            or document.get("content")
            or metadata.get("text")
        )
        if isinstance(text, dict):
            text = text.get("value") or text.get("data")

        resolved: Dict[str, Any] = {
            "source": source_type,
            "doc_type": doc_type,
            "metadata": metadata,
        }
        if upload_id:
            resolved["upload_id"] = upload_id
        if url:
            resolved["url"] = url
        if text:
            resolved["text"] = text

        fetcher_domain = (
            source_info.get("domain")
            or source_info.get("domain_id")
            or payload_section.get("domain")
            or payload_section.get("domain_id")
            or document.get("domain")
            or document.get("domain_id")
            or metadata.get("domain")
        )
        if fetcher_domain:
            resolved["domain"] = str(fetcher_domain).strip().lower()

        for key in (
            "pageLimit",
            "limit",
            "maxPages",
            "fetchedFrom",
            "fetchedTo",
            "startDate",
            "endDate",
            "pageUrls",
            "page_urls",
            "urls",
            "pages",
            "crawlPages",
        ):
            value = (
                source_info.get(key)
                if key in source_info
                else payload_section.get(key)
                if key in payload_section
                else document.get(key)
                if key in document
                else metadata.get(key)
            )
            if value is not None:
                resolved[key] = value

        storage_pref = (
            source_info.get("storagePreference")
            or source_info.get("preferStorage")
            or source_info.get("storage")
            or payload_section.get("storagePreference")
            or payload_section.get("preferStorage")
            or payload_section.get("storage")
            or metadata.get("storagePreference")
            or metadata.get("preferStorage")
            or metadata.get("storage")
        )
        if storage_pref is not None:
            resolved["storagePreference"] = storage_pref
        return resolved

    @staticmethod
    def _extract_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for key in ("metadata", "meta", "attributes", "details"):
            value = document.get(key)
            if isinstance(value, dict):
                metadata.update(value)
        if "title" in document:
            metadata.setdefault("title", document.get("title"))
        company_id = document.get("companyId") or document.get("companyid")
        if company_id is not None:
            metadata.setdefault("companyId", str(company_id))
        document_id = document.get("documentId") or document.get("documentid") or document.get("_id")
        if document_id is not None:
            metadata.setdefault("documentId", str(document_id))
        upload_id = document.get("uploadId") or document.get("upload_id") or document.get("uploadid")
        if upload_id is not None:
            metadata.setdefault("uploadId", upload_id)
        return metadata

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        # Simple whitespace token estimate to avoid pulling in heavy tokenizers.
        if not text:
            return 0
        return len(text.split())

    @staticmethod
    def _cleanup_text(text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        compact = "\n".join(line for line in lines if line)
        return compact.strip()

    @staticmethod
    def _detect_language(text: str) -> Optional[str]:
        try:
            from langdetect import detect  # type: ignore
        except ImportError:
            return None
        try:
            return detect(text)
        except Exception:
            return None

    def _fetch_url_text(self, url: str) -> Tuple[str, Optional[str]]:
        try:
            response = requests.get(url, timeout=(10, 30))
            response.raise_for_status()
        except requests.RequestException as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to fetch URL {url}: {exc}") from exc
        content_type = response.headers.get("Content-Type") or response.headers.get("content-type")
        return response.text, content_type

    def _initial_stats(self, options: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "chunkCount": 0,
            "tokenCount": 0,
            "charCount": 0,
            "chunkSize": int(options.get("chunkSize") or self.chunk_size),
            "chunkOverlap": int(options.get("chunkOverlap") or self.chunk_overlap),
            "minChars": int(options.get("minChars") or 80),
        }

    def _log_document_event(
        self,
        context: DocumentJobContext,
        *,
        level: str,
        message: str,
        state: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            self._get_content_store().append_log_entry(
                company_id=context.company_id,
                document_id=context.document_id,
                job_id=context.job_id,
                level=level,
                message=message,
                state=state,
                user_id=context.user_id,
                trigger=context.trigger,
                details=details,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"Failed to append document log: {exc}")

    def _safe_update_index_state(
        self,
        context: DocumentJobContext,
        *,
        state: Optional[str] = None,
        stats: Any = UPDATE_UNSET,
        error: Any = UPDATE_UNSET,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            update_kwargs: Dict[str, Any] = {
                "company_id": context.company_id,
                "document_id": context.document_id,
                "job_id": context.job_id,
                "trigger": context.trigger,
                "options": context.options,
                "user_id": context.user_id,
                "extra_updates": extra,
            }
            if state is not None:
                update_kwargs["state"] = state
            if stats is not UPDATE_UNSET:
                update_kwargs["stats"] = stats
            if error is not UPDATE_UNSET:
                update_kwargs["error"] = error
            self._get_content_store().update_index_fields(**update_kwargs)
        except Exception as exc:  # noqa: BLE001
            log(f"Failed to update index state for {context.document_id}: {exc}")

    # ------------------------------------------------------------------
    def _coalesce(self, data: Dict[str, Any], *candidates: str) -> Optional[Any]:
        if not isinstance(data, dict):
            return None
        for key in candidates:
            if key in data:
                return data[key]
        lowered = {k.lower(): v for k, v in data.items()}
        for key in candidates:
            lowered_key = key.lower()
            if lowered_key in lowered:
                return lowered[lowered_key]
            stripped = lowered_key.replace("_", "")
            if stripped in lowered:
                return lowered[stripped]
        return None

    def _resolve_s3_location(self, upload_doc: Dict[str, Any], file_doc: Optional[Dict[str, Any]]):
        bucket_default = os.getenv("AWS_S3_BUCKET")
        candidates: List[Dict[str, Any]] = []
        if isinstance(upload_doc, dict):
            candidates.append(upload_doc.get("file") or {})
            candidates.append(upload_doc.get("data") or {})
        if isinstance(file_doc, dict):
            candidates.append(file_doc)

        for blob in candidates:
            if not isinstance(blob, dict):
                continue
            bucket = self._extract_bucket(blob, bucket_default)
            key = self._extract_s3_key(blob, bucket)
            if key:
                filename = blob.get("filename") or blob.get("originalname") or os.path.basename(key)
                return bucket, key, filename

        if not bucket_default:
            raise RuntimeError("Unable to determine S3 bucket and no AWS_S3_BUCKET set.")
        raise RuntimeError("Unable to resolve S3 object key for upload.")

    @staticmethod
    def _extract_bucket(blob: Dict[str, Any], bucket_default: Optional[str]) -> Optional[str]:
        metadata = blob.get("metadata") if isinstance(blob.get("metadata"), dict) else {}
        return (
            blob.get("bucket")
            or blob.get("Bucket")
            or blob.get("bucket_name")
            or metadata.get("bucket")
            or metadata.get("Bucket")
            or bucket_default
        )

    def _extract_s3_key(self, blob: Dict[str, Any], bucket: Optional[str]) -> Optional[str]:
        metadata = blob.get("metadata") if isinstance(blob.get("metadata"), dict) else {}
        direct_key = (
            blob.get("key")
            or blob.get("Key")
            or blob.get("s3Key")
            or metadata.get("key")
            or metadata.get("Key")
            or metadata.get("s3Key")
        )
        if isinstance(direct_key, str) and direct_key.strip():
            return direct_key.strip()

        path_like = (
            blob.get("path")
            or blob.get("url")
            or blob.get("location")
            or metadata.get("path")
            or metadata.get("url")
            or metadata.get("location")
        )
        if not isinstance(path_like, str) or not path_like.strip():
            return None
        return self._normalise_s3_key_from_path(path_like.strip(), bucket)

    @staticmethod
    def _normalise_s3_key_from_path(path_like: str, bucket: Optional[str]) -> Optional[str]:
        # Accept plain object keys.
        if "://" not in path_like and not path_like.startswith("s3://"):
            return path_like.lstrip("/")

        if path_like.startswith("s3://"):
            without = path_like[5:]
            if "/" not in without:
                return None
            _, key = without.split("/", 1)
            return unquote(key.lstrip("/"))

        parsed = urlparse(path_like)
        host = (parsed.netloc or "").lower()
        path = unquote((parsed.path or "").lstrip("/"))
        if not path:
            return None

        # path-style URL: s3.<region>.amazonaws.com/<bucket>/<key>
        if host.startswith("s3.") or host == "s3.amazonaws.com":
            if "/" in path:
                bucket_from_path, key = path.split("/", 1)
                if bucket and bucket_from_path == bucket:
                    return key
                return key if bucket_from_path else path
            return None

        # virtual-host style URL: <bucket>.s3.<region>.amazonaws.com/<key>
        if ".s3." in host and path:
            return path

        # CDN/custom domain fallback: use URL path as key.
        return path


def _install_signal_handlers(worker: IngestWorker) -> None:
    def _handler(signum, frame):  # noqa: D401
        worker.stop()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def main() -> None:
    worker = IngestWorker()
    _install_signal_handlers(worker)
    worker.start()


if __name__ == "__main__":
    main()
