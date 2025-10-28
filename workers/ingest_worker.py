"""
RabbitMQ worker that consumes ingest jobs, chunking and embedding them into FAISS + MongoDB.
"""
from __future__ import annotations

import json
import os
import signal
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pika
import uuid

from init.rabbit_connection import connect_rabbit_with_retry
from services.chunker import chunk_text
from services.embedding_engine import EmbeddingEngine
from services.mongo_store import MongoStore
from services.document_loader import (
    DocumentLoader,
    DocumentDownloadError,
    DocumentParseError,
)
from services.upload_store import UploadStore, UploadNotFoundError


DEFAULT_QUEUE_NAME = "embedding.ingest"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_BATCH_SIZE = 32


def log(msg: str) -> None:
    print(f"[ingest-worker] {msg}", flush=True)


class IngestWorker:
    def __init__(self) -> None:
        self.queue_name = (os.getenv("EMBED_QUEUE_NAME") or DEFAULT_QUEUE_NAME).strip()
        self.chunk_size = int(os.getenv("EMBED_CHUNK_SIZE") or DEFAULT_CHUNK_SIZE)
        self.chunk_overlap = int(os.getenv("EMBED_CHUNK_OVERLAP") or DEFAULT_CHUNK_OVERLAP)
        self.batch_size = int(os.getenv("EMBED_BATCH_SIZE") or DEFAULT_BATCH_SIZE)
        model_name = os.getenv(
            "MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )
        index_path = os.getenv("FAISS_INDEX_PATH", "faiss.index")

        self.store = MongoStore()
        self.upload_store = UploadStore()
        self.loader = DocumentLoader()
        self.engine = EmbeddingEngine(model_name=model_name, index_path=index_path)
        self.connection: pika.BlockingConnection | None = None
        self.channel: pika.adapters.blocking_connection.BlockingChannel | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    def start(self) -> None:
        log("Starting ingest worker...")
        self.connection = connect_rabbit_with_retry()
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name, durable=True, auto_delete=False, exclusive=False)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self._handle_message, auto_ack=False)
        log(f"Awaiting messages on queue '{self.queue_name}'")
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
            # attempt to mark document as failed if info available
            try:
                doc_id = payload.get("doc_id") if isinstance(payload, dict) else None
                if doc_id:
                    self.store.update_document_status(doc_id, status="failed", error=str(exc))
            except Exception:
                pass
            channel.basic_nack(delivery_tag=delivery_tag, requeue=False)

    def _process_payload(self, payload: Dict[str, Any]) -> None:
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
            self._process_upload_ingest(
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
            self._ingest_text(
                doc_id=doc_id,
                doc_type=doc_type,
                source=source,
                metadata=metadata,
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chars=min_chars,
            )

    # ------------------------------------------------------------------
    def _ingest_text(
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
        self.store.update_document_status(doc_id, status="processing", error=None)

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            min_chars=min_chars,
        )
        if not chunks:
            self.store.update_document_status(doc_id, status="ready", chunk_count=0)
            return

        faiss_ids = self.store.reserve_faiss_ids(len(chunks))
        embeddings = self.engine.encode([chunk.text for chunk in chunks], batch_size=self.batch_size)
        self.engine.add_embeddings(embeddings, faiss_ids)

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
        self.store.insert_chunks(chunk_docs)
        self.store.update_document_status(doc_id, status="ready", chunk_count=len(chunk_docs))
        log(f"Processed doc_id={doc_id} chunks={len(chunk_docs)}")

    def _process_upload_ingest(
        self,
        *,
        doc_id: str,
        upload_id: str,
        metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int,
        min_chars: int,
    ) -> None:
        try:
            upload_doc = self.upload_store.get_upload_by_id(upload_id)
        except UploadNotFoundError as exc:
            self.store.update_document_status(doc_id, status="failed", error=str(exc))
            raise

        file_doc = self.upload_store.get_file_by_upload_id(upload_id)
        bucket, key, filename = self._resolve_s3_location(upload_doc, file_doc)

        self.upload_store.update_upload_status(
            upload_id,
            index_status="in_progress",
            is_file_opened=False,
            file_open_error=None,
            embedding_doc_id=doc_id,
        )

        try:
            document = self.loader.fetch_text(key, bucket=bucket)
        except (DocumentDownloadError, DocumentParseError) as exc:
            self.store.update_document_status(doc_id, status="failed", error=str(exc))
            self.upload_store.update_upload_status(
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
            self._ingest_text(
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
            self.upload_store.update_upload_status(
                upload_id,
                index_status="failed",
                is_file_opened=True,
                file_open_error=str(exc),
            )
            raise

        self.upload_store.update_upload_status(
            upload_id,
            index_status="completed",
            is_file_opened=True,
            file_open_error=None,
        )

    def _resolve_s3_location(self, upload_doc: Dict[str, Any], file_doc: Optional[Dict[str, Any]]):
        bucket_default = os.getenv("AWS_S3_BUCKET")
        candidates = []
        if upload_doc:
            candidates.append(upload_doc.get("file") or {})
            candidates.append(upload_doc.get("data") or {})
        if file_doc:
            candidates.append(file_doc)

        for blob in candidates:
            if not isinstance(blob, dict):
                continue
            bucket = (
                blob.get("bucket")
                or blob.get("Bucket")
                or blob.get("bucket_name")
                or bucket_default
            )
            key = blob.get("key") or blob.get("Key") or blob.get("path") or blob.get("s3Key")
            if key:
                filename = blob.get("filename") or blob.get("originalname") or os.path.basename(key)
                return bucket, key, filename

        if not bucket_default:
            raise RuntimeError("Unable to determine S3 bucket and no AWS_S3_BUCKET set.")
        raise RuntimeError("Unable to resolve S3 object key for upload.")


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
