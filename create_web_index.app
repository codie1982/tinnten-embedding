#!/usr/bin/env python3
"""Utility to inspect and ingest the tur_Latn HuggingFace web shards into FAISS."""

from __future__ import annotations

import argparse
import os
import re
import textwrap
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from dotenv import load_dotenv
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from services.chunker import chunk_text
from services.mongo_store import MongoStore
from services.rabbit_publisher import RabbitPublisher


load_dotenv()

DEFAULT_DATASET_DIR = Path(
    os.environ.get(
        "TURLATN_DATASET_DIR",
        "/home/codie/developer/huggingface/fineweb-2/fineweb-2/data/tur_Latn/train",
    )
)
DEFAULT_MODEL = os.environ.get(
    "WEBSEARCH_MODEL",
    os.environ.get("CHUNK_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
)
DEFAULT_INDEX_PATH = os.environ.get("WEBSEARCH_INDEX_PATH", "websearch.index")
DEFAULT_QUEUE_NAME = os.environ.get("WEBSEARCH_QUEUE_NAME") or os.environ.get("EMBED_QUEUE_NAME") or ""
DEFAULT_CHUNK_SIZE = int(os.environ.get("EMBED_CHUNK_SIZE") or 1200)
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("EMBED_CHUNK_OVERLAP") or 200)
DEFAULT_MIN_CHARS = int(os.environ.get("EMBED_MIN_CHARS") or 80)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or build a FAISS index from the tur_Latn HuggingFace parquet shards."
    )
    parser.add_argument("--dataset-dir", "-d", type=Path, default=DEFAULT_DATASET_DIR, help="Root folder containing the tur_Latn parquet files.")
    subparsers = parser.add_subparsers(dest="command")

    preview = subparsers.add_parser("preview", help="Show a few rows without ingesting anything.")
    preview.add_argument("--text-width", type=int, default=320, help="Maximum number of characters to show from the page text.")
    preview.add_argument("--count", "-n", type=int, default=5, help="How many entries to print.")

    build = subparsers.add_parser("build", help="Ingest records into a FAISS index and MongoDB.")
    build.add_argument("--model-name", "-m", default=DEFAULT_MODEL, help="SentenceTransformer model to encode chunks.")
    build.add_argument("--index-path", "-i", default=DEFAULT_INDEX_PATH, help="Path where the FAISS index should live.")
    build.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Target characters per chunk.")
    build.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Character overlap between chunks.")
    build.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS, help="Minimum characters required for a chunk.")
    build.add_argument("--batch-size", type=int, default=32, help="SentenceTransformer batch size.")
    build.add_argument("--max-records", type=int, default=10, help="Stop after ingesting this many records (0 = no limit).")
    build.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start row index (0-based, inclusive).",
    )
    build.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End row index (0-based, exclusive).",
    )
    build.add_argument(
        "--write-mode",
        choices=("queue",),
        default="queue",
        help="Sadece embed edip kuyruğa gönderir (FAISS yazmaz).",
    )
    build.add_argument(
        "--queue-name",
        default=DEFAULT_QUEUE_NAME,
        help="RabbitMQ queue name for write-mode=queue.",
    )
    build.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Doc_id mevcutsa atla. Varsayılan: atlama (False).",
    )
    build.add_argument(
        "--skip-batch-size",
        type=int,
        default=500,
        help="Batch size for checking existing doc_ids when --skip-existing is enabled.",
    )
    build.add_argument("--silent", action="store_true", help="Reduce log chatter.")

    count = subparsers.add_parser("count", help="Report total number of records across the dataset.")
    parser.set_defaults(command="preview")
    return parser.parse_args()


def _find_first_parquet_file(directory: Path) -> Optional[Path]:
    if directory.is_file() and directory.suffix == ".parquet":
        return directory
    if not directory.exists():
        return None
    for candidate in sorted(directory.rglob("*.parquet")):
        if candidate.is_file():
            return candidate
    return None


def _list_parquet_files(directory: Path) -> List[Path]:
    if directory.is_file() and directory.suffix == ".parquet":
        return [directory]
    if not directory.exists():
        return []
    return [candidate for candidate in sorted(directory.rglob("*.parquet")) if candidate.is_file()]


def _iter_parquet_records(
    directory: Path,
    limit: Optional[int] = None,
    *,
    start: int = 0,
    end: Optional[int] = None,
    parquet_files: Optional[List[Path]] = None,
    progress: Optional[Callable[[Path, int, int], None]] = None,
) -> Iterable[Dict[str, Any]]:
    def _scan_file(parquet_file: Path) -> Iterable[Dict[str, Any]]:
        reader = pq.ParquetFile(parquet_file)
        for group_index in range(reader.num_row_groups):
            table = reader.read_row_group(group_index, use_threads=True)
            for row in table.to_pylist():
                yield row

    if start < 0:
        start = 0
    if end is not None and end < 0:
        end = None

    current_index = 0
    returned = 0

    def _maybe_yield(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        nonlocal current_index, returned
        if end is not None and current_index >= end:
            return None
        output = None
        if current_index >= start:
            output = record
            returned += 1
        current_index += 1
        if output is None:
            return None
        if limit and returned >= limit:
            return output
        return output

    parquet_files = parquet_files or _list_parquet_files(directory)
    if not parquet_files:
        return
    total_files = len(parquet_files)

    for file_index, parquet_file in enumerate(parquet_files, start=1):
        if progress is not None:
            progress(parquet_file, file_index, total_files)
        for record in _scan_file(parquet_file):
            maybe = _maybe_yield(record)
            if maybe is not None:
                yield maybe
                if limit and returned >= limit:
                    return
            if end is not None and current_index >= end:
                return


def _short_text(value: Optional[str], width: int) -> str:
    if not value:
        return ""
    collapsed = " ".join(value.split())
    return textwrap.shorten(collapsed, width=width, placeholder=" ...")


def _safe_doc_id(raw: Any) -> str:
    payload = str(raw or "").strip()
    if not payload:
        payload = str(uuid.uuid4())
    slug = re.sub(r"[^a-zA-Z0-9_.:-]+", "-", payload)
    return f"turLatn-{slug}"


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _count_parquet_rows(directory: Path) -> int:
    total = 0

    def _add_file(file_path: Path) -> None:
        nonlocal total
        try:
            total += int(pq.ParquetFile(file_path).metadata.num_rows)
        except Exception:
            pass

    if directory.is_file() and directory.suffix == ".parquet":
        _add_file(directory)
        return total

    if not directory.exists():
        return 0

    for parquet_file in sorted(directory.rglob("*.parquet")):
        if parquet_file.is_file():
            _add_file(parquet_file)
    return total


def _run_preview(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.expanduser()
    parquet_file = _find_first_parquet_file(dataset_dir)
    if parquet_file is None:
        raise SystemExit(f"No parquet shards found under {dataset_dir!r}.")

    print(f"Reading from {parquet_file}")
    print(f"Showing up to {args.count} entries (text width {args.text_width})\n")

    reader = pq.ParquetFile(parquet_file)
    shown = 0
    for group_index in range(reader.num_row_groups):
        table = reader.read_row_group(group_index, use_threads=True)
        for record in table.to_pylist():
            shown += 1
            print(f"Entry {shown}:")
            print(f"  id: {record.get('id')}")
            print(f"  url: {record.get('url')}")
            print(f"  language: {record.get('language')} ({record.get('language_script')})")
            print(f"  dump: {record.get('dump')}")
            print(f"  date: {record.get('date')}")
            print(f"  text: {_short_text(record.get('text'), width=args.text_width)}\n")
            if shown >= args.count:
                return


def _run_count(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.expanduser()
    total = _count_parquet_rows(dataset_dir)
    print(f"Dataset içinde toplam {total} kayıt var.")


def _record_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key in ("url", "dump", "file_path", "language", "language_script", "date"):
        value = record.get(key)
        if value is not None:
            meta[key] = value
    for key in ("language_score", "minhash_cluster_size", "top_langs"):
        if key in record and record[key] is not None:
            meta[key] = record[key]
    meta.setdefault("source", "tur_Latn")
    return meta


def _run_build(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.expanduser()
    if not dataset_dir.exists():
        raise SystemExit(f"{dataset_dir!r} does not exist.")

    total_records = _count_parquet_rows(dataset_dir)
    parquet_files = _list_parquet_files(dataset_dir)
    if not parquet_files:
        raise SystemExit(f"No parquet shards found under {dataset_dir!r}.")
    print(f"Toplam veri sayısı: {total_records} (parquet dosya sayısı: {len(parquet_files)})")

    use_queue = True
    embedder = SentenceTransformer(args.model_name)
    publisher = RabbitPublisher(queue_name=args.queue_name or None)
    store = MongoStore()
    if not args.silent:
        print(f"Kuyruk modu aktif. Hedef kuyruk: {publisher.queue_name}")
        print(f"Worker için: EMBED_QUEUE_NAME={publisher.queue_name}")

    chunk_options: Dict[str, Any] = {
        "chunkSize": args.chunk_size,
        "chunkOverlap": args.chunk_overlap,
        "minChars": args.min_chars,
    }

    processed = 0
    chunked = 0
    queued = 0
    queued_chunks = 0
    skipped = 0
    skipped_existing = 0
    start_time = time.time()

    max_records = args.max_records if args.max_records > 0 else None
    start_index = int(args.start_index or 0)
    end_index = args.end_index if args.end_index is None else int(args.end_index)
    if end_index is not None and end_index <= start_index:
        print("Bitiş indeksi başlangıçtan küçük; işlem yapılmadı.")
        return
    if start_index or end_index is not None:
        end_desc = end_index if end_index is not None else "son"
        print(f"İşlenecek kayıt aralığı: {start_index}..{end_desc} (bitiş hariç)")
    effective_end = end_index if end_index is not None else total_records
    if total_records:
        effective_end = min(effective_end, total_records)
    if start_index >= effective_end:
        print("Başlangıç indeksi veri sonunu aşıyor; işlem yapılmadı.")
        return
    range_total = max(0, effective_end - start_index)
    total_to_process = range_total if max_records is None else min(max_records, range_total)
    if total_to_process <= 0:
        print("İşlenecek kayıt yok.")
        return
    print(f"İşlenecek toplam kayıt: {total_to_process}")
    limit_desc = max_records if max_records is not None else "sınırsız"
    print(f"İşlenecek kayıt limiti: {limit_desc}")

    def _progress_label() -> str:
        remaining = max(0, total_to_process - processed)
        label = f"{processed}/{total_to_process} kalan={remaining}"
        if start_index:
            label += f" sira={start_index + processed - 1}"
        return f"[{label}]"

    def _file_progress(path: Path, index: int, total: int) -> None:
        if args.silent:
            return
        remaining = total - index
        print(f"Parquet dosyası {index}/{total} (kalan {remaining}): {path}")

    def _process_batch(records: List[Dict[str, Any]]) -> None:
        nonlocal processed, chunked, queued, queued_chunks, skipped, skipped_existing
        prepared: List[Dict[str, Any]] = []
        doc_ids: List[str] = []

        for record in records:
            processed += 1
            doc_text = (record.get("text") or "").strip()
            if not doc_text:
                skipped += 1
                if not args.silent:
                    print(f"{_progress_label()} Boş metin, kayıt atlandı.")
                continue
            doc_url = record.get("url") or ""
            doc_id = _safe_doc_id(record.get("id") or doc_url)
            metadata = _record_metadata(record)
            metadata.setdefault("url", doc_url)
            metadata.setdefault("dataset", "tur_Latn")
            prepared.append(
                {
                    "record": record,
                    "doc_id": doc_id,
                    "doc_text": doc_text,
                    "doc_url": doc_url,
                    "metadata": metadata,
                }
            )
            doc_ids.append(doc_id)

        existing_docs = {}
        if args.skip_existing and doc_ids:
            existing_docs = store.get_documents_by_ids(doc_ids)

        for item in prepared:
            doc_id = item["doc_id"]
            doc_text = item["doc_text"]
            metadata = item["metadata"]
            if args.skip_existing and doc_id in existing_docs:
                skipped_existing += 1
                if not args.silent:
                    print(f"{_progress_label()} Var olan doc_id atlandı: {doc_id}")
                continue

            if not use_queue:
                store.create_document(doc_id=doc_id, doc_type="web", source="web", metadata=metadata)
                store.update_document_status(doc_id, status="processing", chunk_count=0)

            try:
                chunks = chunk_text(
                    doc_text,
                    chunk_size=args.chunk_size,
                    overlap=args.chunk_overlap,
                    min_chars=args.min_chars,
                )
                if not chunks:
                    if not use_queue:
                        store.update_document_status(doc_id, status="ready", chunk_count=0)
                    skipped += 1
                    continue

                chunk_texts = [chunk.text for chunk in chunks]
                store.create_document(doc_id=doc_id, doc_type="web", source="web", metadata=metadata)
                embeddings = embedder.encode(
                    chunk_texts,
                    batch_size=args.batch_size,
                    normalize_embeddings=True,
                )
                payload = {
                    "payload_type": "embedded_chunks",
                    "doc_id": doc_id,
                    "doc_type": "web",
                    "source": "web",
                    "metadata": metadata,
                    "chunks": [
                        {
                            "text": chunk.text,
                            "char_start": chunk.char_start,
                            "char_end": chunk.char_end,
                            "chunk_index": chunk.index,
                        }
                        for chunk in chunks
                    ],
                    "embeddings": embeddings.tolist(),
                    "options": dict(chunk_options),
                    "index_path": str(args.index_path),
                }
                publisher.publish(payload)
                store.update_document_status(doc_id, status="queued", chunk_count=len(chunks))
                queued += 1
                queued_chunks += len(chunks)
                if not args.silent:
                    print(f"{_progress_label()} Kuyruğa alındı: {len(chunks)} chunk doc_id={doc_id}")
            except Exception as exc:  # noqa: BLE001
                store.update_document_status(doc_id, status="failed", error=str(exc))
                print(f"Failed to index document {doc_id}: {exc}")

    batch: List[Dict[str, Any]] = []
    batch_size = max(1, int(args.skip_batch_size))
    for record in _iter_parquet_records(
        dataset_dir,
        limit=max_records,
        start=start_index,
        end=end_index,
        parquet_files=parquet_files,
        progress=_file_progress,
    ):
        batch.append(record)
        if len(batch) >= batch_size:
            _process_batch(batch)
            batch = []
    if batch:
        _process_batch(batch)

    elapsed = time.time() - start_time
    summary = (
        f"\nFinished: processed={processed}, chunked={chunked}, queued={queued}, "
        f"queued_chunks={queued_chunks}, skipped={skipped}, skipped_existing={skipped_existing}, "
        f"duration={elapsed:.2f}s, index={args.index_path}"
    )
    print(summary)


def main() -> None:
    args = _parse_args()
    if args.command == "preview":
        _run_preview(args)
        return
    if args.command == "build":
        _run_build(args)
        return
    if args.command == "count":
        _run_count(args)
        return
    _run_preview(args)


if __name__ == "__main__":
    main()
