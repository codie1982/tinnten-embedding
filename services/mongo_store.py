"""
MongoDB persistence helpers for embedding metadata and FAISS bookkeeping.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pymongo import ASCENDING, ReturnDocument
from pymongo.collection import Collection

from init.db import get_database


DEFAULT_EMBED_DB_NAME = "tinnten-embedding"
DEFAULT_DOCUMENT_DB_NAME = "tinnten"
DOCUMENTS_COLL = "embedding_documents"
CHUNKS_COLL = "embedding_chunks"
COUNTERS_COLL = "counters"
FAISS_COUNTER_KEY = "faiss_next_id"


class MongoStore:
    """
    Thin repository layer around Mongo collections used by the embedding service.

    Document metadata and chunk data live in separate databases so we can keep upload
    information inside the primary tinnten DB while embeddings/index metadata stays
    in the dedicated `tinnten-embedding` database.
    """

    def __init__(
        self,
        *,
        document_db_name: Optional[str] = None,
        chunk_db_name: Optional[str] = None,
    ) -> None:
        doc_db_name = (
            (document_db_name or os.getenv("EMBED_DOCUMENT_DB_NAME") or "").strip()
            or (os.getenv("DB_TINNTEN") or "").strip()
            or DEFAULT_DOCUMENT_DB_NAME
        )
        chunk_db_name = (
            (chunk_db_name or os.getenv("EMBED_DB_NAME") or "").strip()
            or DEFAULT_EMBED_DB_NAME
        )

        self.document_db = get_database(doc_db_name)
        self.chunk_db = get_database(chunk_db_name)

        self.documents: Collection = self.document_db[DOCUMENTS_COLL]
        self.chunks: Collection = self.chunk_db[CHUNKS_COLL]
        self.counters: Collection = self.chunk_db[COUNTERS_COLL]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.documents.create_index([("doc_id", ASCENDING)], unique=True, name="doc_id_unique")
        self.documents.create_index([("status", ASCENDING)], name="status_idx")
        self.chunks.create_index([("faiss_id", ASCENDING)], unique=True, name="faiss_id_unique")
        self.chunks.create_index([("doc_id", ASCENDING)], name="chunk_doc_idx")

    # ------------------------------------------------------------------
    # Document helpers
    # ------------------------------------------------------------------
    def create_document(
        self,
        *,
        doc_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session=None,
    ) -> str:
        """
        Insert a new document record with `pending` status.
        """
        now = datetime.now(timezone.utc)
        doc_id = doc_id or str(uuid.uuid4())
        on_insert = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source": source,
            "metadata": metadata or {},
            "status": "pending",
            "chunk_count": 0,
            "created_at": now,
            "updated_at": now,
            "error": None,
        }
        update_fields = {
            "doc_type": doc_type,
            "source": source,
            "metadata": metadata or {},
            "status": "pending",
            "chunk_count": 0,
            "updated_at": now,
            "error": None,
        }
        self.documents.update_one(
            {"doc_id": doc_id},
            {"$setOnInsert": on_insert, "$set": update_fields},
            upsert=True,
            session=session,
        )
        return doc_id

    def update_document_status(
        self,
        doc_id: str,
        *,
        status: str,
        chunk_count: Optional[int] = None,
        error: Optional[str] = None,
        session=None,
    ) -> None:
        now = datetime.now(timezone.utc)
        update: Dict[str, Any] = {
            "status": status,
            "updated_at": now,
            "error": error,
        }
        if chunk_count is not None:
            update["chunk_count"] = int(chunk_count)
        self.documents.update_one({"doc_id": doc_id}, {"$set": update}, upsert=False, session=session)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.find_one({"doc_id": doc_id})

    # ------------------------------------------------------------------
    # Chunk helpers
    # ------------------------------------------------------------------
    def insert_chunks(self, chunks: Iterable[Dict[str, Any]], *, session=None) -> None:
        docs = list(chunks)
        if not docs:
            return
        self.chunks.insert_many(docs, ordered=False, session=session)

    def get_chunks_by_faiss_ids(self, faiss_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        if not faiss_ids:
            return {}
        cursor = self.chunks.find({"faiss_id": {"$in": list(faiss_ids)}})
        return {int(doc["faiss_id"]): doc for doc in cursor}

    def get_chunks_by_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        cursor = self.chunks.find({"doc_id": doc_id}).sort("chunk_index", ASCENDING)
        return list(cursor)

    def delete_chunks_by_doc(self, doc_id: str, *, session=None) -> int:
        result = self.chunks.delete_many({"doc_id": doc_id}, session=session)
        return int(result.deleted_count)

    # ------------------------------------------------------------------
    # Counter helpers
    # ------------------------------------------------------------------
    def reserve_faiss_ids(self, count: int, *, session=None) -> List[int]:
        """
        Atomically reserve a block of FAISS IDs.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        result = self.counters.find_one_and_update(
            {"_id": FAISS_COUNTER_KEY},
            {"$inc": {"seq": count}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
            session=session,
        )
        seq = int(result.get("seq", 0))
        start = seq - count + 1
        return list(range(start, seq + 1))

    # ------------------------------------------------------------------
    # Document maintenance helpers
    # ------------------------------------------------------------------
    def delete_document(self, doc_id: str, *, session=None) -> int:
        result = self.documents.delete_one({"doc_id": doc_id}, session=session)
        return int(result.deleted_count)

    def get_documents_by_ids(self, doc_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        if not doc_ids:
            return {}
        cursor = self.documents.find({"doc_id": {"$in": list(doc_ids)}})
        return {doc["doc_id"]: doc for doc in cursor}
