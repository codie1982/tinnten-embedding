"""
Repository helpers for `contentdocuments` and `contentdocumentlogs`.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence

from pymongo import ASCENDING, ReturnDocument
from pymongo.collection import Collection

from init.db import get_database, get_mongo_client


DEFAULT_CONTENT_DB_NAME = "tinnten"
CONTENT_DOCUMENTS_COLL = "contentdocuments"
CONTENT_DOCUMENT_LOGS_COLL = "contentdocumentlogs"

# Sentinel for optional update fields
_UNSET = object()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "t"}
    return bool(value)


def normalize_index_options(
    raw: Optional[Dict[str, Any]],
    *,
    default_chunk_size: int,
    default_chunk_overlap: int,
    default_min_chars: int = 80,
) -> Dict[str, Any]:
    """
    Normalise option payloads originating from REST controllers or worker messages.
    """
    opts = dict(raw or {})

    chunk_size_value = opts.pop("chunk_size", None) or opts.get("chunkSize")
    if chunk_size_value is None:
        chunk_size_value = default_chunk_size
    opts["chunkSize"] = int(chunk_size_value)

    chunk_overlap_value = opts.pop("chunk_overlap", None) or opts.get("chunkOverlap")
    if chunk_overlap_value is None:
        chunk_overlap_value = default_chunk_overlap
    opts["chunkOverlap"] = int(chunk_overlap_value)

    min_chars_value = None
    for key in ("minChars", "min_chars", "minchars", "min_chars_per_chunk"):
        if key in opts:
            min_chars_value = opts.pop(key)
            break
    if min_chars_value is None:
        min_chars_value = default_min_chars
    opts["minChars"] = int(min_chars_value)

    opts["cleanup"] = _coerce_bool(opts.get("cleanup"))
    opts["ocr"] = _coerce_bool(opts.get("ocr"))
    opts["langDetect"] = _coerce_bool(opts.get("langDetect") or opts.get("lang_detect"))

    scope_value = opts.get("scope")
    if scope_value is not None:
        opts["scope"] = str(scope_value)

    if "source" in opts and isinstance(opts["source"], str):
        opts["source"] = opts["source"].lower()

    return opts


class ContentDocumentStore:
    """
    Thin repository around the content document collections used by the API service.

    The Node API keeps document level state inside `contentdocuments.index.*` and
    exposes log history through `contentdocumentlogs`. The embedding worker mirrors
    those semantics so the UI can poll the API for progress updates.
    """

    def __init__(self, db_name: Optional[str] = None) -> None:
        name = (
            (db_name or os.getenv("CONTENT_DB_NAME") or "").strip()
            or (os.getenv("DB_TINNTEN") or "").strip()
            or DEFAULT_CONTENT_DB_NAME
        )
        self.db = get_database(name)
        self.documents: Collection = self.db[CONTENT_DOCUMENTS_COLL]
        self.logs: Collection = self.db[CONTENT_DOCUMENT_LOGS_COLL]
        self._ensure_indexes()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _ensure_indexes(self) -> None:
        self.documents.create_index(
            [("companyId", ASCENDING), ("documentId", ASCENDING)],
            unique=True,
            name="company_document_unique",
            # Only enforce uniqueness when both identifiers are present to avoid
            # legacy rows with null values blocking index creation.
            partialFilterExpression={
                # `$ne: null` gets rewritten as `$not: {$eq: null}` which is not
                # supported by older MongoDB versions in partial index filters.
                # `$gt: null` matches documents where the field exists and is not null.
                "companyId": {"$gt": None},
                "documentId": {"$gt": None},
            },
        )
        self.documents.create_index([("index.state", ASCENDING)], name="index_state_idx")
        self.logs.create_index(
            [("companyId", ASCENDING), ("documentId", ASCENDING), ("createdAt", ASCENDING)],
            name="document_log_idx",
        )

    @staticmethod
    def start_session():
        client = get_mongo_client()
        return client.start_session()

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def get_documents(
        self,
        company_id: str,
        document_ids: Sequence[str],
        *,
        projection: Optional[MutableMapping[str, int]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch documents by company/document id pair and return a mapping keyed by documentId.
        """
        if not document_ids:
            return {}
        cursor = self.documents.find(
            {"companyId": company_id, "documentId": {"$in": list(document_ids)}},
            projection,
        )
        return {doc["documentId"]: doc for doc in cursor}

    def get_document(self, company_id: str, document_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.find_one({"companyId": company_id, "documentId": document_id})

    # ------------------------------------------------------------------
    # Index state helpers
    # ------------------------------------------------------------------
    def update_index_fields(
        self,
        *,
        company_id: str,
        document_id: str,
        state: Optional[str] = None,
        stats: Any = _UNSET,
        error: Any = _UNSET,
        job_id: Any = _UNSET,
        trigger: Any = _UNSET,
        options: Any = _UNSET,
        user_id: Any = _UNSET,
        extra_updates: Optional[Dict[str, Any]] = None,
        session=None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generic helper mirroring the Node controller's index object structure.
        """
        now = datetime.now(timezone.utc)
        updates: Dict[str, Any] = {"index.lastRunAt": now}
        if state is not None:
            updates["index.state"] = state
        if stats is not _UNSET:
            updates["index.stats"] = stats
        if error is not _UNSET:
            updates["index.errorMsg"] = error
        if job_id is not _UNSET:
            updates["index.jobId"] = job_id
        if trigger is not _UNSET:
            updates["index.trigger"] = trigger
        if options is not _UNSET:
            updates["index.options"] = options
        if user_id is not _UNSET:
            updates["index.userId"] = user_id
        if extra_updates:
            for key, value in extra_updates.items():
                updates[key] = value

        result = self.documents.find_one_and_update(
            {"companyId": company_id, "documentId": document_id},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
            session=session,
        )
        return result

    def reset_index_error(
        self,
        *,
        company_id: str,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        return self.update_index_fields(
            company_id=company_id,
            document_id=document_id,
            error=None,
        )

    def upsert_document_with_source(
        self,
        *,
        company_id: str,
        document_id: str,
        source: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
        options: Dict[str, Any],
        job_id: str,
        user_id: Optional[str] = None,
        trigger: Optional[str] = None,
        title: Optional[str] = None,
        doc_type: Optional[str] = None,
        state: str = "queued",
        session=None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        stats = {
            "chunkCount": 0,
            "tokenCount": 0,
            "charCount": 0,
            "chunkSize": int(options.get("chunkSize") or 0),
            "chunkOverlap": int(options.get("chunkOverlap") or 0),
            "minChars": int(options.get("minChars") or 0),
        }

        set_updates: Dict[str, Any] = {
            "companyId": company_id,
            "documentId": document_id,
            "source": source,
            "metadata": metadata or {},
            "updatedAt": now,
            "index.state": state,
            "index.jobId": job_id,
            "index.options": options,
            "index.stats": stats,
            "index.errorMsg": None,
            "index.trigger": trigger,
            "index.userId": user_id,
            "index.lastRunAt": now,
            "index.queuedAt": now,
            "index.startedAt": None,
            "index.finishedAt": None,
        }
        if title:
            set_updates["title"] = title
        if trigger:
            set_updates["index.trigger"] = trigger
        if doc_type:
            set_updates["docType"] = doc_type

        set_on_insert = {
            "createdAt": now,
        }

        self.documents.update_one(
            {"companyId": company_id, "documentId": document_id},
            {"$set": set_updates, "$setOnInsert": set_on_insert},
            upsert=True,
            session=session,
        )
        doc = self.get_document(company_id, document_id)
        if not doc:
            raise RuntimeError("Failed to upsert content document.")
        return doc

    def append_log_entry(
        self,
        *,
        company_id: str,
        document_id: str,
        job_id: Optional[str],
        level: str,
        message: str,
        state: Optional[str] = None,
        user_id: Optional[str] = None,
        trigger: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        session=None,
    ) -> None:
        entry: Dict[str, Any] = {
            "companyId": company_id,
            "documentId": document_id,
            "jobId": job_id,
            "level": level,
            "message": message,
            "state": state,
            "trigger": trigger,
            "userId": user_id,
            "details": details or {},
            "createdAt": datetime.now(timezone.utc),
        }
        self.logs.insert_one(entry, session=session)

    def bulk_append_logs(
        self,
        entries: Iterable[Dict[str, Any]],
        *,
        session=None,
    ) -> None:
        docs: List[Dict[str, Any]] = list(entries)
        if not docs:
            return
        now = datetime.now(timezone.utc)
        for doc in docs:
            doc.setdefault("createdAt", now)
        self.logs.insert_many(docs, ordered=False, session=session)
