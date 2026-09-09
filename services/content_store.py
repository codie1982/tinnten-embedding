"""
Repository helpers for `contentdocuments` and `contentdocumentlogs`.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError
from pymongo.collection import Collection

from init.db import get_database, get_mongo_client


DEFAULT_CONTENT_DB_NAME = "tinnten"
CONTENT_DOCUMENTS_COLL = "contentdocuments"
CONTENT_DOCUMENT_LOGS_COLL = "contentdocumentlogs"
EMBEDDING_DOCUMENT_LOGS_COLL = "embedding_contentdocumentlogs"

# Sentinel for optional update fields
_UNSET = object()


_CANONICAL_INDEX_STATES = {
    "processing": "indexing",
    "completed": "indexed",
    "failed": "error",
    "ready": "indexed",
    "pending": "queued",
}


def _canonical_index_state(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return _CANONICAL_INDEX_STATES.get(value.strip().lower(), value)


class IndexJobConflictError(RuntimeError):
    """A newer enqueue won the document-level compare-and-set race."""

    def __init__(
        self,
        document_id: str,
        current_job_id: Optional[str] = None,
        current_attempt: Optional[int] = None,
    ) -> None:
        super().__init__(f"a newer index job already owns document {document_id}")
        self.document_id = str(document_id)
        self.current_job_id = str(current_job_id) if current_job_id else None
        self.current_attempt = current_attempt


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


def _safe_object_id(value: Any) -> Optional[ObjectId]:
    if isinstance(value, ObjectId):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return ObjectId(value)
        except (InvalidId, TypeError):
            return None
    return None


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
            (
                db_name
                or os.getenv("CONTENT_DOCUMENT_DB_NAME")
                or os.getenv("EMBED_DOCUMENT_DB_NAME")
                or os.getenv("DB_TINNTEN")
                or ""
            ).strip()
            or DEFAULT_CONTENT_DB_NAME
        )
        self.db = get_database(name)
        self.documents: Collection = self.db[CONTENT_DOCUMENTS_COLL]
        self.logs: Collection = self.db[CONTENT_DOCUMENT_LOGS_COLL]
        self.embedding_logs: Collection = self.db[EMBEDDING_DOCUMENT_LOGS_COLL]
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
        self.embedding_logs.create_index(
            [("companyId", ASCENDING), ("documentId", ASCENDING), ("createdAt", ASCENDING)],
            name="embedding_document_log_idx",
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
        company_id: Optional[str],
        document_ids: Sequence[str],
        *,
        projection: Optional[MutableMapping[str, int]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch documents by company/document id pair and return a mapping keyed by documentId.
        If company_id is None, skip company filter (personal library).
        """
        if not document_ids:
            return {}
        doc_ids = list(dict.fromkeys(str(doc_id) for doc_id in document_ids))
        documents: Dict[str, Dict[str, Any]] = {}

        selectors = [self._document_selector(company_id, doc_id) for doc_id in doc_ids]
        query = selectors[0] if len(selectors) == 1 else {"$or": selectors}
        cursor = self.documents.find(query, projection)
        for doc in cursor:
            key = str(doc.get("documentId") or doc.get("_id"))
            documents[key] = doc
        return documents

    def get_document(self, company_id: Optional[str], document_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.find_one(self._document_selector(company_id, document_id))

    # ------------------------------------------------------------------
    # Index state helpers
    # ------------------------------------------------------------------
    def update_index_fields(
        self,
        *,
        company_id: Optional[str],
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
            canonical_state = _canonical_index_state(state)
            updates["index.state"] = canonical_state
            updates["indexState"] = canonical_state
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

        query = self._document_selector(company_id, document_id)
        # A worker may finish after a newer request replaced index.jobId.  Match
        # the same job (or a legacy record without one) before writing terminal
        # state.  The successful first write canonicalises missing jobId.
        if job_id is not _UNSET:
            query = {"$and": [query, self._current_job_selector(job_id)]}

        return self.documents.find_one_and_update(
            query,
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
            session=session,
        )

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

    # ------------------------------------------------------------------
    # Ingest lock — lease'li compare-and-set
    #
    # `update_index_fields` atomiktir ama LOCK DEĞİLDİR: beklenen bir duruma
    # koşul koymadan yazar, yani iki eşzamanlı re-ingest birbirinin chunk'larını
    # silebilir. Aşağıdaki API gerçek lock semantiği verir:
    #   * lease  → worker ölürse kilit kalıcı olmaz, süresi dolunca devralınır
    #   * job_id → kilidi yalnız sahibi bırakabilir/ilerletebilir
    #   * CAS    → eski bir job, yeni job'ın durumunu ezemez
    # RabbitMQ redelivery'si aynı job_id ile gelse bile canlı kilidi yeniden
    # alamaz. Aynı işi iki worker'ın eşzamanlı yürütmesi de veri yarışıdır.
    # ------------------------------------------------------------------
    def try_acquire_ingest_lock(
        self,
        *,
        company_id: Optional[str],
        document_id: str,
        job_id: str,
        lease_seconds: int = 900,
        session=None,
    ) -> Optional[Dict[str, Any]]:
        """Kilidi almayı dener. Alırsa güncel dokümanı, alamazsa None döner.

        Kilit yalnız dokümanın güncel jobId'si için ve şu durumlarda alınır:
          * hiç kilit yoksa,
          * lease süresi dolmuşsa (ölü worker).

        Canlı kilit aynı job_id'ye ait olsa dahi alınamaz. RabbitMQ aynı mesajı
        eşzamanlı yeniden teslim edebilir; ikinci worker'ın aynı sürümü paralel
        swap etmesi idempotent değil, veri yarışıdır.
        """
        now = datetime.now(timezone.utc)
        lease_until = now + timedelta(seconds=int(lease_seconds))
        base = self._document_selector(company_id, document_id)
        # The current-job selector must also cover the expired-lease branch.
        # Otherwise job A can load the document, job B can be enqueued, and A
        # can then steal its expired lock and overwrite index.jobId back to A.
        query = {
            "$and": [
                base,
                self._current_job_selector(job_id),
                {
                    "$or": [
                        {"index.lock": None},
                        {"index.lock.leaseUntil": {"$lt": now}},
                    ]
                },
            ]
        }
        updates = {
            "index.lock": {"jobId": job_id, "acquiredAt": now, "leaseUntil": lease_until},
            "index.jobId": job_id,
            "index.state": "indexing",
            "indexState": "indexing",
            "index.startedAt": now,
            "index.lastRunAt": now,
        }
        return self.documents.find_one_and_update(
            query,
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
            session=session,
        )

    def renew_ingest_lock(
        self,
        *,
        company_id: Optional[str],
        document_id: str,
        job_id: str,
        lease_seconds: int = 900,
        session=None,
    ) -> bool:
        """Uzun süren işlerde lease'i uzatır. Kilit bizde değilse False."""
        now = datetime.now(timezone.utc)
        query = {
            "$and": [
                self._document_selector(company_id, document_id),
                self._current_job_selector(job_id),
                {"index.lock.jobId": job_id},
                {"index.lock.leaseUntil": {"$gt": now}},
            ]
        }
        result = self.documents.update_one(
            query,
            {"$set": {"index.lock.leaseUntil": now + timedelta(seconds=int(lease_seconds))}},
            session=session,
        )
        return int(result.matched_count or 0) > 0

    def owns_ingest_lock(
        self,
        *,
        company_id: Optional[str],
        document_id: str,
        job_id: str,
        session=None,
    ) -> bool:
        """Return whether ``job_id`` still owns a live, canonical lease."""
        now = datetime.now(timezone.utc)
        query = {
            "$and": [
                self._document_selector(company_id, document_id),
                self._current_job_selector(job_id),
                {"index.lock.jobId": job_id},
                {"index.lock.leaseUntil": {"$gt": now}},
            ]
        }
        return self.documents.count_documents(query, limit=1, session=session) > 0

    def release_ingest_lock(
        self,
        *,
        company_id: Optional[str],
        document_id: str,
        job_id: str,
        state: Optional[str] = None,
        error: Any = _UNSET,
        session=None,
    ) -> bool:
        """Kilidi bırakır — YALNIZ sahibi olan job.

        `jobId` koşulu kritik: lease'i dolduğu için işi devralınmış ESKİ bir job
        geri dönüp durumu `ready`/`failed` yapamaz ve yeni job'ın kilidini
        düşüremez. Kilit bizde değilse hiçbir şey yazılmaz ve False döner.
        """
        now = datetime.now(timezone.utc)
        owner_query = {
            "$and": [
                self._document_selector(company_id, document_id),
                {"index.lock.jobId": job_id},
            ]
        }
        query = {"$and": [owner_query, self._current_job_selector(job_id)]}
        updates: Dict[str, Any] = {"index.lock": None, "index.finishedAt": now, "index.lastRunAt": now}
        if state is not None:
            canonical_state = _canonical_index_state(state)
            updates["index.state"] = canonical_state
            updates["indexState"] = canonical_state
        if error is not _UNSET:
            updates["index.errorMsg"] = error
        result = self.documents.update_one(query, {"$set": updates}, session=session)
        if int(result.matched_count or 0) > 0:
            return True

        # A newer enqueue can replace index.jobId while this worker still owns
        # the old lease.  Clear only that stale lease so the new job can start;
        # do not touch its state/timestamps/error.
        self.documents.update_one(
            owner_query,
            {"$set": {"index.lock": None}},
            session=session,
        )
        return False

    @staticmethod
    def _document_selector(company_id: Optional[str], document_id: str) -> Dict[str, Any]:
        """Match canonical and Node/Mongoose legacy identities safely.

        Canonical rows use ``companyId`` + ``documentId``.  Older Node rows use
        ``companyid`` + Mongo ``_id``.  When a company is supplied every branch
        remains company-scoped; there is no unscoped ``_id`` fallback.
        """
        document_text = str(document_id)
        document_obj = _safe_object_id(document_id)
        document_filters: List[Dict[str, Any]] = [{"documentId": document_text}]
        if document_obj is not None:
            document_filters.extend(
                [
                    {"documentId": document_obj},
                    {"_id": document_obj},
                ]
            )
        document_selector: Dict[str, Any] = {"$or": document_filters}

        if not company_id:
            return document_selector

        company_text = str(company_id)
        company_obj = _safe_object_id(company_id)
        company_filters: List[Dict[str, Any]] = [
            {"companyId": company_text},
            {"companyid": company_text},
        ]
        if company_obj is not None:
            company_filters.extend(
                [
                    {"companyId": company_obj},
                    {"companyid": company_obj},
                ]
            )
        return {
            "$and": [
                {"$or": company_filters},
                document_selector,
            ]
        }

    @staticmethod
    def _current_job_selector(job_id: Any) -> Dict[str, Any]:
        """Accept this job or a pre-jobId legacy record, reject newer jobs."""
        return {
            "$or": [
                {"index.jobId": str(job_id)},
                {"index.jobId": None},
            ]
        }

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
        attempt: Optional[int] = None,
        expected_job_id: Any = _UNSET,
        session=None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        existing_doc = self.get_document(company_id, document_id)
        existing_index = (
            existing_doc.get("index")
            if isinstance(existing_doc, dict) and isinstance(existing_doc.get("index"), dict)
            else {}
        )
        # API callers pass the job id observed at request admission. This makes
        # enqueue itself a CAS: a slow request cannot overwrite a newer job
        # which reached Mongo first. Direct/legacy callers retain the same
        # behaviour by deriving the expectation from the row read above.
        enqueue_expected_job_id = (
            existing_index.get("jobId")
            if expected_job_id is _UNSET
            else expected_job_id
        )
        if attempt is not None:
            attempt = int(attempt)
            if attempt < 1:
                raise ValueError("index attempt must be a positive integer")
        current_attempt = existing_index.get("attempt")
        if attempt is None and current_attempt is not None:
            # Once a producer adopts monotonic attempts, an older caller which
            # carries no ordering information must never supersede it.
            raise IndexJobConflictError(
                document_id,
                existing_index.get("jobId"),
                current_attempt,
            )
        stats = {
            "chunkCount": 0,
            "tokenCount": 0,
            "charCount": 0,
            "chunkSize": int(options.get("chunkSize") or 0),
            "chunkOverlap": int(options.get("chunkOverlap") or 0),
            "minChars": int(options.get("minChars") or 0),
        }

        canonical_state = _canonical_index_state(state)
        set_updates: Dict[str, Any] = {
            "metadata": metadata or {},
            "updatedAt": now,
            "index.source": source,
            "index.state": canonical_state,
            "indexState": canonical_state,
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
        if attempt is not None:
            set_updates["index.attempt"] = attempt
        if title:
            set_updates["title"] = title
        if trigger:
            set_updates["index.trigger"] = trigger
        if doc_type:
            set_updates["docType"] = doc_type

        set_on_insert = {
            "createdAt": now,
        }

        uses_legacy_schema = (
            isinstance(existing_doc, dict)
            and "_id" in existing_doc
            and "documentId" not in existing_doc
            and "companyId" not in existing_doc
        )
        if uses_legacy_schema:
            query: Dict[str, Any] = {"_id": existing_doc["_id"]}
            if "companyid" in existing_doc:
                query["companyid"] = existing_doc["companyid"]
            upsert = False
        else:
            query = {
                "companyId": company_id,
                "documentId": document_id,
            }
            set_updates["companyId"] = company_id
            set_updates["documentId"] = document_id
            upsert = True

        if attempt is None:
            query["index.attempt"] = None
            query["index.jobId"] = enqueue_expected_job_id
        else:
            # A larger attempt supersedes an older one. Equal attempts are
            # idempotent only for the same immutable job id; a different job
            # at the same attempt is ambiguous and therefore rejected.
            query["$or"] = [
                {"index.attempt": None},
                {"index.attempt": {"$lt": attempt}},
                {
                    "$and": [
                        {"index.attempt": attempt},
                        {"index.jobId": job_id},
                    ]
                },
            ]

        try:
            result = self.documents.update_one(
                query,
                {"$set": set_updates, "$setOnInsert": set_on_insert},
                upsert=upsert,
                session=session,
            )
        except DuplicateKeyError as exc:
            # A concurrent canonical insert/update changed index.jobId between
            # admission and this write. The unique identity index turns the
            # failed CAS upsert into DuplicateKeyError rather than matched=0.
            current = self.get_document(company_id, document_id) or {}
            current_index = current.get("index") if isinstance(current.get("index"), dict) else {}
            raise IndexJobConflictError(
                document_id,
                current_index.get("jobId"),
                current_index.get("attempt"),
            ) from exc

        if not result.matched_count and result.upserted_id is None:
            current = self.get_document(company_id, document_id) or {}
            current_index = current.get("index") if isinstance(current.get("index"), dict) else {}
            raise IndexJobConflictError(
                document_id,
                current_index.get("jobId"),
                current_index.get("attempt"),
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
        now = datetime.now(timezone.utc)
        content_document = self.get_document(company_id, document_id) or {}
        document_oid = (
            content_document.get("_id")
            if isinstance(content_document.get("_id"), ObjectId)
            else _safe_object_id(document_id)
        )
        company_oid = _safe_object_id(
            content_document.get("companyid")
            or content_document.get("companyId")
            or company_id
        )
        index = content_document.get("index") if isinstance(content_document.get("index"), dict) else {}
        user_oid = _safe_object_id(
            content_document.get("userid")
            or content_document.get("userId")
            or user_id
            or index.get("userId")
        )
        space_oid = _safe_object_id(
            content_document.get("spaceId")
            or (details or {}).get("spaceId")
        )

        if document_oid is not None and user_oid is not None:
            # Match the canonical Node/Mongoose model exactly. Raw Python
            # strings and missing `event` made rows invisible to ObjectId
            # queries and invalid under the owning schema.
            canonical_entry: Dict[str, Any] = {
                "documentId": document_oid,
                "companyid": company_oid,
                "spaceId": space_oid,
                "userid": user_oid,
                "event": f"index.{str(state or level or 'event').lower()}"[:64],
                "message": str(message or "")[:500],
                "meta": {
                    "jobId": job_id,
                    "level": level,
                    "state": state,
                    "trigger": trigger,
                    "details": details or {},
                },
                "createdAt": now,
                "updatedAt": now,
            }
            self.logs.insert_one(canonical_entry, session=session)
            return

        # Non-ObjectId legacy/crawler documents cannot satisfy the Node audit
        # schema. Keep their diagnostics in an explicitly separate collection
        # instead of polluting `contentdocumentlogs` with a second schema.
        fallback_logs = getattr(self, "embedding_logs", None)
        if fallback_logs is None:
            fallback_logs = self.db[EMBEDDING_DOCUMENT_LOGS_COLL]
        fallback_logs.insert_one(
            {
                "companyId": company_id,
                "documentId": document_id,
                "jobId": job_id,
                "level": level,
                "message": message,
                "state": state,
                "trigger": trigger,
                "userId": user_id,
                "details": details or {},
                "createdAt": now,
            },
            session=session,
        )

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
        fallback_logs = getattr(self, "embedding_logs", None)
        if fallback_logs is None:
            fallback_logs = self.db[EMBEDDING_DOCUMENT_LOGS_COLL]
        fallback_logs.insert_many(docs, ordered=False, session=session)
