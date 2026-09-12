"""
MongoDB persistence helpers for embedding metadata and FAISS bookkeeping.
"""
from __future__ import annotations

import hashlib
import re
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
        # Sürüm-seçici swap sorguları (doc_id + ingest_version) için — eski sürümü
        # temizlerken ve aramada aktif sürümü süzerken sık kullanılır.
        self.chunks.create_index(
            [("doc_id", ASCENDING), ("ingest_version", ASCENDING)],
            name="chunk_doc_version_idx",
        )
        # FAZ 5 — hybrid retrieval altyapısı.
        # 1) Lexical ($text) index: dense FAISS'e paralel BM25-benzeri sözcük araması.
        #    default_language="none": içerik TR/EN karışık; Mongo'nun TR stemmer'ı yok
        #    ve EN stemmer'ı TR kelimeleri bozar → stemming KAPALI (birebir token eşleşmesi).
        # 2) (companyId, domain) compound: Faz 2 domainChunks agregasıyla paylaşımlı;
        #    lexical sorgunun firma/domain daraltmasını da hızlandırır.
        # Not: mevcut index'ler değişmediğinden idempotent — yeniden çalıştırılabilir.
        try:
            self.chunks.create_index(
                [("text", "text")],
                default_language="none",
                name="chunk_text_search",
            )
        except Exception:  # noqa: BLE001 — text index opsiyonel (ör. mongomock desteklemez)
            pass
        self.chunks.create_index(
            [("metadata.companyId", ASCENDING), ("metadata.domain", ASCENDING)],
            name="chunk_company_domain_idx",
        )
        # company_id ÜST SEVİYEDE de tutuluyor (ana ingest yolu) ve firma bazlı
        # sayımlar `$or: [{company_id}, {metadata.companyId}]` şeklinde sorguluyor.
        # Mongo $or'da index-union için HER dalın index'li olmasını ister; bu index
        # olmadan company_id dalı COLLSCAN'e düşer.
        # (company_id) tek başına da prefix olarak kullanılabildiği için compound
        # index hem firma sayımlarını hem kullanıcı filtresini karşılar.
        self.chunks.create_index(
            [("company_id", ASCENDING), ("user_id", ASCENDING)],
            name="chunk_company_user_idx",
        )

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
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        job_id: Optional[str] = None,
        status: str = "pending",
        session=None,
    ) -> str:
        """
        Insert or initialise a document record for an ingest attempt.

        Modern ``content/index`` jobs historically wrote chunks without ever
        creating this parent row. Detail endpoints then returned 404 and active
        chunk queries hid otherwise valid chunks. Optional ownership fields are
        kept top-level as well as in metadata for canonical/legacy readers.
        """
        now = datetime.now(timezone.utc)
        doc_id = doc_id or str(uuid.uuid4())
        on_insert = {
            "created_at": now,
        }
        update_fields = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source": source,
            "metadata": metadata or {},
            "status": status,
            "chunk_count": 0,
            "updated_at": now,
            "error": None,
        }
        if company_id is not None:
            update_fields["company_id"] = str(company_id)
        if user_id is not None:
            update_fields["user_id"] = str(user_id)
        if title is not None:
            update_fields["title"] = str(title)
        if job_id is not None:
            update_fields["job_id"] = str(job_id)
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
        expected_job_id: Optional[str] = None,
        session=None,
    ) -> bool:
        now = datetime.now(timezone.utc)
        update: Dict[str, Any] = {
            "status": status,
            "updated_at": now,
            "error": error,
        }
        if chunk_count is not None:
            update["chunk_count"] = int(chunk_count)
        query: Dict[str, Any] = {"doc_id": doc_id}
        if expected_job_id is not None:
            query["job_id"] = str(expected_job_id)
        result = self.documents.update_one(query, {"$set": update}, upsert=False, session=session)
        return int(result.matched_count or 0) > 0

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.find_one({"doc_id": doc_id})

    # ------------------------------------------------------------------
    # Chunk helpers
    # ------------------------------------------------------------------
    def insert_chunks(self, chunks: Iterable[Dict[str, Any]], *, session=None) -> None:
        docs = list(chunks)
        if not docs:
            return
        # Exact duplicate health can then scan compact hashes instead of pulling
        # every full chunk body. Legacy rows without the field remain supported
        # by ``active_chunk_duplicate_stats_by_company`` below.
        for doc in docs:
            if not doc.get("content_fingerprint") and isinstance(doc.get("text"), str):
                doc["content_fingerprint"] = self.content_fingerprint(doc["text"])
        self.chunks.insert_many(docs, ordered=False, session=session)

    @staticmethod
    def content_fingerprint(text: str) -> str:
        """Stable exact-content identity after harmless whitespace/case folding."""
        normalized = " ".join(str(text or "").casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""

    def get_chunks_by_faiss_ids(self, faiss_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        if not faiss_ids:
            return {}
        cursor = self.chunks.find({"faiss_id": {"$in": list(faiss_ids)}})
        return {int(doc["faiss_id"]): doc for doc in cursor}

    def get_chunks_by_doc(
        self, doc_id: str, *, ingest_version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"doc_id": doc_id}
        if ingest_version:
            query["ingest_version"] = str(ingest_version)
        cursor = self.chunks.find(query).sort("chunk_index", ASCENDING)
        return list(cursor)

    def get_chunks_page_by_doc(
        self,
        doc_id: str,
        *,
        page: int = 1,
        limit: int = 20,
        query: str = "",
        ingest_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return one bounded audit page without loading all chunk bodies."""
        safe_page = max(1, int(page or 1))
        safe_limit = min(100, max(1, int(limit or 20)))
        chunk_filter: Dict[str, Any] = {"doc_id": doc_id}
        if ingest_version:
            chunk_filter["ingest_version"] = str(ingest_version)
        normalized_query = str(query or "").strip()
        if normalized_query:
            chunk_filter["text"] = {"$regex": re.escape(normalized_query), "$options": "i"}

        total = int(self.chunks.count_documents(chunk_filter))
        document_total = int(self.chunks.count_documents({
            "doc_id": doc_id,
            **({"ingest_version": str(ingest_version)} if ingest_version else {}),
        }))
        total_pages = max(1, (total + safe_limit - 1) // safe_limit)
        bounded_page = min(safe_page, total_pages)
        skip = (bounded_page - 1) * safe_limit
        chunks = list(
            self.chunks.find(chunk_filter)
            .sort("chunk_index", ASCENDING)
            .skip(skip)
            .limit(safe_limit)
        )

        previous_chunk = None
        if chunks and not normalized_query:
            first_index = chunks[0].get("chunk_index")
            if first_index is not None:
                previous_chunk = self.chunks.find_one(
                    {
                        "doc_id": doc_id,
                        **({"ingest_version": str(ingest_version)} if ingest_version else {}),
                        "chunk_index": {"$lt": first_index},
                    },
                    sort=[("chunk_index", -1)],
                )

        return {
            "chunks": chunks,
            "previous_chunk": previous_chunk,
            "total": total,
            "document_total": document_total,
            "page": bounded_page,
            "limit": safe_limit,
            "total_pages": total_pages,
        }

    @staticmethod
    def _glob_url_regex(pattern: str) -> Optional[re.Pattern]:
        """Compile dashboard URL globs for stored absolute crawl URLs."""
        value = str(pattern or "").strip()
        if not value or len(value) > 500:
            return None
        pieces: List[str] = []
        index = 0
        while index < len(value):
            char = value[index]
            if char == "*":
                if index + 1 < len(value) and value[index + 1] == "*":
                    index += 1
                pieces.append(".*")
            elif char == "?":
                pieces.append(".")
            else:
                pieces.append(re.escape(char))
            index += 1
        prefix = r"(?:https?://[^/?#]+)?" if value.startswith("/") else ""
        try:
            return re.compile(rf"^{prefix}{''.join(pieces)}$", re.IGNORECASE)
        except re.error:
            return None

    @classmethod
    def _chunk_scope_query(cls, scope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        domains = [str(value).strip().lower() for value in scope.get("domains") or [] if str(value).strip()]
        if not domains:
            domain = str(scope.get("domain") or "").strip().lower()
            if domain:
                domains = [domain, f"www.{domain.removeprefix('www.')}"]
        if not domains:
            return None

        clauses: List[Dict[str, Any]] = [{"metadata.domain": {"$in": list(dict.fromkeys(domains))}}]
        include_regexes = [
            regex for regex in (
                cls._glob_url_regex(value) for value in (scope.get("includePatterns") or [])[:100]
            ) if regex is not None
        ]
        exclude_regexes = [
            regex for regex in (
                cls._glob_url_regex(value) for value in (scope.get("excludePatterns") or [])[:100]
            ) if regex is not None
        ]
        if include_regexes:
            clauses.append({"$or": [{"metadata.url": regex} for regex in include_regexes]})
        if exclude_regexes:
            clauses.append({"$nor": [{"metadata.url": regex} for regex in exclude_regexes]})
        return {"$and": clauses}

    def list_chunk_documents(
        self,
        *,
        company_id: str,
        scopes: Optional[Sequence[Dict[str, Any]]] = None,
        domains: Optional[Sequence[str]] = None,
        page: int = 1,
        limit: int = 20,
        query: str = "",
        state: str = "",
    ) -> Dict[str, Any]:
        """Group crawl chunks into virtual documents for the Space inventory."""
        safe_page = max(1, int(page or 1))
        safe_limit = min(5000, max(1, int(limit or 20)))
        clauses: List[Dict[str, Any]] = [
            self._company_query(company_id),
            {"metadata.source": {"$in": ["fetcher_page", "fetcher_initial"]}},
        ]

        scope_queries = [
            item for item in (self._chunk_scope_query(scope) for scope in (scopes or [])) if item
        ]
        if scope_queries:
            clauses.append({"$or": scope_queries})
        elif domains:
            normalized_domains = [str(value).strip().lower() for value in domains if str(value).strip()]
            if normalized_domains:
                clauses.append({"metadata.domain": {"$in": list(dict.fromkeys(normalized_domains))}})

        normalized_query = str(query or "").strip()[:200]
        if normalized_query:
            pattern = re.compile(re.escape(normalized_query), re.IGNORECASE)
            clauses.append({"$or": [{"metadata.url": pattern}, {"metadata.title": pattern}]})

        # Atomik replace eski chunk sürümlerini denetim/geri dönüş için saklar.
        # Envanter ise aramada kullanılan AKTİF sürümü göstermelidir; aksi halde
        # her re-index aynı sayfanın chunk sayısını katlayarak UI'ı şişirir.
        candidate_doc_ids = [
            str(value)
            for value in self.chunks.distinct("doc_id", {"$and": clauses})
            if value is not None
        ]
        if not candidate_doc_ids:
            return {
                "documents": [], "total": 0, "page": safe_page, "limit": safe_limit,
                "total_pages": 1, "domains": [],
            }
        stored_documents = {
            str(row.get("doc_id")): row
            for row in self.documents.find(
                {"doc_id": {"$in": candidate_doc_ids}},
                {"doc_id": 1, "active_ingest_version": 1, "status": 1,
                 "job_id": 1, "updated_at": 1},
            )
        }
        active_version_clauses: List[Dict[str, Any]] = []
        for doc_id in candidate_doc_ids:
            stored = stored_documents.get(doc_id)
            version = stored.get("active_ingest_version") if stored else None
            clause: Dict[str, Any] = {"doc_id": doc_id}
            if version:
                clause["ingest_version"] = version
            active_version_clauses.append(clause)
        clauses.append({"$or": active_version_clauses})

        requested_state = str(state or "").strip().lower()
        document_company_query = {
            "$or": [
                {"company_id": str(company_id)},
                {"metadata.companyId": str(company_id)},
            ]
        }
        status_query: Dict[str, Any] = {
            "$and": [{"doc_id": {"$exists": True}}, document_company_query]
        }
        state_aliases = {
            "disabled": ["disabled", "removed"],
            "error": ["error", "failed"],
            "indexing": ["indexing", "processing"],
            "queued": ["queued", "pending"],
            "validating": ["validating"],
            "not_indexed": ["not_indexed"],
        }
        if requested_state in state_aliases:
            status_query["$and"].append({"status": {"$in": state_aliases[requested_state]}})
            allowed_doc_ids = [str(row.get("doc_id")) for row in self.documents.find(status_query, {"doc_id": 1})]
            if not allowed_doc_ids:
                return {"documents": [], "total": 0, "page": safe_page, "limit": safe_limit, "total_pages": 1, "domains": []}
            clauses.append({"doc_id": {"$in": allowed_doc_ids}})
        elif requested_state and requested_state not in {"all", "indexed", "ready", "completed"}:
            return {"documents": [], "total": 0, "page": safe_page, "limit": safe_limit, "total_pages": 1, "domains": []}
        elif requested_state in {"indexed", "ready", "completed"}:
            disabled_doc_ids = [
                str(row.get("doc_id"))
                for row in self.documents.find(
                    {"$and": [
                        document_company_query,
                        {"status": {"$nin": ["ready", "indexed", "completed"]}},
                    ]},
                    {"doc_id": 1},
                )
            ]
            if disabled_doc_ids:
                clauses.append({"doc_id": {"$nin": disabled_doc_ids}})

        match = {"$and": clauses}
        group_stage = {
            "$group": {
                "_id": "$doc_id",
                "chunkCount": {"$sum": 1},
                "tokenCount": {"$sum": {"$ifNull": ["$token_count", {"$ifNull": ["$tokens", 0]}]}},
                "lastRunAt": {"$max": {"$ifNull": ["$updated_at", "$created_at"]}},
                "createdAt": {"$min": "$created_at"},
                "url": {"$first": "$metadata.url"},
                "title": {"$first": "$metadata.title"},
                "domain": {"$first": "$metadata.domain"},
                "source": {"$first": "$metadata.source"},
                "sourceSubscriptionId": {"$first": "$metadata.sourceSubscriptionId"},
                "contentHash": {"$first": {"$ifNull": ["$metadata.contentHash", "$metadata.content_hash"]}},
                "jobId": {"$first": {"$ifNull": ["$job_id", "$metadata.jobId"]}},
                "chunkSize": {"$first": "$metadata.chunkSize"},
                "chunkOverlap": {"$first": "$metadata.chunkOverlap"},
                "chunkMode": {"$first": "$metadata.chunkMode"},
                "chunkPolicy": {"$first": "$metadata.chunkPolicy"},
            }
        }
        skip = (safe_page - 1) * safe_limit
        pipeline = [
            {"$match": match},
            group_stage,
            {"$sort": {"lastRunAt": -1, "_id": 1}},
            {"$facet": {
                "documents": [{"$skip": skip}, {"$limit": safe_limit}],
                "total": [{"$count": "value"}],
                "domains": [{"$group": {"_id": "$domain"}}, {"$sort": {"_id": 1}}],
            }},
        ]
        result = next(iter(self.chunks.aggregate(pipeline, allowDiskUse=True)), {})
        rows = result.get("documents") or []
        doc_ids = [str(row.get("_id")) for row in rows]
        states = {
            doc_id: stored_documents[doc_id]
            for doc_id in doc_ids
            if doc_id in stored_documents
        }
        documents = []
        for row in rows:
            doc_id = str(row.pop("_id"))
            state_doc = states.get(doc_id) or {}
            row["documentId"] = doc_id
            row["state"] = str(state_doc.get("status") or "indexed")
            row["jobId"] = state_doc.get("job_id") or row.get("jobId")
            row["lastRunAt"] = state_doc.get("updated_at") or row.get("lastRunAt")
            documents.append(row)
        total = int(((result.get("total") or [{}])[0]).get("value") or 0)
        return {
            "documents": documents,
            "total": total,
            "page": safe_page,
            "limit": safe_limit,
            "total_pages": max(1, (total + safe_limit - 1) // safe_limit),
            "domains": [str(item.get("_id")) for item in (result.get("domains") or []) if item.get("_id")],
        }

    def get_chunk_document(self, doc_id: str, *, company_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return a parent-like summary even for legacy chunk-only documents."""
        clauses: List[Dict[str, Any]] = [{"doc_id": str(doc_id)}]
        if company_id:
            clauses.append(self._company_query(str(company_id)))
        pipeline = [
            {"$match": {"$and": clauses}},
            {"$group": {
                "_id": "$doc_id",
                "chunk_count": {"$sum": 1},
                "created_at": {"$min": "$created_at"},
                "updated_at": {"$max": {"$ifNull": ["$updated_at", "$created_at"]}},
                "metadata": {"$first": "$metadata"},
                "source": {"$first": "$metadata.source"},
                "company_id": {"$first": {"$ifNull": ["$company_id", "$metadata.companyId"]}},
                "user_id": {"$first": {"$ifNull": ["$user_id", "$metadata.userId"]}},
                "title": {"$first": "$metadata.title"},
            }},
        ]
        rows = list(self.chunks.aggregate(pipeline))
        if not rows:
            return None
        summary = rows[0]
        summary["doc_id"] = str(summary.pop("_id"))
        summary["status"] = "indexed"
        stored = self.get_document(str(doc_id))
        if stored:
            active_version = stored.get("active_ingest_version")
            summary.update({key: value for key, value in stored.items() if key != "_id"})
            if active_version:
                active_count = self.chunks.count_documents({
                    "doc_id": str(doc_id),
                    "ingest_version": active_version,
                })
                summary["chunk_count"] = int(active_count)
            else:
                summary["chunk_count"] = int(rows[0].get("chunk_count") or 0)
        return summary

    def delete_chunks_by_doc(self, doc_id: str, *, session=None) -> int:
        """Bir dokümanın TÜM chunk'larını siler — sürümden bağımsız.

        DİKKAT: versiyonlu swap'ta KULLANMA. Swap sırasında eski ve yeni sürüm
        chunk'ları aynı `doc_id` altında bir süre birlikte yaşar; bu metot ikisini
        birden siler ve yeni veriyi de yok eder. Swap için
        `delete_chunks_by_doc_version` kullan. Bu metot yalnız dokümanı tamamen
        kaldırırken (remove endpoint'i) doğrudur.
        """
        result = self.chunks.delete_many({"doc_id": doc_id}, session=session)
        return int(result.deleted_count)

    # ------------------------------------------------------------------
    # Sürüm-seçici işlemler — idempotent re-ingest'in "önce ekle, sonra eskiyi
    # kaldır" swap'ı için. Eski/yeni sürüm geçiş penceresinde birlikte var olur.
    # ------------------------------------------------------------------
    def get_chunks_by_doc_version(self, doc_id: str, version: str) -> List[Dict[str, Any]]:
        cursor = self.chunks.find({"doc_id": doc_id, "ingest_version": version}).sort(
            "chunk_index", ASCENDING
        )
        return list(cursor)

    def delete_chunks_by_doc_version(self, doc_id: str, version: str, *, session=None) -> int:
        """Yalnız belirtilen sürümün chunk'larını siler."""
        result = self.chunks.delete_many(
            {"doc_id": doc_id, "ingest_version": version}, session=session
        )
        return int(result.deleted_count)

    def delete_chunks_by_doc_except_version(
        self, doc_id: str, active_version: str, *, session=None
    ) -> int:
        """Aktif sürüm DIŞINDAKİ her şeyi siler.

        Sürümsüz (legacy) chunk'ları da temizler: `ingest_version` alanı olmayan
        kayıtlar `$ne` ile eşleşir. Bu, ilk versiyonlu ingest'in eski sürümsüz
        chunk'ları geride bırakmamasını sağlar.
        """
        result = self.chunks.delete_many(
            {"doc_id": doc_id, "ingest_version": {"$ne": active_version}}, session=session
        )
        return int(result.deleted_count)

    def get_faiss_ids_by_doc_except_version(self, doc_id: str, active_version: str) -> List[int]:
        """Aktif sürüm dışındaki chunk'ların faiss_id'leri (FAISS temizliği için)."""
        cursor = self.chunks.find(
            {"doc_id": doc_id, "ingest_version": {"$ne": active_version}},
            {"faiss_id": 1},
        )
        out: List[int] = []
        for row in cursor:
            fid = row.get("faiss_id")
            if isinstance(fid, (int, float)):
                out.append(int(fid))
        return out

    def set_active_ingest_version(
        self,
        doc_id: str,
        version: str,
        *,
        expected_job_id: Optional[str] = None,
        session=None,
    ) -> bool:
        """Dokümanın aktif sürümünü işaretler — arama yalnız bunu görür."""
        query: Dict[str, Any] = {"doc_id": doc_id}
        if expected_job_id is not None:
            query["job_id"] = str(expected_job_id)
        result = self.documents.update_one(
            query,
            {"$set": {"active_ingest_version": version, "updated_at": datetime.now(timezone.utc)}},
            upsert=False,
            session=session,
        )
        return int(result.matched_count or 0) > 0

    @staticmethod
    def _company_domain_query(company_id: str, domain: str) -> Dict[str, Any]:
        """
        (company_id, metadata.domain) eşleşmesi. Firma kimliği hem top-level
        `company_id` hem `metadata.companyId` altında olabildiği için ikisini de
        kabul eder — per-sayfa doc'larda o domain'e ait TÜM chunk'ları yakalar.
        """
        cid = str(company_id)
        return {
            "metadata.domain": str(domain),
            "$or": [{"company_id": cid}, {"metadata.companyId": cid}],
        }

    @staticmethod
    def _company_query(company_id: str) -> Dict[str, Any]:
        """
        Firma kimliği iki yerde yaşayabiliyor: ana ingest yolu üst seviye
        `company_id` yazar (ingest_worker `_chunk_and_embed`), pre-embed ve legacy
        yollar yalnızca `metadata.companyId` bırakır. Sayımlar ikisini de kapsamalı.
        """
        cid = str(company_id)
        return {"$or": [{"company_id": cid}, {"metadata.companyId": cid}]}

    def count_chunks_by_company(self, company_id: str) -> int:
        """Firmanın toplam chunk sayısı (sürüm ayrımı yapmadan)."""
        if not str(company_id or "").strip():
            return 0
        return int(self.chunks.count_documents(self._company_query(company_id)))

    def _active_versions_for_doc_ids(
        self, doc_ids: Sequence[Any]
    ) -> Dict[str, Optional[str]]:
        """Map searchable doc ids to their active version; keep parentless legacy ids."""
        normalized_ids = [str(doc_id) for doc_id in doc_ids if doc_id is not None]
        active_versions: Dict[str, Optional[str]] = {
            doc_id: None for doc_id in normalized_ids
        }
        if not normalized_ids:
            return active_versions
        for document in self.documents.find(
            {"doc_id": {"$in": normalized_ids}},
            {"doc_id": 1, "active_ingest_version": 1, "status": 1},
        ):
            doc_id = str(document.get("doc_id") or "")
            if document.get("status") in {"removed", "disabled"}:
                active_versions.pop(doc_id, None)
            else:
                active_versions[doc_id] = document.get("active_ingest_version")
        return active_versions

    def count_active_chunks_by_company(self, company_id: str) -> int:
        """
        Yalnızca AKTİF sürüme ait chunk sayısı.

        Bir chunk "aktif"tir ancak ve ancak `ingest_version` == dokümanın
        `active_ingest_version`'ı ise. Doküman kaydı ayrı bir VERİTABANINDA
        (`tinnten` vs `tinnten-embedding`) durduğu için `$lookup` kullanılamaz —
        iki adımda uygulama tarafında birleştiriyoruz.
        """
        cid = str(company_id or "").strip()
        if not cid:
            return 0

        doc_ids = self.chunks.distinct("doc_id", self._company_query(cid))
        if not doc_ids:
            return 0
        active_versions = self._active_versions_for_doc_ids(doc_ids)

        total = 0
        for doc_id, version in active_versions.items():
            query: Dict[str, Any] = {
                "$and": [self._company_query(cid), {"doc_id": doc_id}],
            }
            if version:
                query["ingest_version"] = version
            total += int(self.chunks.count_documents(query))
        return total

    def count_active_chunks_by_company_domain(self, company_id: str, domain: str) -> int:
        """Count only the current searchable chunk versions for one domain."""
        cid = str(company_id or "").strip()
        normalized_domain = str(domain or "").strip()
        if not cid or not normalized_domain:
            return 0

        base_query = self._company_domain_query(cid, normalized_domain)
        doc_ids = [
            str(value)
            for value in self.chunks.distinct("doc_id", base_query)
            if value is not None
        ]
        if not doc_ids:
            return 0
        active_versions = self._active_versions_for_doc_ids(doc_ids)

        total = 0
        for doc_id, active_version in active_versions.items():
            query: Dict[str, Any] = {
                "$and": [base_query, {"doc_id": doc_id}],
            }
            if active_version:
                query["ingest_version"] = active_version
            total += int(self.chunks.count_documents(query))
        return total

    def sample_active_chunks_by_company(self, company_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Return a small, bounded set of active chunks for retrieval probes.

        Health probes use real tenant text but never generate or persist new
        questions. Sampling one or more chunks across active documents keeps the
        audit cheap while exercising the same embedding/index path as production.
        """
        cid = str(company_id or "").strip()
        sample_limit = max(1, min(int(limit or 8), 24))
        if not cid:
            return []

        doc_ids = self.chunks.distinct("doc_id", self._company_query(cid))
        if not doc_ids:
            return []
        active_versions = self._active_versions_for_doc_ids(doc_ids)
        samples: List[Dict[str, Any]] = []
        for doc_id, version in active_versions.items():
            query: Dict[str, Any] = {
                "$and": [self._company_query(cid), {"doc_id": doc_id}],
                "text": {"$type": "string", "$ne": ""},
            }
            if version:
                query["ingest_version"] = version
            chunk = self.chunks.find_one(query, sort=[("chunk_index", ASCENDING)])
            if chunk:
                samples.append(chunk)
            if len(samples) >= sample_limit:
                break
        return samples

    def active_chunk_duplicate_stats_by_company(
        self, company_id: str, limit: int = 50000
    ) -> Dict[str, Any]:
        """Measure exact duplicate content over a bounded active-chunk sample.

        Documents and chunks live in separate databases, so active-version
        validation is performed while streaming one company-scoped cursor. The
        cap protects the health endpoint for unusually large tenants; the API
        explicitly reports whether the result is sampled.
        """
        cid = str(company_id or "").strip()
        scan_limit = max(100, min(int(limit or 50000), 200000))
        empty = {"scanned": 0, "unique": 0, "duplicates": 0, "ratio": None, "sampled": False}
        if not cid:
            return empty

        doc_ids = self.chunks.distinct("doc_id", self._company_query(cid))
        if not doc_ids:
            return empty
        active_versions = self._active_versions_for_doc_ids(doc_ids)

        fingerprints: set[str] = set()
        scanned = 0
        cursor = self.chunks.find(
            self._company_query(cid),
            {"doc_id": 1, "ingest_version": 1, "content_fingerprint": 1, "text": 1},
        ).batch_size(1000)
        for chunk in cursor:
            doc_id = str(chunk.get("doc_id") or "")
            if doc_id not in active_versions:
                continue
            active_version = active_versions[doc_id]
            if active_version and chunk.get("ingest_version") != active_version:
                continue
            fingerprint = str(chunk.get("content_fingerprint") or "")
            if not fingerprint:
                fingerprint = self.content_fingerprint(chunk.get("text") or "")
            if not fingerprint:
                continue
            fingerprints.add(fingerprint)
            scanned += 1
            if scanned >= scan_limit:
                break

        active_total = self.count_active_chunks_by_company(cid)
        duplicate_count = max(0, scanned - len(fingerprints))
        return {
            "scanned": scanned,
            "unique": len(fingerprints),
            "duplicates": duplicate_count,
            "ratio": round(duplicate_count / scanned, 6) if scanned else None,
            "sampled": active_total > scanned,
            "activeTotal": active_total,
        }

    def active_chunk_embedding_profile_by_company(
        self, company_id: str, limit: int = 50000
    ) -> Dict[str, Any]:
        """Summarise model/dimension metadata for active chunk versions.

        The FAISS file cannot reveal which model produced each vector. New
        ingests therefore persist this provenance beside every chunk. Legacy
        rows remain visible as ``unknown`` instead of being silently treated as
        the current model. The bounded scan keeps the health endpoint safe for
        large tenants and reports when the distribution is sampled.
        """
        cid = str(company_id or "").strip()
        scan_limit = max(100, min(int(limit or 50000), 200000))
        empty = {
            "scanned": 0,
            "sampled": False,
            "unknown": 0,
            "unknownRatio": None,
            "models": [],
            "dimensions": [],
        }
        if not cid:
            return empty

        doc_ids = self.chunks.distinct("doc_id", self._company_query(cid))
        if not doc_ids:
            return empty
        active_versions = self._active_versions_for_doc_ids(doc_ids)

        models: Dict[str, int] = {}
        dimensions: Dict[int, int] = {}
        unknown = 0
        scanned = 0
        cursor = self.chunks.find(
            self._company_query(cid),
            {"doc_id": 1, "ingest_version": 1, "embedding_model": 1, "embedding_dimension": 1},
        ).batch_size(1000)
        for chunk in cursor:
            doc_id = str(chunk.get("doc_id") or "")
            if doc_id not in active_versions:
                continue
            active_version = active_versions[doc_id]
            if active_version and chunk.get("ingest_version") != active_version:
                continue
            model = str(chunk.get("embedding_model") or "").strip()
            dimension = chunk.get("embedding_dimension")
            if model:
                models[model] = models.get(model, 0) + 1
            else:
                unknown += 1
            if isinstance(dimension, (int, float)) and int(dimension) > 0:
                dimensions[int(dimension)] = dimensions.get(int(dimension), 0) + 1
            scanned += 1
            if scanned >= scan_limit:
                break

        active_total = self.count_active_chunks_by_company(cid)
        return {
            "scanned": scanned,
            "sampled": active_total > scanned,
            "unknown": unknown,
            "unknownRatio": round(unknown / scanned, 6) if scanned else None,
            "models": [
                {"model": model, "count": count}
                for model, count in sorted(models.items(), key=lambda item: (-item[1], item[0]))
            ],
            "dimensions": [
                {"dimension": dimension, "count": count}
                for dimension, count in sorted(dimensions.items(), key=lambda item: (-item[1], item[0]))
            ],
        }

    def iter_active_faiss_ids_by_company(
        self, company_id: str, batch_size: int = 5000
    ):
        """
        Firmanın AKTİF chunk'larının faiss_id'lerini akış halinde verir.

        Yeniden inşanın kaynak listesidir; sürüm filtresi ŞARTTIR — yoksa
        `active_ingest_version` ile değiştirilmiş eski sürümler geri dirilir.
        Tüm listeyi belleğe almamak için doküman doküman ilerler.
        """
        cid = str(company_id or "").strip()
        if not cid:
            return

        doc_ids = self.chunks.distinct("doc_id", self._company_query(cid))
        if not doc_ids:
            return

        active_versions = self._active_versions_for_doc_ids(doc_ids)
        for doc_id, version in active_versions.items():
            query: Dict[str, Any] = {
                "$and": [self._company_query(cid), {"doc_id": doc_id}],
            }
            if version:
                query["ingest_version"] = version
            cursor = self.chunks.find(query, {"faiss_id": 1}).batch_size(batch_size)
            for chunk in cursor:
                fid = chunk.get("faiss_id")
                if isinstance(fid, (int, float)):
                    yield int(fid)

    def get_chunk_index_by_company_domain(
        self, company_id: str, domain: str
    ) -> Dict[str, Any]:
        """Bir firmanın bir domain'e ait chunk'larının faiss_id'leri + doc_id'leri."""
        cursor = self.chunks.find(
            self._company_domain_query(company_id, domain),
            {"faiss_id": 1, "doc_id": 1},
        )
        faiss_ids: set = set()
        doc_ids: set = set()
        for c in cursor:
            if isinstance(c.get("faiss_id"), (int, float)):
                faiss_ids.add(int(c["faiss_id"]))
            if c.get("doc_id"):
                doc_ids.add(str(c["doc_id"]))
        return {"faiss_ids": sorted(faiss_ids), "doc_ids": sorted(doc_ids)}

    def delete_chunks_by_company_domain(
        self, company_id: str, domain: str, *, session=None
    ) -> int:
        result = self.chunks.delete_many(
            self._company_domain_query(company_id, domain), session=session
        )
        return int(result.deleted_count)

    # ------------------------------------------------------------------
    # FAZ 5 — lexical (BM25-benzeri) arama; hybrid retrieval için dense'e eşlik eder
    # ------------------------------------------------------------------
    @staticmethod
    def _translate_chunk_filters(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Uygulama-seviyesi filtre sözlüğünü Mongo sorgusuna çevirir; `app.py`
        `_passes_chunk_filters` semantiğiyle BİREBİR aynıdır: "metadata.X" anahtarı
        iç içe alana, aksi halde top-level alana gider; `{"$in": [...]}` korunur,
        skaler değer eşitliğe çevrilir. Bu birebirlik önemli — lexical aday kümesi,
        dense yolunun `_passes_chunk_filters`'ından geçen kümeyle tutarlı kalmalı ki
        RRF füzyonu ÖNCESİ yeniden filtreleme hiçbir geçerli adayı düşürmesin.
        """
        query: Dict[str, Any] = {}
        and_clauses: List[Dict[str, Any]] = []
        for key, val in (filters or {}).items():
            # company_id iki yerde yaşayabiliyor (üst seviye VEYA metadata.companyId).
            # Dense yol (`_passes_chunk_filters`) ikisine de bakıyor; lexical dal
            # bakmazsa aday kümeleri ayrışır ve RRF füzyonu geçerli adayları düşürür.
            if key in ("company_id", "companyId", "companyid"):
                clause = {"$in": list(val["$in"])} if isinstance(val, dict) and "$in" in val else val
                and_clauses.append(
                    {"$or": [{"company_id": clause}, {"metadata.companyId": clause}]}
                )
                continue
            if isinstance(val, dict) and "$in" in val:
                query[key] = {"$in": list(val["$in"])}
            else:
                query[key] = val
        if and_clauses:
            query["$and"] = and_clauses
        return query

    def text_search_chunks(
        self,
        query_text: str,
        filters: Optional[Dict[str, Any]] = None,
        *,
        limit: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        `$text` sözcük araması + `_translate_chunk_filters` daraltması. textScore'a
        göre azalan sıralı chunk döner (en alakalı ilk). Boş sorguda [] döner.
        Not: mongomock `$text`'i desteklemez → birim testlerde bu yol atlanır,
        füzyon saf fonksiyon (`_rrf_fuse`) olarak test edilir (plan kararı #9).
        """
        if not query_text or not str(query_text).strip():
            return []
        mongo_query: Dict[str, Any] = {"$text": {"$search": str(query_text)}}
        mongo_query.update(self._translate_chunk_filters(filters))
        cursor = (
            self.chunks.find(mongo_query, {"score": {"$meta": "textScore"}})
            .sort([("score", {"$meta": "textScore"})])
            .limit(int(limit))
        )
        return list(cursor)

    # ------------------------------------------------------------------
    # Counter helpers
    # ------------------------------------------------------------------
    def _max_existing_faiss_id(self, *, session=None) -> int:
        row = self.chunks.find_one(
            {"faiss_id": {"$type": "number"}},
            {"faiss_id": 1},
            sort=[("faiss_id", -1)],
            session=session,
        )
        if not row:
            return 0
        try:
            return int(row.get("faiss_id") or 0)
        except (TypeError, ValueError):
            return 0

    def reserve_faiss_ids(self, count: int, *, session=None) -> List[int]:
        """
        Atomically reserve a block of FAISS IDs.
        """
        if count <= 0:
            raise ValueError("count must be positive")

        # Keep the counter aligned with existing chunk metadata so restarts or
        # dropped counter docs do not reuse an already persisted faiss_id.
        max_existing = self._max_existing_faiss_id(session=session)
        self.counters.update_one(
            {"_id": FAISS_COUNTER_KEY},
            {"$max": {"seq": max_existing}},
            upsert=True,
            session=session,
        )

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
