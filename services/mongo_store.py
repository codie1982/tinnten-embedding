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
            "created_at": now,
        }
        update_fields = {
            "doc_id": doc_id,
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
        for key, val in (filters or {}).items():
            if isinstance(val, dict) and "$in" in val:
                query[key] = {"$in": list(val["$in"])}
            else:
                query[key] = val
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
