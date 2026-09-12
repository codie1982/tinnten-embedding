from datetime import datetime, timezone

import mongomock

from services.mongo_store import MongoStore


def _store():
    client = mongomock.MongoClient()
    store = MongoStore.__new__(MongoStore)
    store.documents = client["tinnten"]["embedding_documents"]
    store.chunks = client["tinnten-embedding"]["embedding_chunks"]
    return store


def test_chunk_inventory_groups_crawl_pages_without_parent_documents():
    store = _store()
    now = datetime.now(timezone.utc)
    base = {
        "company_id": "company-1",
        "created_at": now,
        "metadata": {
            "companyId": "company-1",
            "domain": "tinten.ai",
            "source": "fetcher_page",
            "url": "https://tinten.ai/tr/document/araclar",
            "title": "Araçlar",
        },
    }
    store.chunks.insert_many([
        {**base, "doc_id": "crawl-doc-1", "chunk_index": 0, "token_count": 12, "text": "ilk"},
        {**base, "doc_id": "crawl-doc-1", "chunk_index": 1, "token_count": 9, "text": "ikinci"},
        {
            **base,
            "doc_id": "outside-pattern",
            "metadata": {**base["metadata"], "url": "https://tinten.ai/tr/pricing"},
            "chunk_index": 0,
            "token_count": 5,
            "text": "hariç",
        },
        {
            **base,
            "company_id": "company-2",
            "doc_id": "other-company",
            "metadata": {**base["metadata"], "companyId": "company-2"},
            "chunk_index": 0,
            "text": "başka",
        },
    ])

    result = store.list_chunk_documents(
        company_id="company-1",
        scopes=[{
            "domains": ["tinten.ai", "www.tinten.ai"],
            "includePatterns": ["/tr/document/*"],
            "excludePatterns": [],
        }],
    )

    assert result["total"] == 1
    assert result["domains"] == ["tinten.ai"]
    assert result["documents"][0]["documentId"] == "crawl-doc-1"
    assert result["documents"][0]["chunkCount"] == 2
    assert result["documents"][0]["tokenCount"] == 21
    assert result["documents"][0]["state"] == "indexed"


def test_chunk_document_detail_falls_back_to_chunk_metadata_and_is_company_scoped():
    store = _store()
    store.chunks.insert_one({
        "company_id": "company-1",
        "doc_id": "legacy-crawl-doc",
        "chunk_index": 0,
        "text": "İçerik",
        "created_at": datetime.now(timezone.utc),
        "metadata": {
            "companyId": "company-1",
            "domain": "tinten.ai",
            "source": "fetcher_page",
            "url": "https://tinten.ai/tr/document/workflow",
            "title": "Workflow",
        },
    })

    detail = store.get_chunk_document("legacy-crawl-doc", company_id="company-1")

    assert detail["doc_id"] == "legacy-crawl-doc"
    assert detail["chunk_count"] == 1
    assert detail["status"] == "indexed"
    assert detail["metadata"]["url"].endswith("/workflow")
    assert store.get_chunk_document("legacy-crawl-doc", company_id="company-2") is None


def test_chunk_inventory_uses_embedding_document_only_for_lifecycle_state():
    store = _store()
    now = datetime.now(timezone.utc)
    store.chunks.insert_one({
        "company_id": "company-1",
        "doc_id": "disabled-crawl-doc",
        "chunk_index": 0,
        "text": "Pasif içerik",
        "created_at": now,
        "metadata": {
            "companyId": "company-1",
            "domain": "tinten.ai",
            "source": "fetcher_page",
            "url": "https://tinten.ai/tr/document/pasif",
        },
    })
    store.documents.insert_one({
        "doc_id": "disabled-crawl-doc",
        "company_id": "company-1",
        "status": "disabled",
        "updated_at": now,
    })

    all_rows = store.list_chunk_documents(company_id="company-1", domains=["tinten.ai"], state="all")
    indexed_rows = store.list_chunk_documents(company_id="company-1", domains=["tinten.ai"], state="indexed")
    disabled_rows = store.list_chunk_documents(company_id="company-1", domains=["tinten.ai"], state="disabled")

    assert all_rows["documents"][0]["state"] == "disabled"
    assert indexed_rows["total"] == 0
    assert disabled_rows["total"] == 1


def test_chunk_inventory_and_detail_only_count_active_ingest_version():
    store = _store()
    now = datetime.now(timezone.utc)
    metadata = {
        "companyId": "company-1",
        "domain": "tinten.ai",
        "source": "fetcher_page",
        "url": "https://tinten.ai/tr/document/araclar",
        "title": "Araçlar",
    }
    store.documents.insert_one({
        "doc_id": "versioned-crawl-doc",
        "company_id": "company-1",
        "status": "ready",
        "active_ingest_version": "v2",
        "updated_at": now,
    })
    store.chunks.insert_many([
        {
            "company_id": "company-1", "doc_id": "versioned-crawl-doc",
            "ingest_version": "v1", "chunk_index": index, "text": f"eski-{index}",
            "created_at": now, "metadata": metadata,
        }
        for index in range(4)
    ] + [
        {
            "company_id": "company-1", "doc_id": "versioned-crawl-doc",
            "ingest_version": "v2", "chunk_index": index, "text": f"yeni-{index}",
            "created_at": now, "metadata": metadata,
        }
        for index in range(2)
    ])

    inventory = store.list_chunk_documents(company_id="company-1", domains=["tinten.ai"])
    detail = store.get_chunk_document("versioned-crawl-doc", company_id="company-1")
    page = store.get_chunks_page_by_doc(
        "versioned-crawl-doc", ingest_version=detail["active_ingest_version"]
    )

    assert inventory["documents"][0]["chunkCount"] == 2
    assert detail["chunk_count"] == 2
    assert page["document_total"] == 2
    assert [row["text"] for row in page["chunks"]] == ["yeni-0", "yeni-1"]
    assert store.count_active_chunks_by_company_domain("company-1", "tinten.ai") == 2


def test_company_active_count_and_rebuild_ids_keep_parentless_legacy_chunks():
    store = _store()
    now = datetime.now(timezone.utc)
    base = {
        "company_id": "company-1",
        "created_at": now,
        "metadata": {"companyId": "company-1", "domain": "tinten.ai"},
    }
    store.documents.insert_many([
        {
            "doc_id": "versioned", "company_id": "company-1", "status": "ready",
            "active_ingest_version": "v2",
        },
        {"doc_id": "disabled", "company_id": "company-1", "status": "disabled"},
    ])
    store.chunks.insert_many([
        {**base, "doc_id": "versioned", "ingest_version": "v1", "faiss_id": 1},
        {**base, "doc_id": "versioned", "ingest_version": "v2", "faiss_id": 2},
        {**base, "doc_id": "legacy-parentless", "faiss_id": 3},
        {**base, "doc_id": "disabled", "faiss_id": 4},
    ])

    assert store.count_active_chunks_by_company("company-1") == 2
    assert set(store.iter_active_faiss_ids_by_company("company-1")) == {2, 3}
