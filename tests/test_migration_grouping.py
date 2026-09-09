"""
Adım 3 — migration script'inin SAF mantığı (faiss/mongo bağımsız).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from migrate_global_to_per_company_faiss import (  # noqa: E402
    resolve_company_id,
    sanitize_company_id,
    group_faiss_ids_by_company,
    company_index_path,
    filter_migratable_chunks,
)


def test_resolve_company_id_precedence():
    # metadata.companyId önce (per-sayfa chunk)
    assert resolve_company_id({"metadata": {"companyId": "c1"}, "company_id": None}) == "c1"
    # top-level company_id fallback (legacy)
    assert resolve_company_id({"metadata": {}, "company_id": "c2"}) == "c2"
    # firma-sız (personal)
    assert resolve_company_id({"metadata": {}, "company_id": None}) is None
    assert resolve_company_id({}) is None


def test_sanitize_company_id_matches_app_regex():
    assert sanitize_company_id("6a2b1893be2ff2f3a426c218") == "6a2b1893be2ff2f3a426c218"
    assert sanitize_company_id("../../etc/passwd") == "etcpasswd"  # path traversal savunması
    assert sanitize_company_id("  ab_CD-12  ") == "ab_CD-12"


def test_group_faiss_ids_by_company():
    chunks = [
        {"faiss_id": 1, "metadata": {"companyId": "c1"}},
        {"faiss_id": 2, "metadata": {"companyId": "c1"}},
        {"faiss_id": 3, "company_id": "c2", "metadata": {}},
        {"faiss_id": 4, "metadata": {}, "company_id": None},   # personal → global kalır
        {"metadata": {"companyId": "c1"}},                     # faiss_id yok → atla
    ]
    groups, company_less, no_faiss = group_faiss_ids_by_company(chunks)
    assert sorted(groups["c1"]) == [1, 2]
    assert groups["c2"] == [3]
    assert company_less == 1
    assert no_faiss == 1


def test_company_index_path():
    p = company_index_path("/app/data/faiss/faiss.index", "6a2b1893")
    assert p == "/app/data/faiss/company/6a2b1893.index"
    assert company_index_path("/app/data/faiss/faiss.index", "  ") is None


def test_filter_migratable_chunks_requires_live_matching_parent_and_active_version():
    chunks = [
        {"doc_id": "valid", "faiss_id": 1, "company_id": "c1", "ingest_version": "v2"},
        {"doc_id": "missing", "faiss_id": 2, "company_id": "c1"},
        {"doc_id": "removed", "faiss_id": 3, "company_id": "c1"},
        {"doc_id": "no-company", "faiss_id": 4, "company_id": "c1"},
        {"doc_id": "mismatch", "faiss_id": 5, "company_id": "c1"},
        {"doc_id": "stale", "faiss_id": 6, "company_id": "c1", "ingest_version": "v1"},
    ]
    documents = [
        {"doc_id": "valid", "status": "ready", "company_id": "c1", "active_ingest_version": "v2"},
        {"doc_id": "removed", "status": "removed", "company_id": "c1"},
        {"doc_id": "no-company", "status": "ready"},
        {"doc_id": "mismatch", "status": "ready", "company_id": "c2"},
        {"doc_id": "stale", "status": "ready", "company_id": "c1", "active_ingest_version": "v2"},
    ]

    eligible, stats = filter_migratable_chunks(chunks, documents)

    assert [chunk["faiss_id"] for chunk in eligible] == [1]
    assert stats == {
        "parentless": 1,
        "inactive_parent": 1,
        "parent_company_missing": 1,
        "company_mismatch": 1,
        "stale_version": 1,
    }


def test_filter_migratable_chunks_allows_legacy_version_when_parent_has_no_active_version():
    chunks = [{"doc_id": "legacy", "faiss_id": 7, "metadata": {"companyId": "c1"}}]
    documents = [{"doc_id": "legacy", "status": "ready", "metadata": {"companyId": "c1"}}]

    eligible, stats = filter_migratable_chunks(chunks, documents)

    assert [chunk["faiss_id"] for chunk in eligible] == [7]
    assert sum(stats.values()) == 0
