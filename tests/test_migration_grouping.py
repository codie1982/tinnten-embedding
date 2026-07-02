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
