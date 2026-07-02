"""
FAZ 2c — index-state callback body sözleşmesi.
Per-sayfa (fetcher_page/initial) doc'larda callback body'si `metadata.domain` +
`stats.domainChunks` taşımalı ki server website ENTRY'sini domain ile bulup
badge'i güncellesin. Diğer source'larda bu alanlar OLMAMALI.
"""
from unittest.mock import MagicMock, patch

from services.tinnten_server_client import TinntenServerClient


def _client():
    with patch("services.tinnten_server_client.get_keycloak_service") as kc:
        kc.return_value = MagicMock(get_service_token=lambda: "tok")
        c = TinntenServerClient()
        c.base_url = "http://server:5001"
        return c


def _capture_patch_body(client, **kwargs):
    with patch("services.tinnten_server_client.requests.patch") as pat:
        pat.return_value = MagicMock(status_code=200)
        client.update_document_index_state("b55eaea10f281ccd07647aa5", "completed", **kwargs)
        return pat.call_args.kwargs["json"]


def test_fetcher_page_callback_carries_domain_and_domain_chunks():
    body = _capture_patch_body(
        _client(),
        company_id="C1",
        domain="example.com",
        source="fetcher_page",
        domain_chunks=42,
    )
    assert body["state"] == "indexed"  # completed → indexed
    assert body["metadata"] == {"domain": "example.com", "source": "fetcher_page"}
    assert body["stats"]["domainChunks"] == 42
    assert body["companyid"] == "C1"


def test_non_fetcher_source_omits_domain_metadata():
    body = _capture_patch_body(
        _client(),
        company_id="C1",
        stats={"chunkCount": 5},
        # domain/source verilmedi (ör. upload/library)
    )
    assert "metadata" not in body
    assert "domainChunks" not in body.get("stats", {})


def test_domain_without_fetcher_source_is_ignored():
    # domain var ama source fetcher_page değil → metadata eklenmez (güvenli)
    body = _capture_patch_body(
        _client(), company_id="C1", domain="example.com", source="upload", domain_chunks=9
    )
    assert "metadata" not in body
