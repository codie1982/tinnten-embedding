"""
FAZ 0 dayanağı — top-level `text` ile /content/index çağrısı, içeriği TAM ve
`source={type:"text"}` olarak saklamalı (URL fetch YOK, kırpma YOK).

Bu, modailgi 1069-char bug'ının (payload'da url → ingest URL'i fetch edip inline
metni yok sayıyordu) tekrarını önleyen kritik sözleşmedir. Fetcher'ın
trigger_fetcher_page_inline'ı tam bu şekli gönderir.
"""


def test_top_level_text_becomes_text_source_untruncated(client, mocker):
    import app

    big_text = "# Başlık\n\n" + ("İçerik satırı. " * 5000)  # ~70k char
    upsert = mocker.patch.object(
        app.content_store, "upsert_document_with_source", return_value={"doc_id": "ci-doc"}
    )
    mocker.patch.object(app.content_store, "append_log_entry", return_value=None)
    mocker.patch("app._publish_content_index_message", return_value=None)

    resp = client.post(
        "/api/v10/content/index",
        json={
            "companyId": "C1",
            "documentId": "b55eaea10f281ccd07647aa5",
            "text": big_text,
            "metadata": {"domain": "example.com", "url": "https://example.com/p1",
                         "source": "fetcher_page"},
            "trigger": "fetcher_page",
        },
    )

    assert 200 <= resp.status_code < 300
    # source = {type: "text", text: <TAM metin>} — URL değil, kırpılmamış
    source = upsert.call_args.kwargs["source"]
    assert source["type"] == "text"
    assert source["text"] == big_text  # bayt-tam, kırpma yok
    assert "url" not in source  # url source'a girmemeli (yoksa embedding fetch eder)
    # metadata.domain korunur (search domain-filtresi için)
    md = upsert.call_args.kwargs["metadata"]
    assert md.get("domain") == "example.com"


def test_url_only_payload_does_not_become_text_source(client, mocker):
    """Karşıt kontrol: text yoksa ve url varsa source url olur (fetch davranışı).
    Fetcher inline yolu bu tuzağa DÜŞMEZ çünkü daima top-level text yollar."""
    import app

    upsert = mocker.patch.object(
        app.content_store, "upsert_document_with_source", return_value={"doc_id": "d"}
    )
    mocker.patch.object(app.content_store, "append_log_entry", return_value=None)
    mocker.patch("app._publish_content_index_message", return_value=None)

    resp = client.post(
        "/api/v10/content/index",
        json={"companyId": "C1", "documentId": "d", "url": "https://example.com/p1"},
    )
    assert 200 <= resp.status_code < 300
    assert upsert.call_args.kwargs["source"]["type"] in {"url", "import_url", "web"}
