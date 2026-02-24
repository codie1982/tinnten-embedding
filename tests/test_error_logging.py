def test_content_index_prepare_logs_error(client, mocker):
    mocker.patch("app.content_store.upsert_document_with_source", side_effect=RuntimeError("db down"))
    append_mock = mocker.patch("app.embedding_error_logger.append")

    response = client.post("/api/v10/content/index", json={"companyId": "comp-1", "text": "hello world"})
    assert response.status_code == 500
    data = response.get_json()
    assert "failed to prepare document" in data["error"]

    append_mock.assert_called()
    events = [call.args[0].get("event") for call in append_mock.call_args_list]
    assert "content_index_prepare_failed" in events
