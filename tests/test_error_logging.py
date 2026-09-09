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


def test_upload_prepare_failure_marks_upload_index_failed(client, mocker):
    import app

    mocker.patch.object(
        app.content_store,
        "upsert_document_with_source",
        side_effect=RuntimeError("mongo down"),
    )
    update_upload = mocker.patch.object(app.upload_store, "update_upload_status")

    response = client.post(
        "/api/v10/content/index",
        json={
            "companyId": "C1",
            "uploadId": "UP1",
            "trigger": "upload_scan_clean",
        },
    )

    assert response.status_code == 500
    update_upload.assert_called_once_with(
        "UP1",
        index_status="failed",
        is_file_opened=False,
        file_open_error="failed to prepare document: mongo down",
    )
