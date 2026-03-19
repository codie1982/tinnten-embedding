def test_root_health_check(client, mocker):
    mocker.patch("app.store.index.ntotal", 123)
    mock_engine = mocker.Mock()
    mock_engine.count.return_value = 456
    mock_engine.index_dimension.return_value = 1536
    mock_engine.model_dimension.return_value = 1536
    mocker.patch("app._get_cached_chunk_engine", return_value=mock_engine)

    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert data["message"] == "tinnten-embedding up"
    assert data["index_size"] == 123
    assert data["chunk_index_size"] == 456
    assert data["chunk_engine_loaded"] is True


def test_root_health_check_does_not_warm_chunk_engine(client, mocker):
    mocker.patch("app.store.index.ntotal", 123)
    cached_engine = mocker.patch("app._get_cached_chunk_engine", return_value=None)
    get_chunk_engine = mocker.patch("app.get_chunk_engine")

    response = client.get("/")

    assert response.status_code == 200
    data = response.get_json()
    assert data["chunk_index_size"] is None
    assert data["chunk_engine_loaded"] is False
    cached_engine.assert_called_once()
    get_chunk_engine.assert_not_called()
