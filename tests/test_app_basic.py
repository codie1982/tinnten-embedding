def test_root_health_check(client, mocker):
    # Mock the response from store.index.ntotal and chunk_engine
    mocker.patch("app.store.index.ntotal", 123)
    mocker.patch("app.chunk_engine.count", return_value=456)
    mocker.patch("app.chunk_engine.index_dimension", return_value=1536)
    mocker.patch("app.chunk_engine.model_dimension", return_value=1536)

    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert data["message"] == "tinnten-embedding up"
    assert data["index_size"] == 123
    assert data["chunk_index_size"] == 456
