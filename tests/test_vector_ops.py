import pytest
from unittest.mock import MagicMock

def test_generate_vector(client, mocker):
    mock_vec = [0.1, 0.2, 0.3]
    mocker.patch("app.store.vectorize_text", return_value=mock_vec)

    response = client.post("/api/v10/llm/vector", json={"text": "hello world"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["vector"] == mock_vec

def test_upsert_vector(client, mocker):
    mock_response = {"id": 1, "status": "success"}
    mocker.patch("app.store.upsert_vector", return_value=mock_response)

    payload = {
        "text": "test text",
        "external_id": "ext123",
        "metadata": {"key": "value"}
    }
    response = client.post("/api/v10/vector/upsert", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["id"] == 1
    assert data["status"] == "success"

def test_search_vector_general(client, mocker):
    mock_results = [{"id": 1, "score": 0.9, "text": "result 1"}]
    mocker.patch("app.store.search", return_value=mock_results)

    payload = {
        "text": "search query",
        "type": "vector",
        "k": 1
    }
    response = client.post("/api/v10/vector/search", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["results"]) == 1
    assert data["results"][0]["text"] == "result 1"
