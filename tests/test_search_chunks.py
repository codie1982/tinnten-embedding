import pytest
import numpy as np

def test_chunk_search_success(client, mocker):
    # Mock chunk_engine.encode
    mock_vec = np.zeros((1, 768), dtype=np.float32)
    mocker.patch("app.chunk_engine.encode", return_value=mock_vec)
    
    # Mock chunk_engine.search
    mocker.patch("app.chunk_engine.search", return_value=(
        np.array([[0.1, 0.2]]), # scores
        np.array([[10, 20]])   # ids
    ))
    
    # Mock chunk_store.get_chunks_by_faiss_ids
    mocker.patch("app.chunk_store.get_chunks_by_faiss_ids", return_value={
        10: {"doc_id": "doc1", "text": "chunk 10", "chunk_id": "c10", "metadata": {}},
        20: {"doc_id": "doc2", "text": "chunk 20", "chunk_id": "c20", "metadata": {}}
    })
    
    # Mock chunk_store.get_documents_by_ids
    mocker.patch("app.chunk_store.get_documents_by_ids", return_value={
        "doc1": {"status": "active"},
        "doc2": {"status": "active"}
    })

    payload = {
        "text": "test query",
        "type": "chunk",
        "k": 2
    }
    response = client.post("/api/v10/vector/search", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["results"]) == 2
    assert data["results"][0]["id"] == 10
    assert data["results"][1]["id"] == 20

def test_chunk_search_missing_text(client):
    response = client.post("/api/v10/vector/search", json={"type": "chunk"})
    assert response.status_code == 400
    assert "Provide either 'text' or 'vector'" in response.get_json()["error"]
