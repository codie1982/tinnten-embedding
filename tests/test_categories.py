import pytest

def test_search_category(client, mocker):
    mock_results = [{
        "id": 101,
        "score": 0.95,
        "text": "Electronics",
        "external_id": "cat_123",
        "companyId": "comp_1",
        "metadata": {"description": "Gadgets"}
    }]
    mocker.patch("app.category_store.search", return_value=mock_results)
    
    # Mocking for _map_category_entry
    mocker.patch("app.category_store.find_ids_by_external_id", return_value=[101])
    mocker.patch("app.category_store.get_entry", return_value=mock_results[0])
    
    payload = {
        "text": "phone",
        "type": "category",
        "k": 1
    }
    response = client.post("/api/v10/vector/search", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["results"]) == 1
    assert data["results"][0]["categoryId"] == "cat_123"
    assert data["results"][0]["text"] == "Electronics"

def test_search_attribute(client, mocker):
    mock_results = [{
        "id": 1001,
        "score": 0.88,
        "text": "Color: Red",
        "external_id": "attr_456",
        "companyId": "comp_1",
        "metadata": {"categoryId": "cat_123"}
    }]
    mocker.patch("app.attribute_store.search", return_value=mock_results)
    
    payload = {
        "text": "red color",
        "type": "attribute",
        "k": 1
    }
    response = client.post("/api/v10/vector/search", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["results"]) == 1
    assert data["results"][0]["attributeId"] == "attr_456"
    assert data["results"][0]["text"] == "Color: Red"
