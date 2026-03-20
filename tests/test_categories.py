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


def test_list_categories_slug_filter_supports_text_fallback(client, mocker):
    mocked_items = [
        {
            "id": 1,
            "external_id": "cat_1",
            "text": "Home Appliances",
            "companyId": "comp_1",
            "metadata": {"slug": "home-appliances"},
        },
        {
            "id": 2,
            "external_id": "cat_2",
            "text": "Gaming Laptop",
            "companyId": "comp_1",
            "metadata": {},
        },
    ]
    mocker.patch("app.category_store.list_entries", return_value=(2, mocked_items))

    response = client.get("/api/v10/categories?slug=gaming-laptop")
    assert response.status_code == 200

    data = response.get_json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["categoryId"] == "cat_2"
    assert data["items"][0]["slug"] == "gaming-laptop"


def test_get_category_by_slug_identifier(client, mocker):
    slug_entry = {
        "id": 7,
        "external_id": "cat_7",
        "text": "Smart Phone",
        "companyId": "comp_1",
        "metadata": {"slug": "smart-phone"},
    }
    mocker.patch("app.category_store.find_ids_by_external_id", return_value=[])
    mocker.patch("app.category_store.list_entries", return_value=(1, [slug_entry]))
    mocker.patch("app.category_store.get_entry", return_value=slug_entry)
    mocker.patch("app.attribute_store.list_entries", return_value=(0, []))

    response = client.get("/api/v10/categories/smart-phone")
    assert response.status_code == 200
    data = response.get_json()
    assert data["categoryId"] == "cat_7"
    assert data["slug"] == "smart-phone"


def test_create_category_normalizes_localized_object_payload(client, mocker):
    created_entry = {
        "id": 11,
        "external_id": "cat_11",
        "text": "Ev ve Yasam",
        "companyId": None,
        "metadata": {"slug": "ev-ve-yasam"},
    }
    upsert_mock = mocker.patch(
        "app.category_store.upsert_vector",
        return_value={"id": 11, "external_id": "cat_11"},
    )
    mocker.patch("app.category_store.get_entry", return_value=created_entry)
    mocker.patch("app.category_store.list_entries", return_value=(0, []))

    response = client.post(
        "/api/v10/categories",
        json={
            "text": {"tr": "Ev ve Yasam"},
            "name": {"tr": "Ev ve Yasam"},
            "label": {"tr": "Ev ve Yasam"},
            "slug": {"tr": "Ev ve Yasam"},
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["text"] == "Ev ve Yasam"
    assert data["slug"] == "ev-ve-yasam"
    upsert_mock.assert_called_once()
    assert upsert_mock.call_args.args[0] == "Ev ve Yasam"
    assert upsert_mock.call_args.args[3]["label"] == "Ev ve Yasam"
    assert upsert_mock.call_args.args[3]["slug"] == "ev-ve-yasam"


def test_get_category_normalizes_placeholder_fields_to_safe_fallback(client, mocker):
    broken_entry = {
        "id": 6,
        "external_id": "5f9454e7-f655-4388-a011-1a9ff984360e",
        "text": "[object Object]",
        "companyId": None,
        "metadata": {"slug": "object-object", "parentId": None},
    }
    mocker.patch("app.category_store.find_ids_by_external_id", return_value=[6])
    mocker.patch("app.category_store.get_entry", return_value=broken_entry)
    mocker.patch("app.attribute_store.list_entries", return_value=(0, []))

    response = client.get("/api/v10/categories/5f9454e7-f655-4388-a011-1a9ff984360e")

    assert response.status_code == 200
    data = response.get_json()
    assert data["categoryId"] == "5f9454e7-f655-4388-a011-1a9ff984360e"
    assert data["text"] == "5f9454e7-f655-4388-a011-1a9ff984360e"
    assert data["slug"] == "5f9454e7-f655-4388-a011-1a9ff984360e"
