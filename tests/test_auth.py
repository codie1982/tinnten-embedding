from services.keycloak_service import KeycloakTokenError


def test_health_endpoint_skips_auth(client, mocker):
    mocker.patch("app.REQUIRE_KEYCLOAK_AUTH", True)
    mocker.patch("app.store.index.ntotal", 0)
    validate_mock = mocker.patch("app.get_keycloak_service").return_value.validate_bearer_header
    mocker.patch("app._get_cached_chunk_engine", return_value=None)

    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json()["message"] == "tinnten-embedding up"
    validate_mock.assert_not_called()


def test_protected_endpoint_requires_auth(client, mocker):
    mocker.patch("app.REQUIRE_KEYCLOAK_AUTH", True)
    mocker.patch("app.get_keycloak_service").return_value.validate_bearer_header.side_effect = KeycloakTokenError("Invalid token")

    response = client.post("/api/v10/llm/vector", json={"text": "hello"})
    assert response.status_code == 401
    assert response.get_json()["error"] == "unauthorized"


def test_auth_success(client, mocker):
    mocker.patch("app.REQUIRE_KEYCLOAK_AUTH", True)
    mocker.patch("app.get_keycloak_service").return_value.validate_bearer_header.return_value = {"sub": "user_123"}
    mocker.patch("app.store.vectorize_text", return_value=[0.1, 0.2])

    response = client.post("/api/v10/llm/vector", json={"text": "hello"}, headers={"Authorization": "Bearer valid_token"})
    assert response.status_code == 200
    assert response.get_json()["vector"] == [0.1, 0.2]
