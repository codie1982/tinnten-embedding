import pytest
from services.keycloak_service import KeycloakTokenError

def test_auth_unauthorized(client, mocker):
    mocker.patch("app.REQUIRE_KEYCLOAK_AUTH", True)
    mocker.patch("app.get_keycloak_service").return_value.validate_bearer_header.side_effect = KeycloakTokenError("Invalid token")
    
    response = client.get("/")
    assert response.status_code == 401
    assert "unauthorized" in response.get_json()["error"]

def test_auth_success(client, mocker):
    mocker.patch("app.REQUIRE_KEYCLOAK_AUTH", True)
    mocker.patch("app.get_keycloak_service").return_value.validate_bearer_header.return_value = {"sub": "user_123"}
    
    # Mock store as well to avoid failure in root()
    mocker.patch("app.store.index.ntotal", 0)
    mocker.patch("app.chunk_engine.count", return_value=0)
    mocker.patch("app.chunk_engine.index_dimension", return_value=768)
    mocker.patch("app.chunk_engine.model_dimension", return_value=768)

    response = client.get("/", headers={"Authorization": "Bearer valid_token"})
    assert response.status_code == 200
    assert response.get_json()["message"] == "tinnten-embedding up"
