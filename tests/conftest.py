import pytest
from unittest.mock import MagicMock, patch
import mongomock
import os

@pytest.fixture(autouse=True)
def mock_env():
    with patch.dict(os.environ, {
        "MONGO_URI": "mongodb://localhost:27017/testdb",
        "DB_TINNTEN": "testdb",
        "REQUIRE_KEYCLOAK_AUTH": "false",
        "ALLOW_UNAUTH_CONTENT_INDEX": "true"
    }):
        yield

@pytest.fixture
def mock_mongo():
    with patch("init.db.get_mongo_client") as mock_client:
        client = mongomock.MongoClient()
        mock_client.return_value = client
        yield client

@pytest.fixture
def app_with_mocks(mocker):
    # Mock global objects in app.py before importing it
    mocker.patch("vector_store.EmbeddingIndex")
    mocker.patch("vector_store.MetaRepository")
    mocker.patch("services.UploadStore")
    mocker.patch("services.DocumentLoader")
    mocker.patch("services.ContentDocumentStore")
    mocker.patch("services.rabbit_publisher.RabbitPublisher")
    mocker.patch("services.embedding_engine.EmbeddingEngine")
    mocker.patch("services.mongo_store.MongoStore")
    mocker.patch("services.keycloak_service.get_keycloak_service")
    
    import app
    # Refresh the app object if needed or just return it
    return app.app

@pytest.fixture
def client(app_with_mocks):
    return app_with_mocks.test_client()
