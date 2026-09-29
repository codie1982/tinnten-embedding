import pytest

from services.inbound_service_auth import (
    ServiceAuthConfigError,
    ServiceCallerForbidden,
    authorize_service_claims,
    validate_quota_runtime_config,
)


def secure_env(**overrides):
    env = {
        "INDEX_QUOTA_ENABLED": "1",
        "QUOTA_FAIL_OPEN": "0",
        "TINNTEN_SERVER_BASE_URL": "https://server.internal",
        "KEYCLOAK_BASE_URL": "https://identity.internal",
        "REALM": "tinnten",
        "SERVICES_CLIENT_ID": "embedding-outbound",
        "SERVICES_CLIENT_SECRET": "secret",
        "REQUIRE_KEYCLOAK_AUTH": "true",
        "ALLOW_UNAUTH_CONTENT_INDEX": "false",
        "INBOUND_SERVICE_CLIENT_IDS": "tinnten-server,tinnten-fetcher",
    }
    env.update(overrides)
    return env


def test_index_quota_api_rejects_unauthenticated_or_bypass_configuration():
    with pytest.raises(ServiceAuthConfigError) as caught:
        validate_quota_runtime_config(
            secure_env(
                REQUIRE_KEYCLOAK_AUTH="false",
                ALLOW_UNAUTH_CONTENT_INDEX="true",
                INBOUND_SERVICE_CLIENT_IDS="",
            ),
            api_process=True,
        )
    message = str(caught.value)
    assert "REQUIRE_KEYCLOAK_AUTH" in message
    assert "ALLOW_UNAUTH_CONTENT_INDEX" in message
    assert "INBOUND_SERVICE_CLIENT_IDS" in message


def test_index_quota_worker_rejects_fail_open_or_missing_service_credentials():
    with pytest.raises(ServiceAuthConfigError) as caught:
        validate_quota_runtime_config(
            secure_env(QUOTA_FAIL_OPEN="true", TINNTEN_SERVER_BASE_URL=""),
            api_process=False,
        )
    assert "QUOTA_FAIL_OPEN" in str(caught.value)
    assert "TINNTEN_SERVER_BASE_URL" in str(caught.value)


def test_service_claims_bind_active_token_to_allowed_caller_audience_and_scope():
    env = secure_env(
        INBOUND_SERVICE_AUDIENCE="embedding-api",
        INBOUND_SERVICE_REQUIRED_SCOPE="embedding.write",
    )
    assert (
        authorize_service_claims(
            {
                "active": True,
                "client_id": "tinnten-fetcher",
                "aud": "embedding-api",
                "scope": "openid embedding.write",
            },
            env,
        )
        == "tinnten-fetcher"
    )
    with pytest.raises(ServiceCallerForbidden):
        authorize_service_claims(
            {"active": True, "azp": "browser", "aud": "embedding-api"},
            env,
        )
