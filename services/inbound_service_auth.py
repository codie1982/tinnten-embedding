"""Fail-closed inbound service-token policy for tenant-scoped index APIs."""

from __future__ import annotations

import os
from collections.abc import Mapping


TRUE_VALUES = {"1", "true", "yes", "on"}


class ServiceAuthConfigError(RuntimeError):
    """A quota-enabled process is missing a mandatory security setting."""


class ServiceCallerForbidden(RuntimeError):
    """The token is active but does not belong to an allowed service client."""


def env_enabled(name: str, env: Mapping[str, str] | None = None) -> bool:
    source = env if env is not None else os.environ
    return str(source.get(name) or "").strip().lower() in TRUE_VALUES


def inbound_client_ids(env: Mapping[str, str] | None = None) -> set[str]:
    source = env if env is not None else os.environ
    return {
        item.strip()
        for item in str(source.get("INBOUND_SERVICE_CLIENT_IDS") or "").split(",")
        if item.strip()
    }


def service_caller_required(env: Mapping[str, str] | None = None) -> bool:
    return env_enabled("INBOUND_SERVICE_AUTH_REQUIRED", env) or env_enabled(
        "INDEX_QUOTA_ENABLED", env
    )


def _audiences(claims: Mapping[str, object]) -> set[str]:
    raw = claims.get("aud")
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, (list, tuple, set)):
        return {str(value) for value in raw if str(value).strip()}
    return set()


def authorize_service_claims(
    claims: Mapping[str, object] | None,
    env: Mapping[str, str] | None = None,
) -> str:
    source = env if env is not None else os.environ
    payload = claims if isinstance(claims, Mapping) else {}
    allowed = inbound_client_ids(source)
    caller = str(payload.get("azp") or payload.get("client_id") or "").strip()
    if not allowed or not caller or caller not in allowed:
        raise ServiceCallerForbidden("service_client_not_allowed")

    required_audience = str(source.get("INBOUND_SERVICE_AUDIENCE") or "").strip()
    if required_audience and required_audience not in _audiences(payload):
        raise ServiceCallerForbidden("service_audience_not_allowed")

    required_scope = str(source.get("INBOUND_SERVICE_REQUIRED_SCOPE") or "").strip()
    scopes = {item for item in str(payload.get("scope") or "").split() if item}
    if required_scope and required_scope not in scopes:
        raise ServiceCallerForbidden("service_scope_missing")
    return caller


def validate_quota_runtime_config(
    env: Mapping[str, str] | None = None,
    *,
    api_process: bool,
) -> None:
    """Abort API/worker startup when index quota could operate fail-open."""

    source = env if env is not None else os.environ
    if not env_enabled("INDEX_QUOTA_ENABLED", source):
        return

    errors: list[str] = []
    if env_enabled("QUOTA_FAIL_OPEN", source):
        errors.append("QUOTA_FAIL_OPEN must be disabled")
    if not str(source.get("TINNTEN_SERVER_BASE_URL") or "").strip():
        errors.append("TINNTEN_SERVER_BASE_URL is required")
    for key in ("KEYCLOAK_BASE_URL", "REALM", "SERVICES_CLIENT_ID", "SERVICES_CLIENT_SECRET"):
        if not str(source.get(key) or "").strip():
            errors.append(f"{key} is required")

    if api_process:
        if not env_enabled("REQUIRE_KEYCLOAK_AUTH", source):
            errors.append("REQUIRE_KEYCLOAK_AUTH=true is required")
        if env_enabled("ALLOW_UNAUTH_CONTENT_INDEX", source):
            errors.append("ALLOW_UNAUTH_CONTENT_INDEX must be disabled")
        if not inbound_client_ids(source):
            errors.append("INBOUND_SERVICE_CLIENT_IDS is required")

    if errors:
        raise ServiceAuthConfigError(
            "index quota security configuration is invalid: " + "; ".join(errors)
        )
