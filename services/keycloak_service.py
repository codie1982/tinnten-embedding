import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import requests


class KeycloakError(RuntimeError):
    """Base error raised when a Keycloak operation fails."""


class KeycloakConfigError(KeycloakError):
    """Raised when mandatory Keycloak configuration is missing."""


class KeycloakTokenError(KeycloakError):
    """Raised when a Keycloak token could not be created or validated."""


@dataclass(slots=True)
class _TokenCache:
    access_token: str
    token_type: Optional[str]
    scope: Optional[str]
    expires_at: float
    cache_key: str


class KeycloakService:
    """
    Lightweight helper around the Keycloak token endpoints.

    It can mint service-to-service tokens using the configured confidential client
    and validate / introspect incoming Bearer tokens.
    """

    _CACHE_LEEWAY_SECONDS = 5

    def __init__(self, *, timeout: Optional[float] = 10.0) -> None:
        self.base_url = (os.getenv("KEYCLOAK_BASE_URL") or "").rstrip("/")
        self.realm = (os.getenv("REALM") or "").strip()
        self.realm_master = (os.getenv("REALM_MASTER") or "").strip()
        self.client_id = (os.getenv("CLIENT_ID") or "").strip()
        self.client_secret = (os.getenv("CLIENT_SECRET") or "").strip()
        self.services_client_id = (os.getenv("SERVICES_CLIENT_ID") or "").strip()
        self.services_client_secret = (os.getenv("SERVICES_CLIENT_SECRET") or "").strip()

        if not self.base_url:
            raise KeycloakConfigError("KEYCLOAK_BASE_URL is required.")
        if not self.realm:
            raise KeycloakConfigError("REALM is required.")

        self.timeout = timeout
        self._session = requests.Session()
        self._service_token_cache: Optional[_TokenCache] = None

    # -- Public API -----------------------------------------------------
    def get_service_token(
        self,
        scope: Union[str, Sequence[str], None] = None,
        *,
        use_cache: bool = True,
    ) -> str:
        """
        Returns a Keycloak access token minted via the service client credentials flow.
        """
        return self.get_service_token_info(scope=scope, use_cache=use_cache)["access_token"]

    def get_auth_header(
        self,
        scope: Union[str, Sequence[str], None] = None,
        *,
        use_cache: bool = True,
    ) -> Dict[str, str]:
        """
        Returns a dictionary containing the Authorization header with a Bearer token.
        """
        token = self.get_service_token(scope=scope, use_cache=use_cache)
        return {"Authorization": f"Bearer {token}"}

    def get_service_token_info(
        self,
        scope: Union[str, Sequence[str], None] = None,
        *,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve a service token payload (access token + metadata).
        """
        if not self.services_client_id or not self.services_client_secret:
            raise KeycloakConfigError("SERVICES_CLIENT_ID and SERVICES_CLIENT_SECRET must be set.")

        normalized_scope = self._normalize_scope(scope)
        cache_key = normalized_scope or ""

        cached = self._service_token_cache
        if (
            use_cache
            and cached
            and cached.cache_key == cache_key
            and cached.expires_at - self._CACHE_LEEWAY_SECONDS > time.time()
        ):
            return {
                "access_token": cached.access_token,
                "token_type": cached.token_type,
                "scope": cached.scope,
                "expires_at": cached.expires_at,
            }

        token_url = self._build_url(f"/realms/{self.realm}/protocol/openid-connect/token")
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.services_client_id,
            "client_secret": self.services_client_secret,
        }
        if normalized_scope:
            payload["scope"] = normalized_scope

        response = self._session.post(
            token_url,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise KeycloakTokenError(
                f"Keycloak service token request failed ({response.status_code}): {response.text}"
            )

        data = response.json()
        access_token = data.get("access_token")
        if not access_token:
            raise KeycloakTokenError("Keycloak response does not include access_token.")

        expires_in = data.get("expires_in") or 0
        expires_at = time.time() + float(expires_in)
        cache_entry = _TokenCache(
            access_token=access_token,
            token_type=data.get("token_type"),
            scope=data.get("scope") or normalized_scope,
            expires_at=expires_at,
            cache_key=cache_key,
        )
        if use_cache:
            self._service_token_cache = cache_entry

        return {
            "access_token": cache_entry.access_token,
            "token_type": cache_entry.token_type,
            "scope": cache_entry.scope,
            "expires_at": cache_entry.expires_at,
        }

    def clear_service_token_cache(self) -> None:
        """Invalidate any cached service token."""
        self._service_token_cache = None

    def introspect_token(
        self,
        token: str,
        *,
        token_type_hint: str = "access_token",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate a token using Keycloak's introspection endpoint.

        Returns the parsed payload (which includes the `active` flag).
        """
        if not token:
            raise KeycloakTokenError("Token value is required for introspection.")

        introspect_client_id = (client_id or self.services_client_id or self.client_id or "").strip()
        introspect_client_secret = (client_secret or self.services_client_secret or self.client_secret or "").strip()
        if not introspect_client_id or not introspect_client_secret:
            raise KeycloakConfigError("A confidential client_id/client_secret pair is required for introspection.")

        payload = {
            "token": token,
            "client_id": introspect_client_id,
            "client_secret": introspect_client_secret,
        }
        if token_type_hint:
            payload["token_type_hint"] = token_type_hint

        introspection_url = self._build_url(f"/realms/{self.realm}/protocol/openid-connect/token/introspect")
        response = self._session.post(
            introspection_url,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise KeycloakTokenError(
                f"Keycloak token introspection failed ({response.status_code}): {response.text}"
            )

        data = response.json()
        if not isinstance(data, dict):
            raise KeycloakTokenError("Unexpected token introspection response.")
        return data

    def validate_bearer_header(
        self,
        authorization_header: Optional[str],
    ) -> Dict[str, Any]:
        """
        Extract a Bearer token from an Authorization header and validate it.
        """
        token = self.extract_bearer_token(authorization_header)
        payload = self.introspect_token(token)
        if not payload.get("active"):
            raise KeycloakTokenError("Token is not active.")
        return payload

    @staticmethod
    def extract_bearer_token(authorization_header: Optional[str]) -> str:
        if not authorization_header:
            raise KeycloakTokenError("Authorization header missing.")
        if not authorization_header.lower().startswith("bearer "):
            raise KeycloakTokenError("Authorization header is not Bearer type.")
        token = authorization_header.split(" ", 1)[1].strip()
        if not token:
            raise KeycloakTokenError("Bearer token is empty.")
        return token

    # -- Internal helpers -----------------------------------------------
    def _build_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _normalize_scope(scope: Union[str, Sequence[str], None]) -> str:
        if scope is None:
            return ""
        if isinstance(scope, str):
            return scope.strip()
        return " ".join(str(item).strip() for item in scope if item).strip()


_keycloak_singleton: Optional[KeycloakService] = None


def get_keycloak_service() -> KeycloakService:
    """
    Lazily create a singleton KeycloakService instance.
    """
    global _keycloak_singleton
    if _keycloak_singleton is None:
        _keycloak_singleton = KeycloakService()
    return _keycloak_singleton

