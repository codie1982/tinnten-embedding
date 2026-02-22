import os

import requests

from services.keycloak_service import KeycloakConfigError, get_keycloak_service


class TinntenServerClient:
    def __init__(self):
        self.base_url = (os.getenv("TINNTEN_SERVER_BASE_URL") or "").rstrip("/")
        self.timeout = float(os.getenv("TINNTEN_SERVER_TIMEOUT_SECONDS") or 10.0)
        self.keycloak = None
        self.keycloak_init_error = None
        try:
            self.keycloak = get_keycloak_service()
        except KeycloakConfigError as exc:
            self.keycloak_init_error = str(exc)

    def _build_headers(self):
        headers = {"Accept": "application/json"}
        if not self.keycloak:
            raise RuntimeError(
                "Keycloak service token provider is unavailable for internal request."
            )
        access_token = self.keycloak.get_service_token()
        headers["Authorization"] = f"Bearer {access_token}"
        return headers

    def get_company_owner_contact(self, company_id):
        company_id = str(company_id or "").strip()
        if not company_id:
            return None
        if not self.base_url:
            raise RuntimeError("TINNTEN_SERVER_BASE_URL is not configured.")

        url = f"{self.base_url}/api/v10/internal/companies/{company_id}/contact"
        response = requests.get(
            url,
            headers=self._build_headers(),
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Tinnten server internal request failed ({response.status_code}): {response.text}"
            )

        payload = response.json() if response.content else {}
        if not isinstance(payload, dict):
            return None
        if payload.get("success") is False:
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        return data


_tinnten_server_client = None


def get_tinnten_server_client():
    global _tinnten_server_client
    if _tinnten_server_client is None:
        _tinnten_server_client = TinntenServerClient()
    return _tinnten_server_client
