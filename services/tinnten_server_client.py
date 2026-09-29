import os

import requests

from services.keycloak_service import KeycloakConfigError, get_keycloak_service


class QuotaServiceError(RuntimeError):
    """The central quota service could not complete a mutation."""


class QuotaExceededError(QuotaServiceError):
    """A reserve request was rejected because the account limit is exhausted."""

    def __init__(self, message, *, data=None, status_code=None):
        super().__init__(message)
        self.data = data if isinstance(data, dict) else {}
        self.status_code = status_code


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

        try:
            payload = response.json() if response.content else {}
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            return None
        if payload.get("success") is False:
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        return data

    def mutate_quota(
        self,
        *,
        section,
        metric,
        action,
        company_id,
        operation_key,
        amount,
        bootstrap_used=None,
        owner_type=None,
        owner_id=None,
        metadata=None,
    ):
        """Drive one idempotent reserve/commit/release ledger operation."""
        normalized_action = str(action or "").strip().lower()
        if normalized_action not in {
            "reserve",
            "commit",
            "release",
            "cancel-reservation",
        }:
            raise ValueError(
                "quota action must be reserve, commit, release or cancel-reservation"
            )
        company_id = str(company_id or "").strip()
        operation_key = str(operation_key or "").strip()
        if not company_id or not operation_key:
            raise ValueError("company_id and operation_key are required")
        try:
            normalized_amount = int(amount)
        except (TypeError, ValueError) as exc:
            raise ValueError("quota amount must be an integer") from exc
        if normalized_amount <= 0:
            raise ValueError("quota amount must be greater than zero")
        if not self.base_url:
            raise QuotaServiceError("TINNTEN_SERVER_BASE_URL is not configured.")

        body = {
            "companyId": company_id,
            "operationKey": operation_key,
            "amount": normalized_amount,
        }
        if bootstrap_used is not None:
            body["bootstrapUsed"] = max(0, int(bootstrap_used))
        if owner_type:
            body["ownerType"] = str(owner_type)
        if owner_id:
            body["ownerId"] = str(owner_id)
        if isinstance(metadata, dict) and metadata:
            body["metadata"] = metadata

        url = (
            f"{self.base_url}/api/v10/internal/quotas/"
            f"{section}/{metric}/{normalized_action}"
        )
        try:
            response = requests.post(
                url,
                json=body,
                headers=self._build_headers(),
                timeout=self.timeout,
            )
        except Exception as exc:
            raise QuotaServiceError(f"Tinnten quota request failed: {exc}") from exc

        try:
            payload = response.json() if response.content else {}
        except ValueError:
            payload = {}
        data = payload.get("data") if isinstance(payload, dict) else None
        error_code = ""
        if isinstance(data, dict):
            error_code = str(data.get("code") or "").strip().lower()
        if not error_code and isinstance(payload, dict):
            error_code = str(payload.get("code") or "").strip().lower()
        is_account_limit = error_code in {
            "account_resource_limit_exceeded",
            "account_limit_exceeded",
        }
        if (
            response.status_code in {403, 409}
            and normalized_action == "reserve"
            and is_account_limit
        ):
            message = (
                (payload or {}).get("message")
                if isinstance(payload, dict)
                else None
            ) or "quota_exceeded"
            raise QuotaExceededError(
                message,
                data=data,
                status_code=response.status_code,
            )
        if response.status_code >= 400 or (
            isinstance(payload, dict) and payload.get("success") is False
        ):
            message = (
                (payload or {}).get("message")
                if isinstance(payload, dict)
                else None
            ) or response.text or "quota mutation failed"
            raise QuotaServiceError(
                f"Tinnten quota request failed ({response.status_code}): {message}"
            )
        return data if isinstance(data, dict) else (payload or {})

    def get_quota_usage(self, *, section, metric, company_id):
        company_id = str(company_id or "").strip()
        if not company_id:
            raise ValueError("company_id is required")
        if not self.base_url:
            raise QuotaServiceError("TINNTEN_SERVER_BASE_URL is not configured.")
        url = f"{self.base_url}/api/v10/internal/quotas/{section}/{metric}/usage"
        try:
            response = requests.get(
                url,
                params={"companyId": company_id},
                headers=self._build_headers(),
                timeout=self.timeout,
            )
        except Exception as exc:
            raise QuotaServiceError(f"Tinnten quota usage request failed: {exc}") from exc
        try:
            payload = response.json() if response.content else {}
        except ValueError:
            payload = {}
        if response.status_code >= 400 or (
            isinstance(payload, dict) and payload.get("success") is False
        ):
            raise QuotaServiceError(
                f"Tinnten quota usage request failed ({response.status_code}): {response.text}"
            )
        data = payload.get("data") if isinstance(payload, dict) else None
        return data if isinstance(data, dict) else (payload or {})

    def update_document_index_state(
        self,
        document_id: str,
        state: str,
        *,
        error_msg: str | None = None,
        stats: dict | None = None,
        company_id: str | None = None,
        domain: str | None = None,
        source: str | None = None,
        page_url: str | None = None,
        page_title: str | None = None,
        page_source_subscription_id: str | None = None,
        domain_chunks: int | None = None,
        job_id: str | None = None,
        attempt: int | None = None,
    ) -> bool:
        """Notify tinnten-server of a document index state change.

        Maps embedding-service states to tinnten-server states:
          completed → indexed
          failed    → error
          indexing  → indexing
        Returns True on success, False on soft failure (caller should not crash).
        """
        if not self.base_url:
            return False

        # Map embedding service state names to tinnten-server enum values
        state_map = {
            "completed": "indexed",
            "failed": "error",
            "indexing": "indexing",
            "processing": "indexing",
            "queued": "queued",
        }
        mapped_state = state_map.get(state, state)

        body = {"state": mapped_state}
        if job_id:
            body["jobId"] = str(job_id)
        if attempt is not None:
            body["attempt"] = int(attempt)
        if error_msg:
            body["errorMsg"] = error_msg
        if stats and isinstance(stats, dict):
            # Keep the server's historical aliases while carrying the richer
            # worker measurements. The server merges this object into
            # index.stats; dropping extraction/chunking fields here made the
            # API database disagree with the worker database.
            allowed_stat_keys = (
                "chunkCount",
                "tokenCount",
                "charCount",
                "chunkSize",
                "chunkOverlap",
                "minChars",
                "effectiveMinChars",
                "chunkMode",
                "chunkPolicy",
                "chunkStrategy",
                "resolvedChunkStrategy",
                "chunkCharacterCount",
                "extractedCharCount",
                "meaningfulCharCount",
                "failureReason",
                "cleanup",
                "ocr",
                "langDetect",
                "scope",
            )
            body["stats"] = {
                key: stats[key]
                for key in allowed_stat_keys
                if key in stats and stats[key] is not None
            }
            body["stats"].update(
                {
                    "chunks": int(stats.get("chunkCount") or stats.get("chunks") or 0),
                    "tokens": int(stats.get("tokenCount") or stats.get("tokens") or 0),
                }
            )
        if company_id:
            body["companyid"] = company_id
        # Per-sayfa (fetcher_page/initial) doc'ları için: server, website entry'sini
        # tek faissIndexId yerine DOMAIN ile bulur; domainChunks agrega chunk sayısı.
        if domain and source in ("fetcher_page", "fetcher_initial"):
            body["metadata"] = {
                "domain": str(domain),
                "source": str(source),
                "documentId": str(document_id),
            }
            if page_url:
                body["metadata"]["url"] = str(page_url)
            if page_title:
                body["metadata"]["title"] = str(page_title)
            if page_source_subscription_id:
                body["metadata"]["sourceSubscriptionId"] = str(page_source_subscription_id)
            if isinstance(domain_chunks, int):
                body.setdefault("stats", {})
                body["stats"]["domainChunks"] = int(domain_chunks)

        try:
            url = f"{self.base_url}/api/v10/internal/content/documents/{document_id}/index-state"
            response = requests.patch(
                url,
                json=body,
                headers=self._build_headers(),
                timeout=self.timeout,
            )
            return response.status_code < 400
        except Exception:
            return False


_tinnten_server_client = None


def get_tinnten_server_client():
    global _tinnten_server_client
    if _tinnten_server_client is None:
        _tinnten_server_client = TinntenServerClient()
    return _tinnten_server_client
