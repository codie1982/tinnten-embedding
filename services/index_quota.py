"""Central quota protocol helpers for company-scoped vector indexes."""

from __future__ import annotations

import hashlib

from services.tinnten_server_client import get_tinnten_server_client


def index_quota_operation_key(
    company_id: str,
    document_id: str,
    revision: str,
    metric: str,
    *,
    purpose: str = "swap",
) -> str:
    digest = hashlib.sha256(
        "\n".join(
            [
                str(company_id or "").strip(),
                str(document_id or "").strip(),
                str(revision or "").strip(),
                str(metric or "").strip(),
                str(purpose or "").strip(),
            ]
        ).encode("utf-8")
    ).hexdigest()
    return f"index-{purpose}:{digest}"


def mutate_index_quota(
    *,
    metric: str,
    action: str,
    company_id: str,
    document_id: str,
    revision: str,
    amount: int,
    bootstrap_used: int | None = None,
    purpose: str = "swap",
    metadata: dict | None = None,
):
    return get_tinnten_server_client().mutate_quota(
        section="index",
        metric=metric,
        action=action,
        company_id=company_id,
        operation_key=index_quota_operation_key(
            company_id,
            document_id,
            revision,
            metric,
            purpose=purpose,
        ),
        amount=amount,
        bootstrap_used=bootstrap_used,
        owner_type="embedding_document",
        owner_id=document_id,
        metadata={
            **(metadata or {}),
            "companyId": str(company_id or "").strip(),
            "documentId": str(document_id),
            "revision": str(revision),
            "purpose": str(purpose),
            "indexIdentity": f"company:{str(company_id or '').strip()}",
        },
    )
