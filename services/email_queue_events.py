import logging
import os
import threading
import time
from datetime import datetime

from services.rabbit_publisher import RabbitPublisher
from services.tinnten_server_client import get_tinnten_server_client


logger = logging.getLogger("tinnten.embedding.email_events")


def _flag_enabled(value):
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


class EmbeddingEmailEvents:
    # İndexleme e-postaları KALDIRILDI (started/completed/failed/upload).
    # `_enabled` KALICI False — EMBEDDING_EMAIL_EVENTS_ENABLED bayrağından BAĞIMSIZ
    # (bayrak açılsa bile mail gitmez). Tüm send_* metodları `_publish_for_company`
    # üzerinden geçer, o da her ağ çağrısından ÖNCE `_enabled`'a bakıp bail eder →
    # tam, yan-etkisiz no-op. Geri bildirim artık abonelik state'i + UI/SSE ile verilir.
    def __init__(self):
        self.publisher = None
        self.server_client = None
        self.cache_ttl_seconds = 300
        self._enabled = False  # HARD-DISABLED — indexleme mailleri kaldırıldı
        self._cache = {}
        self._cache_lock = threading.RLock()
        logger.info("EmbeddingEmailEvents hard-disabled (indexleme e-postaları kaldırıldı).")

    def _resolve_company_context(self, company_id):
        normalized = str(company_id or "").strip()
        if not normalized:
            return None

        now_ts = time.time()
        with self._cache_lock:
            cached = self._cache.get(normalized)
            if cached and cached.get("expires_at", 0) > now_ts:
                return cached.get("value")

        value = None
        try:
            value = self.server_client.get_company_owner_contact(normalized)
        except Exception as exc:
            logger.warning("Failed to resolve company context companyId=%s: %s", normalized, exc)

        with self._cache_lock:
            self._cache[normalized] = {
                "value": value,
                "expires_at": now_ts + max(30, self.cache_ttl_seconds),
            }
        return value

    @staticmethod
    def _company_name(company_context):
        context = company_context or {}
        company = context.get("company") or {}
        return str(company.get("name") or "").strip() or "-"

    @staticmethod
    def _owner_email(company_context):
        context = company_context or {}
        owner = context.get("owner") or {}
        return str(owner.get("email") or "").strip() or None

    @staticmethod
    def _format_dt(value):
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S UTC")
        return "-"

    def _publish_for_company(self, *, company_id, event_type, subject, content):
        if not self._enabled:
            return False
        context = self._resolve_company_context(company_id)
        to = self._owner_email(context)
        if not to:
            return False

        payload = {
            "type": event_type,
            "data": {"to": to, "subject": subject},
            "content": {
                "company_id": str(company_id),
                "company_name": self._company_name(context),
                **(content or {}),
            },
        }
        try:
            self.publisher.publish(payload)
            return True
        except Exception as exc:
            logger.error(
                "Failed to publish embedding email event=%s companyId=%s: %s",
                event_type,
                company_id,
                exc,
            )
            return False

    def send_index_started(
        self,
        *,
        company_id,
        document_id,
        job_id,
        source,
        trigger,
    ):
        return self._publish_for_company(
            company_id=company_id,
            event_type="embedding_index_started",
            subject="Indexleme basladi",
            content={
                "document_id": str(document_id),
                "job_id": str(job_id),
                "source": str(source or "-"),
                "trigger": str(trigger or "-"),
            },
        )

    def send_index_completed(
        self,
        *,
        company_id,
        document_id,
        job_id,
        source,
        stats,
        finished_at,
    ):
        stats_payload = stats or {}
        return self._publish_for_company(
            company_id=company_id,
            event_type="embedding_index_completed",
            subject="Indexleme tamamlandi",
            content={
                "document_id": str(document_id),
                "job_id": str(job_id),
                "source": str(source or "-"),
                "chunk_count": int(stats_payload.get("chunkCount") or 0),
                "token_count": int(stats_payload.get("tokenCount") or 0),
                "char_count": int(stats_payload.get("charCount") or 0),
                "finished_at": self._format_dt(finished_at),
            },
        )

    def send_upload_access_failed(
        self,
        *,
        company_id,
        document_id,
        job_id,
        reason,
    ):
        return self._publish_for_company(
            company_id=company_id,
            event_type="embedding_upload_access_failed",
            subject="Yuklenen dosyaya erisilemiyor",
            content={
                "document_id": str(document_id),
                "job_id": str(job_id),
                "reason": str(reason or "Dosya okunamadi"),
            },
        )

    def send_upload_format_unsupported(
        self,
        *,
        company_id,
        document_id,
        job_id,
        reason,
    ):
        return self._publish_for_company(
            company_id=company_id,
            event_type="embedding_upload_format_unsupported",
            subject="Dosya formati desteklenmiyor",
            content={
                "document_id": str(document_id),
                "job_id": str(job_id),
                "reason": str(reason or "Desteklenmeyen dosya formati"),
            },
        )

    def send_index_failed(
        self,
        *,
        company_id,
        document_id,
        job_id,
        source,
        stage,
        reason,
    ):
        return self._publish_for_company(
            company_id=company_id,
            event_type="embedding_index_failed",
            subject="Indexleme basarisiz",
            content={
                "document_id": str(document_id),
                "job_id": str(job_id),
                "source": str(source or "-"),
                "stage": str(stage or "-"),
                "reason": str(reason or "Bilinmeyen hata"),
            },
        )
