"""
Helper for publishing ingest jobs to RabbitMQ.
"""
from __future__ import annotations

import json
import os
from threading import Lock
from typing import Any, Dict

import pika

from init.rabbit_connection import get_rabbit_connection

DEFAULT_QUEUE_NAME = "content_indexing_queue"


class RabbitPublisher:
    def __init__(self, queue_name: str | None = None) -> None:
        env_queue = (os.getenv("CONTENT_INDEX_QUEUE_NAME") or os.getenv("EMBED_QUEUE_NAME") or "").strip()
        target_queue = queue_name or env_queue or DEFAULT_QUEUE_NAME
        self.queue_name = target_queue.strip()
        if not self.queue_name:
            raise ValueError("queue_name cannot be empty")
        self._connection: pika.BlockingConnection | None = None
        self._lock = Lock()

    def _get_connection(self) -> pika.BlockingConnection:
        with self._lock:
            if self._connection is None or self._connection.is_closed:
                self._connection = get_rabbit_connection(refresh=True)
            return self._connection

    def _declare_queue(self, channel: pika.adapters.blocking_connection.BlockingChannel) -> None:
        channel.queue_declare(queue=self.queue_name, durable=True, auto_delete=False, exclusive=False)

    def publish(self, payload: Dict[str, Any]) -> None:
        message = json.dumps(payload, ensure_ascii=False)
        body = message.encode("utf-8")
        props = pika.BasicProperties(
            delivery_mode=2,  # make message persistent
            content_type="application/json",
        )
        # pika BlockingConnection THREAD-SAFE DEĞİL: gunicorn gthread (Faz 0,
        # --threads 8) altında birden çok HTTP thread'i aynı bağlantıyı sürünce
        # ioloop bozuluyor ("pop from an empty deque" / StreamLostError) ve
        # publish KAYBOLUYORDU (5 dk'da 38 content_index_publish_failed).
        # Çözüm: publish'in TAMAMI kilit altında serileşir (localhost'ta ms
        # ölçekli, ~22 msg/dk — darboğaz değil); bayat/kopmuş bağlantıda bir
        # kez taze bağlantıyla yeniden denenir.
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._connection is None or self._connection.is_closed:
                        self._connection = get_rabbit_connection(refresh=True)
                    channel = self._connection.channel()
                    try:
                        self._declare_queue(channel)
                        channel.basic_publish(
                            exchange="",
                            routing_key=self.queue_name,
                            body=body,
                            properties=props,
                        )
                        return
                    finally:
                        try:
                            channel.close()
                        except Exception:
                            pass
                except Exception:
                    # Bağlantıyı düşür; ikinci deneme taze bağlantıyla yapılır.
                    try:
                        if self._connection is not None:
                            self._connection.close()
                    except Exception:
                        pass
                    self._connection = None
                    if attempt == 2:
                        raise
