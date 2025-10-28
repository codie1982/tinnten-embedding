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

DEFAULT_QUEUE_NAME = "embedding.ingest"


class RabbitPublisher:
    def __init__(self, queue_name: str | None = None) -> None:
        self.queue_name = (queue_name or os.getenv("EMBED_QUEUE_NAME") or DEFAULT_QUEUE_NAME).strip()
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
        connection = self._get_connection()
        channel = connection.channel()
        try:
            self._declare_queue(channel)
            channel.basic_publish(
                exchange="",
                routing_key=self.queue_name,
                body=message.encode("utf-8"),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # make message persistent
                    content_type="application/json",
                ),
            )
        finally:
            try:
                channel.close()
            except Exception:
                pass
