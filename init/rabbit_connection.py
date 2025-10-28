"""
RabbitMQ connection helpers mirroring the Node.js implementation.
"""
from __future__ import annotations

import os
import ssl
import time
from threading import Lock
from typing import Optional

import pika
from pika.exceptions import AMQPConnectionError


class RabbitConfigError(RuntimeError):
    """Raised when RabbitMQ configuration is missing or invalid."""


class RabbitConnectionError(RuntimeError):
    """Raised when a RabbitMQ connection could not be established."""


_connection: Optional[pika.BlockingConnection] = None
_lock = Lock()


def _build_parameters() -> pika.ConnectionParameters:
    protocol = (os.getenv("RABBITMQ_PROTOCOL") or "amqp").lower()
    host = os.getenv("RABBITMQ_HOST") or "rabbitmq"
    port = int(os.getenv("RABBITMQ_PORT") or 5672)
    username = os.getenv("RABBITMQ_USERNAME") or ""
    password = os.getenv("RABBITMQ_PASSWORD") or ""
    vhost = os.getenv("RABBITMQ_VHOST") or "/"
    heartbeat = int(os.getenv("RABBITMQ_HEARTBEAT") or 600)
    blocked_timeout = float(os.getenv("RABBITMQ_BLOCKED_TIMEOUT") or 300.0)

    credentials = None
    if username:
        credentials = pika.PlainCredentials(username, password)

    ssl_options = None
    if protocol == "amqps":
        context = ssl.create_default_context()
        ssl_options = pika.SSLOptions(context, host)

    return pika.ConnectionParameters(
        host=host,
        port=port,
        virtual_host=vhost,
        credentials=credentials,
        ssl_options=ssl_options,
        heartbeat=heartbeat,
        blocked_connection_timeout=blocked_timeout,
    )


def get_rabbit_connection(refresh: bool = False) -> pika.BlockingConnection:
    """
    Return a cached RabbitMQ connection.
    """
    global _connection

    if _connection and _connection.is_open and not refresh:
        return _connection

    with _lock:
        if _connection and _connection.is_open and not refresh:
            return _connection

        params = _build_parameters()
        try:
            connection = pika.BlockingConnection(params)
        except AMQPConnectionError as exc:
            raise RabbitConnectionError(f"RabbitMQ connection failed: {exc}") from exc

        _connection = connection
        return _connection


def connect_rabbit_with_retry(retries: int = 10, delay: float = 5.0) -> pika.BlockingConnection:
    """
    Attempt to connect to RabbitMQ, retrying with the supplied backoff settings.
    """
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return get_rabbit_connection(refresh=True)
        except RabbitConnectionError as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                break

    raise RabbitConnectionError(
        f"RabbitMQ connection could not be established after {retries} attempts."
    ) from last_error
