"""
Elasticsearch client helper mirroring the Node.js initialiser.
"""
from __future__ import annotations

import os
from threading import Lock
from typing import Optional

from elasticsearch import Elasticsearch
try:
    from elastic_transport import ConnectionError as ESConnectionError  # type: ignore
except ImportError:  # pragma: no cover - elastic-transport shipped with elasticsearch
    ESConnectionError = Exception  # type: ignore[misc,assignment]


class ElasticsearchConfigError(RuntimeError):
    """Raised when Elasticsearch configuration is missing."""


class ElasticsearchClientError(RuntimeError):
    """Raised when the Elasticsearch client cannot be created."""


_client: Optional[Elasticsearch] = None
_lock = Lock()


def get_elasticsearch_client(refresh: bool = False) -> Elasticsearch:
    """
    Return a cached Elasticsearch client configured from the environment.
    """
    global _client

    node_url = (os.getenv("ELASTICSEARCH_BASE_URL") or "http://localhost:9200").strip()
    if not node_url:
        raise ElasticsearchConfigError("ELASTICSEARCH_BASE_URL environment variable is required.")

    if _client is not None and not refresh:
        return _client

    with _lock:
        if _client is not None and not refresh:
            return _client

        try:
            client = Elasticsearch(hosts=[node_url])
            # Perform a lightweight ping to verify connectivity.
            client.ping()
        except (ESConnectionError, Exception) as exc:  # noqa: BLE001
            raise ElasticsearchClientError(f"Failed to initialize Elasticsearch client: {exc}") from exc

        _client = client
        return _client
