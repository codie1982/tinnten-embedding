"""
MongoDB connection utilities for the embedding service.

These helpers replicate the semantics of the Node.js `connectDB` helper by providing a
singleton `MongoClient` as well as a convenience accessor for the default database.
"""
from __future__ import annotations

import os
from threading import Lock
from typing import Optional

from pymongo import MongoClient
from pymongo.errors import PyMongoError


class MongoConfigError(RuntimeError):
    """Raised when MongoDB configuration is missing or invalid."""


class MongoConnectionError(RuntimeError):
    """Raised when a MongoDB connection cannot be established."""


_client: Optional[MongoClient] = None
_lock = Lock()


def get_mongo_client(uri: Optional[str] = None, *, refresh: bool = False) -> MongoClient:
    """
    Return a singleton MongoClient instance.

    Parameters
    ----------
    uri:
        Optional override for the MongoDB connection string. Defaults to the `MONGO_URI`
        environment variable.
    refresh:
        When True the cached client is disposed and a new connection is established.
    """
    global _client

    mongo_uri = (uri or os.getenv("MONGO_URI") or "").strip()
    if not mongo_uri:
        raise MongoConfigError("MONGO_URI environment variable is required.")

    if _client is not None and not refresh:
        return _client

    with _lock:
        if _client is not None and not refresh:
            return _client
        try:
            timeout = int(os.getenv("MONGO_TIMEOUT_MS") or 30000)
            client = MongoClient(
                mongo_uri,
                appname="tinnten-embedding",
                serverSelectionTimeoutMS=timeout,
            )
            # Trigger a roundtrip to validate the connection.
            client.admin.command("ping")
        except PyMongoError as exc:
            raise MongoConnectionError(f"MongoDB connection failed: {exc}") from exc

        _client = client
        return _client


def get_database(name: Optional[str] = None):
    """
    Return a handle to the configured MongoDB database.

    Parameters
    ----------
    name:
        Optional override for the database name. Defaults to `DB_TINNTEN`.
    """
    db_name = (name or os.getenv("DB_TINNTEN") or "").strip()
    if not db_name:
        raise MongoConfigError("DB_TINNTEN environment variable is required.")

    client = get_mongo_client()
    return client[db_name]
