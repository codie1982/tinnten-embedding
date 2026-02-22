"""
Helpers for interacting with the `uploads` collection in the main tinnten database.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo import ReturnDocument

from init.db import get_database, get_mongo_client


DEFAULT_UPLOAD_DB_NAME = "tinnten"
UPLOADS_COLL = "uploads"
FILES_COLL = "files"


class UploadNotFoundError(RuntimeError):
    """Raised when an upload document cannot be located."""


_UNSET = object()


class UploadStore:
    def __init__(self, db_name: Optional[str] = None) -> None:
        name = (
            (db_name or os.getenv("EMBED_DOCUMENT_DB_NAME") or "").strip()
            or (os.getenv("DB_TINNTEN") or "").strip()
            or DEFAULT_UPLOAD_DB_NAME
        )
        self.db = get_database(name)
        self.uploads = self.db[UPLOADS_COLL]
        self.files = self.db[FILES_COLL]

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def get_upload_by_id(self, upload_id: str, *, projection: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        doc = self.uploads.find_one({"uploadid": upload_id}, projection)
        if not doc:
            raise UploadNotFoundError(f"upload record not found for uploadid={upload_id}")
        return doc

    def get_file_by_upload_id(self, upload_id: str, *, projection: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
        return self.files.find_one({"uploadid": upload_id}, projection)

    # ------------------------------------------------------------------
    # Status updates
    # ------------------------------------------------------------------
    def update_upload_status(
        self,
        upload_id: str,
        *,
        index_status: Optional[str] = None,
        is_file_opened: Optional[bool] = None,
        file_open_error: Optional[str] | object = _UNSET,
        embedding_doc_id: Optional[str] = None,
        session=None,
    ) -> Optional[Dict[str, Any]]:
        update: Dict[str, Any] = {"updatedAt": datetime.now(timezone.utc)}
        if index_status is not None:
            update["index_status"] = index_status
        if is_file_opened is not None:
            update["is_file_opened"] = is_file_opened
        if file_open_error is not _UNSET:
            update["file_open_error"] = file_open_error
        if embedding_doc_id is not None:
            update["embedding_doc_id"] = embedding_doc_id

        result = self.uploads.find_one_and_update(
            {"uploadid": upload_id},
            {"$set": update},
            return_document=ReturnDocument.AFTER,
            session=session,
        )
        return result

    # ------------------------------------------------------------------
    # Session helper
    # ------------------------------------------------------------------
    @staticmethod
    def start_session():
        client = get_mongo_client()
        return client.start_session()
