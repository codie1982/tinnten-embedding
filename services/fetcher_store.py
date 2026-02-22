"""
Read-only helpers for crawl artifacts produced by tinnten-fetcher.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import DESCENDING, MongoClient
from pymongo.errors import PyMongoError


DEFAULT_FETCHER_DB_NAME = "tinnten-fetcher"


class FetcherStore:
    def __init__(self, *, db_name: Optional[str] = None, mongo_uri: Optional[str] = None) -> None:
        resolved_db_name = (
            (db_name or os.getenv("FETCHER_DB_NAME") or "").strip()
            or DEFAULT_FETCHER_DB_NAME
        )
        resolved_uri = (mongo_uri or os.getenv("FETCHER_MONGO_URI") or os.getenv("MONGO_URI") or "").strip()
        if not resolved_uri:
            raise RuntimeError("FETCHER_MONGO_URI or MONGO_URI is required for FetcherStore.")
        timeout = int(os.getenv("MONGO_TIMEOUT_MS") or 30000)
        self.client = MongoClient(
            resolved_uri,
            appname="tinnten-embedding-fetcher",
            serverSelectionTimeoutMS=timeout,
        )
        try:
            self.client.admin.command("ping")
        except PyMongoError as exc:
            raise RuntimeError(f"Fetcher MongoDB connection failed: {exc}") from exc
        self.db = self.client[resolved_db_name]
        self.domains = self.db["domains"]
        self.crawl_results = self.db["crawl_results"]
        self.crawl_logs = self.db["crawl_logs"]
        self.crawl_media = self.db["crawl_media"]

    def get_domain(self, domain: str) -> Optional[Dict[str, Any]]:
        if not domain:
            return None
        return self.domains.find_one({"domain": str(domain).strip().lower()})

    def list_crawl_results(
        self,
        *,
        domain: str,
        limit: int = 50,
        fetched_from: Optional[datetime] = None,
        fetched_to: Optional[datetime] = None,
        urls: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"domain": str(domain).strip().lower()}
        if urls:
            normalized_urls = [str(u).strip() for u in urls if str(u).strip()]
            if normalized_urls:
                query["url"] = {"$in": normalized_urls}

        fetched_window: Dict[str, Any] = {}
        if fetched_from is not None:
            fetched_window["$gte"] = fetched_from
        if fetched_to is not None:
            fetched_window["$lte"] = fetched_to
        if fetched_window:
            query["fetched_at"] = fetched_window

        projection = {
            "_id": 0,
            "url": 1,
            "domain": 1,
            "title": 1,
            "description": 1,
            "markdown": 1,
            "html": 1,
            "fetched_at": 1,
            "content_preferences": 1,
            "extraction_id": 1,
            "processing_status": 1,
        }
        capped_limit = max(1, min(int(limit or 50), 500))
        cursor = self.crawl_results.find(query, projection).sort("fetched_at", DESCENDING).limit(capped_limit)
        return list(cursor)

    def latest_logs_by_urls(self, urls: List[str]) -> Dict[str, Dict[str, Any]]:
        url_values = [str(u).strip() for u in urls if str(u).strip()]
        if not url_values:
            return {}

        pipeline = [
            {"$match": {"url": {"$in": url_values}}},
            {"$sort": {"fetched_at": -1}},
            {"$group": {"_id": "$url", "doc": {"$first": "$$ROOT"}}},
        ]
        logs: Dict[str, Dict[str, Any]] = {}
        for row in self.crawl_logs.aggregate(pipeline):
            url = row.get("_id")
            doc = row.get("doc")
            if url and isinstance(doc, dict):
                logs[str(url)] = doc
        return logs

    def list_crawl_media_by_parent_urls(
        self,
        urls: List[str],
        *,
        limit_per_url: int = 20,
    ) -> Dict[str, List[Dict[str, Any]]]:
        url_values = [str(u).strip() for u in urls if str(u).strip()]
        if not url_values:
            return {}

        projection = {"_id": 0, "parent_url": 1, "type": 1, "src": 1, "alt": 1, "metadata": 1}
        cursor = self.crawl_media.find({"parent_url": {"$in": url_values}}, projection)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        capped = max(1, min(int(limit_per_url or 20), 100))
        for row in cursor:
            parent_url = str(row.get("parent_url") or "").strip()
            if not parent_url:
                continue
            bucket = grouped.setdefault(parent_url, [])
            if len(bucket) >= capped:
                continue
            bucket.append(row)
        return grouped
