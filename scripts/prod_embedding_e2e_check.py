#!/usr/bin/env python3
"""
Prod smoke test for tinnten-embedding fetcher indexing flow.

What it validates:
1) API request is accepted and queued.
2) ingest worker processes the job to completed state.
3) contentdocuments/contentdocumentlogs are written.
4) embedding_chunks are written with faiss_id values.
5) FAISS file exists and contains those IDs (when readable).

Usage examples:
  python scripts/prod_embedding_e2e_check.py \
    --base-url http://localhost:5003 \
    --company-id <COMPANY_ID> \
    --domain example.com \
    --mode domain

  python scripts/prod_embedding_e2e_check.py \
    --base-url http://localhost:5003 \
    --company-id <COMPANY_ID> \
    --domain example.com \
    --mode pages \
    --page-url https://example.com/a \
    --page-url https://example.com/b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

import requests
if TYPE_CHECKING:
    from bson import ObjectId
    from pymongo import MongoClient

try:
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def normalize_domain(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or parsed.path or "").strip().lower()
    if not host:
        return ""
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    return host.strip(".")


def normalize_page_url(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if "#" in raw:
        raw = raw.split("#", 1)[0].strip()
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or "/"
        out = f"{scheme}://{netloc}{path}"
        if parsed.query:
            out = f"{out}?{parsed.query}"
        return out
    return raw


def parse_page_urls(args: argparse.Namespace) -> List[str]:
    values: List[str] = []
    for item in args.page_url:
        normalized = normalize_page_url(item)
        if normalized:
            values.append(normalized)

    if args.page_urls_json:
        try:
            decoded = json.loads(args.page_urls_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --page-urls-json value: {exc}") from exc
        if not isinstance(decoded, list):
            raise ValueError("--page-urls-json must be a JSON array")
        for item in decoded:
            normalized = normalize_page_url(str(item))
            if normalized:
                values.append(normalized)

    if args.page_urls_file:
        file_path = Path(args.page_urls_file)
        raw = file_path.read_text(encoding="utf-8")
        try:
            decoded = json.loads(raw)
            if not isinstance(decoded, list):
                raise ValueError("JSON file must contain an array")
            for item in decoded:
                normalized = normalize_page_url(str(item))
                if normalized:
                    values.append(normalized)
        except json.JSONDecodeError:
            for line in raw.splitlines():
                normalized = normalize_page_url(line)
                if normalized:
                    values.append(normalized)

    seen = set()
    unique_values: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        unique_values.append(v)
    return unique_values


def resolve_faiss_path(cli_path: Optional[str]) -> Path:
    candidates = [
        cli_path,
        os.getenv("CHUNK_INDEX_PATH"),
        os.getenv("FAISS_INDEX_PATH"),
        "faiss.index",
    ]
    selected = ""
    for item in candidates:
        if item and str(item).strip():
            selected = str(item).strip()
            break
    if not selected:
        selected = "faiss.index"
    path = Path(selected)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _safe_object_id(value: Any):
    try:
        from bson import ObjectId  # type: ignore
    except Exception:
        return None
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def company_filters(company_id: str) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = [
        {"companyId": company_id},
        {"companyid": company_id},
    ]
    oid = _safe_object_id(company_id)
    if oid is not None:
        filters.extend([{"companyId": oid}, {"companyid": oid}])
    return filters


@dataclass
class CaseResult:
    name: str
    document_id: str
    job_id: str
    endpoint: str
    queued_http_status: int
    final_state: str
    chunk_count: int
    log_count: int
    faiss_ids_checked: int
    passed: bool
    details: Dict[str, Any]


def read_faiss_snapshot(index_path: Path) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "path": str(index_path),
        "exists": index_path.exists(),
        "ntotal": None,
        "dimension": None,
        "ids": None,
        "error": None,
    }
    if faiss is None:
        snapshot["error"] = "faiss module not available"
        return snapshot
    if not index_path.exists():
        return snapshot
    try:
        index = faiss.read_index(str(index_path))
        snapshot["ntotal"] = int(index.ntotal)
        snapshot["dimension"] = int(index.d)
        id_values = None
        if hasattr(index, "id_map"):
            try:
                id_values = set(int(v) for v in faiss.vector_to_array(index.id_map).tolist())
            except Exception:
                id_values = None
        snapshot["ids"] = id_values
        return snapshot
    except Exception as exc:  # noqa: BLE001
        snapshot["error"] = str(exc)
        return snapshot


def build_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def post_json(session: requests.Session, url: str, token: Optional[str], payload: Dict[str, Any]) -> requests.Response:
    return session.post(url, headers=build_headers(token), json=payload, timeout=(10, 120))


def poll_document_state(
    collection,
    *,
    company_id: str,
    document_id: str,
    timeout_seconds: int,
    poll_interval: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    deadline = time.time() + timeout_seconds
    filters = company_filters(company_id)
    query = {"$and": [{"$or": filters}, {"documentId": document_id}]}

    while time.time() < deadline:
        doc = collection.find_one(query)
        if doc:
            index_obj = doc.get("index") if isinstance(doc.get("index"), dict) else {}
            state = (index_obj.get("state") or doc.get("indexState") or "").lower()
            if state in {"completed", "failed"}:
                return doc, state
        time.sleep(poll_interval)

    doc = collection.find_one(query)
    state = "timeout"
    if doc:
        index_obj = doc.get("index") if isinstance(doc.get("index"), dict) else {}
        state = (index_obj.get("state") or doc.get("indexState") or "unknown").lower()
    return doc, state


def collect_case_data(
    *,
    mongo_client,
    content_db_name: str,
    embed_db_name: str,
    company_id: str,
    document_id: str,
    job_id: str,
) -> Dict[str, Any]:
    content_db = mongo_client[content_db_name]
    embed_db = mongo_client[embed_db_name]

    content_docs = content_db["contentdocuments"]
    content_logs = content_db["contentdocumentlogs"]
    chunks = embed_db["embedding_chunks"]

    filters = company_filters(company_id)
    doc_query = {"$and": [{"$or": filters}, {"documentId": document_id}]}
    log_query = {
        "$and": [
            {"$or": filters},
            {"documentId": document_id},
        ]
    }

    doc = content_docs.find_one(doc_query)
    log_count = int(content_logs.count_documents(log_query))
    try:
        from pymongo import DESCENDING  # type: ignore
    except Exception:
        DESCENDING = -1
    latest_logs = list(content_logs.find(log_query).sort("createdAt", DESCENDING).limit(5))

    chunk_docs = list(chunks.find({"doc_id": document_id}, {"faiss_id": 1, "chunk_index": 1}))
    chunk_count = len(chunk_docs)
    faiss_ids = [int(row.get("faiss_id")) for row in chunk_docs if row.get("faiss_id") is not None]

    state = "unknown"
    error_msg = None
    stats = {}
    if doc:
        idx = doc.get("index") if isinstance(doc.get("index"), dict) else {}
        state = (idx.get("state") or doc.get("indexState") or "unknown").lower()
        error_msg = idx.get("errorMsg")
        stats = idx.get("stats") if isinstance(idx.get("stats"), dict) else {}

    return {
        "doc": doc,
        "state": state,
        "errorMsg": error_msg,
        "stats": stats,
        "logCount": log_count,
        "latestLogStates": [row.get("state") for row in latest_logs],
        "chunkCount": chunk_count,
        "faissIds": faiss_ids,
        "jobId": job_id,
    }


def run_case(
    *,
    case_name: str,
    endpoint: str,
    payload: Dict[str, Any],
    session: requests.Session,
    token: Optional[str],
    mongo_client,
    content_db_name: str,
    embed_db_name: str,
    faiss_path: Path,
    timeout_seconds: int,
    poll_interval: float,
    keep_data: bool,
    base_url: str,
) -> CaseResult:
    document_id = str(payload["documentId"])

    before_snapshot = read_faiss_snapshot(faiss_path)
    log(f"[{case_name}] POST {endpoint} documentId={document_id}")
    response = post_json(session, endpoint, token, payload)

    queued_status = response.status_code
    if queued_status != 202:
        raise RuntimeError(f"{case_name}: queue request failed status={queued_status} body={response.text}")

    data = response.json() if response.content else {}
    job_id = str(data.get("jobId") or payload.get("jobId") or "")

    content_docs = mongo_client[content_db_name]["contentdocuments"]
    doc, state = poll_document_state(
        content_docs,
        company_id=str(payload["companyId"]),
        document_id=document_id,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
    )

    details = collect_case_data(
        mongo_client=mongo_client,
        content_db_name=content_db_name,
        embed_db_name=embed_db_name,
        company_id=str(payload["companyId"]),
        document_id=document_id,
        job_id=job_id,
    )

    after_snapshot = read_faiss_snapshot(faiss_path)
    details["beforeFaiss"] = {k: v for k, v in before_snapshot.items() if k != "ids"}
    details["afterFaiss"] = {k: v for k, v in after_snapshot.items() if k != "ids"}

    passed = True
    fail_reasons: List[str] = []

    if state != "completed":
        passed = False
        fail_reasons.append(f"state is {state}")

    chunk_count = int(details.get("chunkCount") or 0)
    if chunk_count <= 0:
        passed = False
        fail_reasons.append("chunkCount is 0")

    log_count = int(details.get("logCount") or 0)
    if log_count <= 0:
        passed = False
        fail_reasons.append("contentdocumentlogs missing")

    faiss_ids = details.get("faissIds") or []
    faiss_checked = 0
    if after_snapshot.get("exists"):
        if faiss is None:
            passed = False
            fail_reasons.append("faiss python module unavailable")
        elif after_snapshot.get("error"):
            passed = False
            fail_reasons.append(f"faiss read error: {after_snapshot['error']}")
        else:
            ids_set = after_snapshot.get("ids")
            if isinstance(ids_set, set) and faiss_ids:
                missing = [fid for fid in faiss_ids if fid not in ids_set]
                faiss_checked = len(faiss_ids)
                if missing:
                    passed = False
                    fail_reasons.append(f"{len(missing)} faiss_id values missing in index")
            else:
                ntotal = int(after_snapshot.get("ntotal") or 0)
                if ntotal <= 0:
                    passed = False
                    fail_reasons.append("faiss index ntotal is 0")
    else:
        passed = False
        fail_reasons.append(f"faiss file missing at {faiss_path}")

    details["failReasons"] = fail_reasons

    if passed and not keep_data:
        cleanup_url = f"{base_url.rstrip('/')}/api/v10/content/index/remove"
        cleanup_payload = {
            "companyId": str(payload["companyId"]),
            "documentId": document_id,
        }
        try:
            cleanup_resp = post_json(session, cleanup_url, token, cleanup_payload)
            details["cleanupStatus"] = cleanup_resp.status_code
            details["cleanupBody"] = cleanup_resp.text[:400]
        except Exception as exc:  # noqa: BLE001
            details["cleanupStatus"] = "error"
            details["cleanupBody"] = str(exc)

    final_state = "completed" if passed else "failed"
    return CaseResult(
        name=case_name,
        document_id=document_id,
        job_id=job_id,
        endpoint=endpoint,
        queued_http_status=queued_status,
        final_state=final_state,
        chunk_count=chunk_count,
        log_count=log_count,
        faiss_ids_checked=faiss_checked,
        passed=passed,
        details=details,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prod e2e smoke test for tinnten-embedding fetcher indexing")
    parser.add_argument("--base-url", default=os.getenv("EMBEDDING_BASE_URL", "http://localhost:5003"))
    parser.add_argument("--token", default=os.getenv("EMBEDDING_BEARER_TOKEN"))
    parser.add_argument("--company-id", default=os.getenv("COMPANY_ID"), required=os.getenv("COMPANY_ID") is None)
    parser.add_argument("--domain", default=os.getenv("FETCHER_TEST_DOMAIN"), required=os.getenv("FETCHER_TEST_DOMAIN") is None)
    parser.add_argument("--mode", choices=["domain", "pages", "both"], default=os.getenv("EMBED_TEST_MODE", "both"))

    parser.add_argument("--page-limit", type=int, default=int(os.getenv("EMBED_TEST_PAGE_LIMIT", "20")))
    parser.add_argument("--page-url", action="append", default=[])
    parser.add_argument("--page-urls-json", default=os.getenv("EMBED_TEST_PAGE_URLS_JSON"))
    parser.add_argument("--page-urls-file", default=os.getenv("EMBED_TEST_PAGE_URLS_FILE"))

    parser.add_argument("--fetched-from", default=os.getenv("EMBED_TEST_FETCHED_FROM"))
    parser.add_argument("--fetched-to", default=os.getenv("EMBED_TEST_FETCHED_TO"))
    parser.add_argument("--storage-preference", default=os.getenv("EMBED_TEST_STORAGE", "db,s3,disk"))

    parser.add_argument("--timeout", type=int, default=int(os.getenv("EMBED_TEST_TIMEOUT", "900")))
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("EMBED_TEST_POLL_INTERVAL", "5")))

    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI"), required=os.getenv("MONGO_URI") is None)
    parser.add_argument("--content-db", default=os.getenv("DB_TINNTEN", "tinnten"))
    parser.add_argument("--embed-db", default=os.getenv("EMBED_DB_NAME", "tinnten-embedding"))
    parser.add_argument("--faiss-path", default=os.getenv("CHUNK_INDEX_PATH") or os.getenv("FAISS_INDEX_PATH") or "faiss.index")

    parser.add_argument("--keep-data", action="store_true", help="Do not remove created test document/index after validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    domain = normalize_domain(args.domain)
    if not domain:
        print("ERROR: invalid domain", file=sys.stderr)
        return 2

    page_urls = parse_page_urls(args)
    if args.mode in {"pages", "both"} and not page_urls:
        print("ERROR: pages/both mode requires --page-url or --page-urls-json/--page-urls-file", file=sys.stderr)
        return 2

    run_id = f"prod-e2e-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    faiss_path = resolve_faiss_path(args.faiss_path)

    log(f"Run id: {run_id}")
    log(f"Base URL: {base_url}")
    log(f"Domain: {domain}")
    log(f"CompanyId: {args.company_id}")
    log(f"Mode: {args.mode}")
    log(f"Mongo content DB: {args.content_db} embed DB: {args.embed_db}")
    log(f"FAISS path: {faiss_path}")

    try:
        from pymongo import MongoClient  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: pymongo is required to run this script: {exc}", file=sys.stderr)
        return 2

    mongo = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=15000)
    try:
        mongo.admin.command("ping")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Mongo ping failed: {exc}", file=sys.stderr)
        return 2

    session = requests.Session()
    results: List[CaseResult] = []

    try:
        if args.mode in {"domain", "both"}:
            doc_id = f"{run_id}-domain"
            payload: Dict[str, Any] = {
                "companyId": str(args.company_id),
                "documentId": doc_id,
                "domain": domain,
                "pageLimit": int(args.page_limit),
                "storagePreference": args.storage_preference,
                "trigger": "prod_e2e_domain",
                "metadata": {
                    "testRunId": run_id,
                    "testCase": "domain",
                    "domain": domain,
                    "createdAt": utc_now(),
                },
                "indexOptions": {
                    "chunkSize": 1200,
                    "chunkOverlap": 200,
                    "minChars": 80,
                },
            }
            if args.fetched_from:
                payload["fetchedFrom"] = args.fetched_from
            if args.fetched_to:
                payload["fetchedTo"] = args.fetched_to

            endpoint = f"{base_url}/api/v10/content/index/fetcher"
            results.append(
                run_case(
                    case_name="domain",
                    endpoint=endpoint,
                    payload=payload,
                    session=session,
                    token=args.token,
                    mongo_client=mongo,
                    content_db_name=args.content_db,
                    embed_db_name=args.embed_db,
                    faiss_path=faiss_path,
                    timeout_seconds=int(args.timeout),
                    poll_interval=float(args.poll_interval),
                    keep_data=bool(args.keep_data),
                    base_url=base_url,
                )
            )

        if args.mode in {"pages", "both"}:
            doc_id = f"{run_id}-pages"
            payload = {
                "companyId": str(args.company_id),
                "documentId": doc_id,
                "domain": domain,
                "pageUrls": page_urls,
                "storagePreference": args.storage_preference,
                "trigger": "prod_e2e_pages",
                "metadata": {
                    "testRunId": run_id,
                    "testCase": "pages",
                    "domain": domain,
                    "requestedPageCount": len(page_urls),
                    "createdAt": utc_now(),
                },
                "indexOptions": {
                    "chunkSize": 1200,
                    "chunkOverlap": 200,
                    "minChars": 80,
                },
            }
            endpoint = f"{base_url}/api/v10/content/index/fetcher/pages"
            results.append(
                run_case(
                    case_name="pages",
                    endpoint=endpoint,
                    payload=payload,
                    session=session,
                    token=args.token,
                    mongo_client=mongo,
                    content_db_name=args.content_db,
                    embed_db_name=args.embed_db,
                    faiss_path=faiss_path,
                    timeout_seconds=int(args.timeout),
                    poll_interval=float(args.poll_interval),
                    keep_data=bool(args.keep_data),
                    base_url=base_url,
                )
            )

    finally:
        session.close()
        mongo.close()

    print("\n=== PROD E2E RESULT ===", flush=True)
    failures = 0
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        if not result.passed:
            failures += 1
        print(
            f"- case={result.name} status={status} endpoint={result.endpoint} "
            f"doc={result.document_id} job={result.job_id} chunks={result.chunk_count} logs={result.log_count}",
            flush=True,
        )
        if not result.passed:
            reasons = result.details.get("failReasons") or []
            for reason in reasons:
                print(f"    reason: {reason}", flush=True)
            if result.details.get("errorMsg"):
                print(f"    index.errorMsg: {result.details['errorMsg']}", flush=True)

    print("\n=== RAW DETAILS (JSON) ===", flush=True)
    print(
        json.dumps(
            [
                {
                    "case": r.name,
                    "passed": r.passed,
                    "documentId": r.document_id,
                    "jobId": r.job_id,
                    "endpoint": r.endpoint,
                    "details": r.details,
                }
                for r in results
            ],
            ensure_ascii=True,
            indent=2,
            default=str,
        ),
        flush=True,
    )

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
