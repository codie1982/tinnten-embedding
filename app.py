import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from bson import ObjectId
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from dotenv import load_dotenv

from vector_store import EmbeddingIndex, MetaRepository
from services import (
    DocumentLoader,
    DocumentDownloadError,
    DocumentParseError,
    UploadStore,
    UploadNotFoundError,
    ContentDocumentStore,
    normalize_index_options,
)
from services.embedding_engine import EmbeddingEngine
from services.mongo_store import MongoStore
from services.rabbit_publisher import RabbitPublisher
from services.keycloak_service import get_keycloak_service, KeycloakError, KeycloakTokenError
from init.db import get_database

load_dotenv()

BASE_DIR = os.path.dirname(__file__) or "."
logger = logging.getLogger("tinnten.embedding")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _path_from_env(*keys, default: str) -> str:
    for key in keys:
        if not key:
            continue
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return default


def _default_meta_path(index_path: str, fallback: str) -> str:
    base_dir = os.path.dirname(index_path) or "."
    return os.path.join(base_dir, fallback)


def _resolve_index_path(raw_path: str | None, default: str, *, base_dir: str = BASE_DIR) -> str:
    """
    Normalize index paths so they consistently resolve relative to the service base dir.
    """
    candidate = (raw_path or "").strip() or default
    candidate = os.path.expanduser(candidate)
    if not os.path.isabs(candidate):
        candidate = os.path.abspath(os.path.join(base_dir, candidate))
    return candidate


CHUNK_INDEX_PATH = _path_from_env(
    "CHUNK_INDEX_PATH",
    "CONTENT_INDEX_PATH",
    "FAISS_INDEX_PATH",
    "INDEX_PATH",
    default="faiss.index",
)
GENERAL_INDEX_PATH = _path_from_env("GENERAL_INDEX_PATH", default="general.index")
GENERAL_META_PATH = _path_from_env("GENERAL_META_PATH", default=_default_meta_path(GENERAL_INDEX_PATH, "general_meta.json"))
CATEGORY_INDEX_PATH = _path_from_env(
    "CATEGORY_INDEX_PATH",
    default=os.path.join(os.path.dirname(GENERAL_INDEX_PATH) or ".", "category.index"),
)
CATEGORY_META_PATH = _path_from_env(
    "CATEGORY_META_PATH",
    default=_default_meta_path(CATEGORY_INDEX_PATH, "category_meta.json"),
)
ATTRIBUTE_INDEX_PATH = _path_from_env(
    "ATTRIBUTE_INDEX_PATH",
    default=os.path.join(os.path.dirname(GENERAL_INDEX_PATH) or ".", "attribute.index"),
)
ATTRIBUTE_META_PATH = _path_from_env(
    "ATTRIBUTE_META_PATH",
    default=_default_meta_path(ATTRIBUTE_INDEX_PATH, "attribute_meta.json"),
)
DEFAULT_CHUNK_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_CHUNK_MODEL)
CHUNK_MODEL_NAME = os.getenv("CHUNK_MODEL_NAME", DEFAULT_CHUNK_MODEL)
CATEGORY_MODEL_NAME = os.getenv("CATEGORY_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
ATTRIBUTE_MODEL_NAME = os.getenv("ATTRIBUTE_MODEL_NAME", CATEGORY_MODEL_NAME)
CATEGORY_META_DB_NAME = os.getenv("CATEGORY_META_DB_NAME")
CATEGORY_META_COLLECTION = os.getenv("CATEGORY_META_COLLECTION", "category_faiss_metadata")
ATTRIBUTE_META_DB_NAME = os.getenv("ATTRIBUTE_META_DB_NAME")
ATTRIBUTE_META_COLLECTION = os.getenv("ATTRIBUTE_META_COLLECTION", "attribute_faiss_metadata")
DEFAULT_INDEX_CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE") or 1200)
DEFAULT_INDEX_CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP") or 200)
DEFAULT_INDEX_MIN_CHARS = int(os.getenv("EMBED_MIN_CHARS") or 80)
DEFAULT_WEBSEARCH_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
WEBSEARCH_MODEL_NAME = os.getenv("WEBSEARCH_MODEL") or os.getenv("CHUNK_MODEL_NAME") or DEFAULT_WEBSEARCH_MODEL
WEBSEARCH_INDEX_PATH = _resolve_index_path(os.getenv("WEBSEARCH_INDEX_PATH"), os.path.join(BASE_DIR, "websearch.index"))
REQUIRE_KEYCLOAK_AUTH = (os.getenv("REQUIRE_KEYCLOAK_AUTH") or "true").strip().lower() not in {"0", "false", "no", "off"}
CONTENT_INDEX_PUBLISH_RETRIES = int(os.getenv("CONTENT_INDEX_PUBLISH_RETRIES") or 3)
CONTENT_INDEX_PUBLISH_RETRY_DELAY_SECONDS = float(os.getenv("CONTENT_INDEX_PUBLISH_RETRY_DELAY_SECONDS") or 2.0)
ALLOW_UNAUTH_CONTENT_INDEX = (os.getenv("ALLOW_UNAUTH_CONTENT_INDEX") or "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Ana metin, kategori ve attribute için ayrı FAISS + gerekirse farklı modeller
store = EmbeddingIndex(
    model_name=MODEL_NAME,
    index_path=GENERAL_INDEX_PATH,
    meta_path=GENERAL_META_PATH,
)
category_meta_repo = MetaRepository(db_name=CATEGORY_META_DB_NAME, collection=CATEGORY_META_COLLECTION)
category_store = EmbeddingIndex(
    model_name=CATEGORY_MODEL_NAME,
    index_path=CATEGORY_INDEX_PATH,
    meta_path=CATEGORY_META_PATH,
    meta_repo=category_meta_repo,
)
attribute_meta_repo = MetaRepository(db_name=ATTRIBUTE_META_DB_NAME, collection=ATTRIBUTE_META_COLLECTION)
attribute_store = EmbeddingIndex(
    model_name=ATTRIBUTE_MODEL_NAME,
    index_path=ATTRIBUTE_INDEX_PATH,
    meta_path=ATTRIBUTE_META_PATH,
    meta_repo=attribute_meta_repo,
)
upload_store = UploadStore()
document_loader = DocumentLoader()
content_store = ContentDocumentStore()
job_publisher = RabbitPublisher()
# Worker’la aynı FAISS + Mongo chunk akışı üzerinden arama için
chunk_engine = EmbeddingEngine(model_name=CHUNK_MODEL_NAME, index_path=CHUNK_INDEX_PATH)
chunk_store = MongoStore()


@lru_cache(maxsize=4)
def _get_websearch_engine(model_name: str, index_path: str) -> EmbeddingEngine:
    return EmbeddingEngine(model_name=model_name, index_path=index_path)
def _should_require_auth() -> bool:
    if not REQUIRE_KEYCLOAK_AUTH:
        return False
    if request.method == "OPTIONS":
        return False
    if ALLOW_UNAUTH_CONTENT_INDEX and request.path == "/api/v10/content/index":
        return False
    return True


@app.before_request
def enforce_keycloak_auth():
    if not _should_require_auth():
        return None
    try:
        token_info = get_keycloak_service().validate_bearer_header(request.headers.get("Authorization"))
        g.token_info = token_info
    except KeycloakTokenError as exc:
        return jsonify({"error": "unauthorized", "reason": str(exc)}), 401
    except KeycloakError as exc:
        return jsonify({"error": "keycloak_error", "reason": str(exc)}), 503
    return None


def _get_payload_value(data: dict, *keys: str):
    if not isinstance(data, dict):
        return None
    for key in keys:
        if key in data:
            return data[key]
    lowered = {k.lower(): v for k, v in data.items() if isinstance(k, str)}
    for key in keys:
        lowered_key = key.lower()
        if lowered_key in lowered:
            return lowered[lowered_key]
        stripped = lowered_key.replace("_", "")
        if stripped in lowered:
            return lowered[stripped]
    return None


def _parse_int(value, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _parse_filter_arg(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        if not raw_value.strip():
            return None
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return "__invalid__"
        return parsed
    return None


def _should_use_external_id():
    by = (request.args.get("by") or "").strip().lower()
    if by in {"external_id", "externalid", "external"}:
        return True
    if "external_id" in request.args or "externalId" in request.args:
        return True
    return False


def _resolve_ids_for_identifier(target_store, identifier: str, use_external: bool):
    if use_external or not str(identifier).isdigit():
        ids = target_store.find_ids_by_external_id(str(identifier))
        return ids, "external_id"
    return [int(identifier)], "id"


def _safe_object_id(value):
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _validate_company_user(company_id: str | None, user_id: str | None):
    if not company_id:
        return False, "companyId is required"
    if not user_id:
        return False, "userId is required"

    db = get_database()
    companies = db["companies"]

    company_filters = [{"companyId": str(company_id)}, {"companyid": str(company_id)}, {"id": str(company_id)}]
    oid = _safe_object_id(company_id)
    if oid:
        company_filters.append({"_id": oid})

    query = {
        "$and": [
            {"$or": company_filters},
            {"active": True},
            {"adminActive": True},
        ]
    }
    company = companies.find_one(query, {"_id": 1, "userId": 1, "user_id": 1, "userid": 1, "active": 1, "adminActive": 1})
    if not company:
        return False, "company_not_found_or_inactive"

    company_user = company.get("userId") or company.get("user_id") or company.get("userid")
    if company_user is None:
        return False, "company_user_missing"

    def _match(a, b):
        a_oid = _safe_object_id(a)
        b_oid = _safe_object_id(b)
        if a_oid and b_oid:
            return a_oid == b_oid
        return str(a) == str(b)

    if not _match(company_user, user_id):
        return False, "user_mismatch"

    return True, None


def _normalize_category_payload(payload: dict) -> dict:
    meta_payload = payload.get("metadata") or {}
    if meta_payload and not isinstance(meta_payload, dict):
        raise ValueError("metadata must be an object")

    meta_payload = dict(meta_payload)
    for key in ("companyId", "company_id", "companyid"):
        meta_payload.pop(key, None)

    parent_id = payload.get("parentId") or payload.get("parent_id") or payload.get("parent")
    if parent_id is not None:
        meta_payload["parentId"] = str(parent_id)

    description = payload.get("description") or payload.get("desc")
    if description is not None:
        meta_payload["description"] = str(description)

    company_id = payload.get("companyId") or payload.get("company_id") or payload.get("companyid")
    if company_id is not None:
        payload["companyId"] = str(company_id)

    payload["metadata"] = meta_payload
    return payload


def _map_category_entry_basic(entry: dict | None) -> dict | None:
    if not entry:
        return None
    metadata = entry.get("metadata") or {}
    out = {
        "categoryId": entry.get("external_id"),
        "text": entry.get("text"),
        "companyId": entry.get("companyId"),
        "description": metadata.get("description"),
        "metadata": metadata,
    }
    if entry.get("score") is not None:
        out["score"] = entry.get("score")
    return out


def _get_category_by_external_id(external_id: str | None):
    if not external_id:
        return None
    ids = category_store.find_ids_by_external_id(str(external_id))
    for cid in ids:
        entry = category_store.get_entry(cid)
        if entry:
            return entry
    return None


def _build_category_parent_chain(entry: dict | None) -> list[dict]:
    parents: list[dict] = []
    if not entry:
        return parents
    seen: set[str] = set()
    parent_id = (entry.get("metadata") or {}).get("parentId")
    while parent_id:
        if parent_id in seen:
            break
        seen.add(parent_id)
        parent_entry = _get_category_by_external_id(parent_id)
        mapped = _map_category_entry_basic(parent_entry)
        if not mapped:
            break
        parents.append(mapped)
        parent_id = (parent_entry.get("metadata") or {}).get("parentId")
    # reverse to have root-first ordering
    parents.reverse()
    return parents


def _map_category_entry(entry: dict | None, include_parents: bool = True, include_attributes: bool = False) -> dict | None:
    base = _map_category_entry_basic(entry)
    if not base:
        return None

    if include_parents:
        parents = _build_category_parent_chain(entry)
        if parents:
            base["parents"] = parents
        path_parts = [p.get("text") for p in (parents or []) if p.get("text")]
        if entry and entry.get("text"):
            path_parts.append(entry.get("text"))
        if path_parts:
            base["parentPath"] = ".".join(path_parts)

    if include_attributes:
        base["attributes"] = _fetch_attributes_for_category(base.get("categoryId"), base.get("companyId"), entry)

    return base


def _fetch_attributes_for_category(category_external_id: str | None, company_id: str | None, entry: dict | None = None):
    """
    Fetch attributes for a category, including attributes from all parent categories.
    Returns combined list: parent attributes first (root -> descendants), then current category's attributes.
    """
    if not category_external_id:
        return []
    
    # Collect all category IDs to fetch attributes for (parents + current)
    category_ids = []
    
    # If entry is provided, use parent chain
    if entry:
        parents = _build_category_parent_chain(entry)
        for parent in parents:
            parent_id = parent.get("categoryId")
            if parent_id:
                category_ids.append(str(parent_id))
    
    # Add current category at the end
    category_ids.append(str(category_external_id))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_category_ids = []
    for cid in category_ids:
        if cid not in seen:
            seen.add(cid)
            unique_category_ids.append(cid)
    
    # Fetch attributes for all categories
    all_attributes = []
    seen_attribute_codes = set()
    
    for cat_id in unique_category_ids:
        filt = {"metadata.categoryId": str(cat_id)}
        if company_id:
            filt["companyId"] = str(company_id)
        total, items = attribute_store.list_entries(limit=None, simple_filter=filt)
        for item in items:
            mapped = _map_attribute_entry(item)
            if mapped:
                # Use code as unique identifier to avoid duplicates
                code = mapped.get("code") or mapped.get("text")
                if code and code not in seen_attribute_codes:
                    seen_attribute_codes.add(code)
                    all_attributes.append(mapped)
    
    return all_attributes



def _normalize_attribute_payload(payload: dict) -> dict:
    meta_payload = payload.get("metadata") or {}
    if meta_payload and not isinstance(meta_payload, dict):
        raise ValueError("metadata must be an object")

    meta_payload = dict(meta_payload)
    for key in ("companyId", "company_id", "companyid"):
        meta_payload.pop(key, None)

    category_id = payload.get("categoryId") or payload.get("category_id") or payload.get("category")
    if category_id is not None:
        meta_payload["categoryId"] = str(category_id)

    company_id = payload.get("companyId") or payload.get("company_id") or payload.get("companyid")
    if company_id is not None:
        payload["companyId"] = str(company_id)

    payload["metadata"] = meta_payload
    return payload


def _map_attribute_entry(entry: dict | None) -> dict | None:
    if not entry:
        return None
    out = {
        "attributeId": entry.get("external_id"),
        "text": entry.get("text"),
        "companyId": entry.get("companyId"),
        "metadata": entry.get("metadata") or {},
    }
    if entry.get("score") is not None:
        out["score"] = entry.get("score")
    return out


def _resolve_upload_s3_location(upload_doc, file_doc):
    bucket_default = os.getenv("AWS_S3_BUCKET")
    candidates = []
    if isinstance(upload_doc, dict):
        candidates.append(upload_doc.get("file") or {})
        candidates.append(upload_doc.get("data") or {})
    if isinstance(file_doc, dict):
        candidates.append(file_doc)

    for blob in candidates:
        if not isinstance(blob, dict):
            continue
        bucket = (
            blob.get("bucket")
            or blob.get("Bucket")
            or blob.get("bucket_name")
            or bucket_default
        )
        key = blob.get("key") or blob.get("Key") or blob.get("path") or blob.get("s3Key")
        if key:
            filename = blob.get("filename") or blob.get("originalname") or os.path.basename(key)
            return bucket, key, filename

    if not bucket_default:
        raise RuntimeError("Unable to determine S3 bucket and AWS_S3_BUCKET is not set.")
    raise RuntimeError("Unable to resolve S3 object key for upload.")


def _vector_upsert_response(
    target_store,
    label: str = "upsert",
    payload: dict | None = None,
    transform=None,
):
    try:
        payload = payload if payload is not None else (request.get_json() or {})
        text = payload.get("text")
        vector = payload.get("vector")
        external_id = payload.get("external_id")
        metadata = payload.get("metadata") or {}
        if metadata and not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        company_id = payload.get("companyId") or payload.get("company_id") or payload.get("companyid")
        metadata = dict(metadata or {})
        int_id = payload.get("id")
        out = target_store.upsert_vector(text, vector, external_id, metadata, int_id, company_id)
        if transform:
            out = transform(out)
        return jsonify(out)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"{label} failed: {exc}"}), 400


def _vector_search_response(target_store, label: str = "search", payload: dict | None = None, transform=None):
    try:
        payload = payload if payload is not None else (request.get_json() or {})
        text = payload.get("text")
        vector = payload.get("vector")
        k = int(payload.get("k") or 5)
        filt = payload.get("filter") or {}
        results = target_store.search(text, vector, k, filt)
        if transform:
            transformed = []
            for item in results:
                mapped = transform(item)
                if mapped:
                    transformed.append(mapped)
            results = transformed
        return jsonify({"results": results})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"{label} failed: {exc}"}), 400


def _passes_chunk_filters(chunk: dict, filters: dict) -> bool:
    for key, val in (filters or {}).items():
        if key.startswith("metadata."):
            sub = key.split(".", 1)[1]
            if (chunk.get("metadata") or {}).get(sub) != val:
                return False
        else:
            if chunk.get(key) != val:
                return False
    return True


def _reconstruct_from_chunks(chunks: list[dict]) -> str:
    """
    Rebuild text from overlapping chunks using their char_start offsets.
    Pads gaps with spaces; trims overlapping prefix from each subsequent chunk.
    """
    if not chunks:
        return ""
    ordered = sorted(
        chunks,
        key=lambda c: (
            int(c.get("char_start") or c.get("chunk_index") or 0),
            int(c.get("chunk_index") or 0),
        ),
    )
    buffer: list[str] = []
    current_len = 0
    for chunk in ordered:
        text = chunk.get("text") or ""
        start = int(chunk.get("char_start") or current_len)
        if start > current_len:
            buffer.append(" " * (start - current_len))
            current_len += start - current_len
        overlap = current_len - start
        if overlap < len(text):
            buffer.append(text[overlap:])
            current_len += len(text) - overlap
    return "".join(buffer)


def _chunk_search_response(payload: dict):
    k = int(payload.get("k") or 5)
    if k <= 0:
        return jsonify({"error": "k must be positive"}), 400

    vector = payload.get("vector")
    if vector is None:
        text = payload.get("text")
        if not text or not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Provide either 'text' or 'vector'."}), 400
        query_vec = chunk_engine.encode([text], batch_size=1)
    else:
        arr = np.array(vector, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            return jsonify({"error": "vector must be 1D or 2D array"}), 400
        query_vec = arr.astype(np.float32, copy=False)

    filters = payload.get("filter") or {}
    try:
        scores, ids = chunk_engine.search(query_vec, k)
    except RuntimeError:
        return jsonify({"results": []})
    except ValueError as exc:
        index_obj = getattr(chunk_engine, "_index", None)
        index_dim = getattr(index_obj, "d", None)
        return (
            jsonify(
                {
                    "error": str(exc),
                    "hint": "Embedding dimension mismatch: ensure CHUNK_MODEL_NAME matches the model used to build the index.",
                    "index_path": CHUNK_INDEX_PATH,
                    "model_name": getattr(chunk_engine, "model_name", None),
                    "index_dim": index_dim,
                    "query_dim": int(query_vec.shape[1]),
                }
            ),
            400,
        )

    faiss_ids = [int(i) for i in ids[0] if i != -1]
    chunk_docs = chunk_store.get_chunks_by_faiss_ids(faiss_ids)
    doc_status_map = chunk_store.get_documents_by_ids({c.get("doc_id") for c in chunk_docs.values() if c})
    radius = int(payload.get("radius") or 0)
    if radius < 0:
        return jsonify({"error": "radius must be >= 0"}), 400
    doc_chunk_cache: dict[str, list[dict]] = {}

    results = []
    for idx, score in zip(ids[0], scores[0]):
        if idx == -1:
            continue
        chunk = chunk_docs.get(int(idx))
        if not chunk:
            continue
        doc_info = doc_status_map.get(chunk.get("doc_id"))
        if doc_info and str(doc_info.get("status")).lower() in {"disabled", "removed"}:
            continue
        if not _passes_chunk_filters(chunk, filters):
            continue
        combined_text = chunk.get("text")
        if radius > 0:
            doc_id = str(chunk.get("doc_id") or "")
            chunk_index = chunk.get("chunk_index") if "chunk_index" in chunk else chunk.get("index")
            if doc_id and chunk_index is not None:
                if doc_id not in doc_chunk_cache:
                    doc_chunk_cache[doc_id] = chunk_store.get_chunks_by_doc(doc_id)
                all_chunks = doc_chunk_cache.get(doc_id) or []
                try:
                    target_idx = next(
                        i for i, c in enumerate(all_chunks) if int(c.get("chunk_index", -1)) == int(chunk_index)
                    )
                    start = max(0, target_idx - radius)
                    end = min(len(all_chunks), target_idx + radius + 1)
                    combined_text = _reconstruct_from_chunks(all_chunks[start:end])
                except StopIteration:
                    combined_text = chunk.get("text")
        results.append(
            {
                "type": "chunk",
                "id": int(idx),
                "score": float(score),
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "text": combined_text,
                "metadata": chunk.get("metadata") or {},
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
                "doc_type": chunk.get("doc_type"),
                "source": chunk.get("source"),
                "radius": radius,
            }
        )
    return jsonify({"results": results})


def _websearch_faiss_search(query: str, k: int, model_name: str, index_path: str):
    engine = _get_websearch_engine(model_name, os.path.abspath(index_path))
    query_vec = engine.encode([query], batch_size=1)
    scores, ids = engine.search(query_vec, int(k))
    return scores[0].tolist(), [int(i) for i in ids[0]]


def _short_text_for_log(text: str | None, max_len: int = 160) -> str:
    if not text:
        return ""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 3] + "..."


def _deactivate_index_state(company_id: str | None, document_id: str, state: str) -> None:
    if not company_id:
        return
    try:
        content_store.update_index_fields(
            company_id=company_id,
            document_id=document_id,
            state=state,
            error=None,
        )
    except Exception:
        # Avoid blocking the API on index state updates
        pass


@app.route("/", methods=["GET"])
def root():
    """Health check: returns basic status text and current FAISS index size."""
    general_size = int(store.index.ntotal) if store.index is not None else 0
    try:
        chunk_size = chunk_engine.count()
    except Exception:
        chunk_size = None
    try:
        chunk_index_dim = chunk_engine.index_dimension()
        chunk_model_dim = chunk_engine.model_dimension()
    except Exception:
        chunk_index_dim = None
        chunk_model_dim = None
    return jsonify(
        {
            "message": "tinnten-embedding up",
            "index_size": general_size,
            "chunk_index_size": chunk_size,
            "chunk_index_path": CHUNK_INDEX_PATH,
            "chunk_model_name": CHUNK_MODEL_NAME,
            "chunk_index_dim": chunk_index_dim,
            "chunk_model_dim": chunk_model_dim,
        }
    )

@app.route("/api/v10/llm/vector", methods=["POST"])
def generate_vector():
    """
    Encode raw text into an embedding vector.

    JSON body:
        text (str, required): Content to vectorize.
    """
    try:
        data = request.get_json() or {}
        text = data.get("text")
        vec = store.vectorize_text(text)
        return jsonify({"vector": vec})
    except Exception as e:
        return jsonify({"error": f"vectorization failed: {str(e)}"}), 400

@app.route("/api/v10/vector/upsert", methods=["POST"])
def upsert_vector():
    """
    Insert or update a single vector in FAISS.

    JSON body:
        text (str, optional): Will be encoded if vector not provided.
        vector (list[float], optional): Raw vector to upsert.
        external_id (str, optional): Reference identifier to store with metadata.
        metadata (dict, optional): Arbitrary metadata payload.
        id (int | str, optional): Existing FAISS ID to overwrite; accepts custom text keys too.
    """
    return _vector_upsert_response(store, label="vector upsert")

@app.route("/api/v10/vector/search", methods=["POST"])
def search_vector():
    """
    Run a similarity search against the FAISS index.

    JSON body:
        type (str, optional): "chunk" (default), "category", "attribute", or "vector"
        text / vector (optional): Query input; vector skips encoding
        k (int, optional): Number of results to return (default 5).
        filter (dict, optional): Metadata filters applied to stored entries.
    """
    payload = request.get_json() or {}
    target = (payload.get("type") or payload.get("target") or "chunk").strip().lower()

    if target in {"chunk", "chunks", "content"}:
        return _chunk_search_response(payload)
    if target in {"category", "categories"}:
        return _vector_search_response(
            category_store,
            label="category search",
            payload=payload,
            transform=_map_category_entry,
        )
    if target in {"attribute", "attributes"}:
        return _vector_search_response(
            attribute_store,
            label="attribute search",
            payload=payload,
            transform=_map_attribute_entry,
        )

    # Fallback to legacy/vector search against the general index
    return _vector_search_response(store, label="vector search", payload=payload)


@app.route("/api/v10/websearch/search", methods=["POST"])
def search_web_index():
    """
    Run a similarity search against the websearch FAISS index built from tur_Latn.
    """
    payload = request.get_json() or {}
    text = (payload.get("text") or payload.get("query") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    k = int(payload.get("k") or 5)
    if k <= 0:
        return jsonify({"error": "k must be positive"}), 400
    radius = int(payload.get("radius") or 0)
    if radius < 0:
        return jsonify({"error": "radius must be >= 0"}), 400

    filters = payload.get("filter") or {}
    if filters and not isinstance(filters, dict):
        return jsonify({"error": "filter must be an object"}), 400

    model_name = payload.get("model_name") or payload.get("modelName") or WEBSEARCH_MODEL_NAME
    index_path = _resolve_index_path(payload.get("index_path") or payload.get("indexPath"), WEBSEARCH_INDEX_PATH)
    if not os.path.exists(index_path):
        return jsonify({"error": "websearch index not found", "index_path": index_path}), 400

    try:
        scores, ids = _websearch_faiss_search(text, k, model_name, index_path)
        app.logger.info("Websearch: text='%s' k=%d model='%s' index='%s' -> found %d IDs", text, k, model_name, index_path, len(ids))
    except Exception as exc:  # noqa: BLE001
        app.logger.error("Websearch failed: %s", exc)
        return jsonify({"error": f"websearch failed: {exc}"}), 400

    faiss_ids = [int(i) for i in ids if i != -1]
    chunk_docs = chunk_store.get_chunks_by_faiss_ids(faiss_ids)
    doc_status_map = chunk_store.get_documents_by_ids({c.get("doc_id") for c in chunk_docs.values() if c})
    doc_chunk_cache: dict[str, list[dict]] = {}

    results = []
    for idx, score in zip(ids, scores):
        if idx == -1:
            continue
        chunk = chunk_docs.get(int(idx))
        if not chunk:
            app.logger.warning("Websearch: Chunk not found for FAISS ID %d", idx)
            continue
        doc_info = doc_status_map.get(chunk.get("doc_id"))
        status = str(doc_info.get("status") if doc_info else "unknown").lower()
        if doc_info and status in {"disabled", "removed"}:
            app.logger.info("Websearch: Skipping chunk %d (doc_id=%s status=%s)", idx, chunk.get("doc_id"), status)
            continue
        if filters and not _passes_chunk_filters(chunk, filters):
            app.logger.info("Websearch: Skipping chunk %d (doc_id=%s) due to filters (metadata=%s)", idx, chunk.get("doc_id"), chunk.get("metadata"))
            continue

        combined_text = chunk.get("text")
        if radius > 0:
            doc_id = str(chunk.get("doc_id") or "")
            chunk_index = chunk.get("chunk_index") if "chunk_index" in chunk else chunk.get("index")
            if doc_id and chunk_index is not None:
                if doc_id not in doc_chunk_cache:
                    doc_chunk_cache[doc_id] = chunk_store.get_chunks_by_doc(doc_id)
                all_chunks = doc_chunk_cache.get(doc_id) or []
                try:
                    target_idx = next(
                        i for i, c in enumerate(all_chunks) if int(c.get("chunk_index", -1)) == int(chunk_index)
                    )
                    start = max(0, target_idx - radius)
                    end = min(len(all_chunks), target_idx + radius + 1)
                    combined_text = _reconstruct_from_chunks(all_chunks[start:end])
                except StopIteration:
                    combined_text = chunk.get("text")

        results.append(
            {
                "type": "websearch",
                "id": int(idx),
                "score": float(score),
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "text": combined_text,
                "metadata": chunk.get("metadata") or {},
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
                "doc_type": chunk.get("doc_type"),
                "source": chunk.get("source"),
                "radius": radius,
            }
        )

    if results:
        log_items = []
        for item in results[:5]:
            text_snippet = _short_text_for_log(item.get("text"))
            log_items.append(
                {
                    "id": item.get("id"),
                    "score": float(item.get("score", 0)),
                    "doc_id": item.get("doc_id"),
                    "source": item.get("source"),
                    "text": text_snippet,
                }
            )
        logger.info(
            "websearch response count=%s index=%s model=%s items=%s",
            len(results),
            index_path,
            model_name,
            log_items,
        )

    return jsonify({"results": results})


@app.route("/api/v10/categories/upsert", methods=["POST"])
def upsert_category_vector():
    """
    Insert or update a category vector in its dedicated FAISS index.

    JSON body:
        parentId (str, optional): Parent category identifier; stored under metadata.parentId.
        companyId (str, required): Firma kimliği; metadata yerine üst seviyede tutulur.
        userId (str, required): Şirket sahibi kullanıcı.
    """
    payload = request.get_json() or {}
    company_id = payload.get("companyId") or payload.get("company_id") or payload.get("companyid")
    user_id = payload.get("userId") or payload.get("user_id") or payload.get("userid")
    if not company_id:
        return jsonify({"error": "companyId is required"}), 400
    ok, reason = _validate_company_user(str(company_id), str(user_id) if user_id is not None else None)
    if not ok:
        status = 400 if reason in {"companyId is required", "userId is required"} else 403
        return jsonify({"error": reason}), status
    try:
        payload = _normalize_category_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return _vector_upsert_response(
        category_store,
        label="category upsert",
        payload=payload,
        transform=lambda out: {"categoryId": out.get("external_id")},
    )


@app.route("/api/v10/categories/search", methods=["POST"])
def search_category_vector():
    """
    Run a similarity search on the category FAISS index.
    """
    return _vector_search_response(category_store, label="category search", transform=_map_category_entry)

@app.route("/api/v10/categories", methods=["GET"])
@app.route("/categories", methods=["GET"])
@app.route("/embedding/categories", methods=["GET"])
def list_categories():
    limit = _parse_int(request.args.get("limit"), 100, minimum=0)
    offset = _parse_int(request.args.get("offset"), 0, minimum=0)
    filter_arg = _parse_filter_arg(request.args.get("filter"))
    if filter_arg == "__invalid__":
        return jsonify({"error": "filter must be valid JSON"}), 400
    if filter_arg is not None and not isinstance(filter_arg, dict):
        return jsonify({"error": "filter must be an object"}), 400

    external_id = request.args.get("external_id") or request.args.get("externalId")
    if external_id is not None:
        filter_arg = dict(filter_arg or {})
        filter_arg["external_id"] = str(external_id)

    text = request.args.get("text")
    if text is not None:
        filter_arg = dict(filter_arg or {})
        filter_arg["text"] = str(text)

    # companyId filter desteği
    company_id = request.args.get("companyId") or request.args.get("company_id") or request.args.get("companyid")
    if company_id is not None:
        filter_arg = dict(filter_arg or {})
        filter_arg["companyId"] = str(company_id)
    # eski format filtre anahtarı gelirse normalize et
    if filter_arg and "metadata.companyId" in filter_arg:
        filter_arg["companyId"] = str(filter_arg.pop("metadata.companyId"))

    # parentId filter desteği (alt kategorileri getirmek için)
    parent_id = request.args.get("parentId") or request.args.get("parent_id") or request.args.get("parentid")
    if parent_id is not None:
        filter_arg = dict(filter_arg or {})
        filter_arg["parentId"] = str(parent_id)
    # eski format filtre anahtarı gelirse normalize et
    if filter_arg and "metadata.parentId" in filter_arg:
        filter_arg["parentId"] = str(filter_arg.pop("metadata.parentId"))

    total, items = category_store.list_entries(
        offset=offset,
        limit=limit,
        simple_filter=filter_arg,
    )
    mapped = [_map_category_entry(item) for item in items]
    mapped = [m for m in mapped if m]
    return jsonify({"total": total, "offset": offset, "limit": limit, "items": mapped})

@app.route("/api/v10/categories", methods=["POST"])
@app.route("/categories", methods=["POST"])
@app.route("/embedding/categories", methods=["POST"])
def create_category():
    payload = request.get_json() or {}
    
    # companyId zorunluluğu
    company_id = payload.get("companyId") or payload.get("company_id") or payload.get("companyid")
    user_id = payload.get("userId") or payload.get("user_id") or payload.get("userid")
    if not company_id:
        return jsonify({"error": "companyId is required"}), 400
    ok, reason = _validate_company_user(str(company_id), str(user_id) if user_id is not None else None)
    if not ok:
        status = 400 if reason in {"companyId is required", "userId is required"} else 403
        return jsonify({"error": reason}), status
    
    try:
        payload = _normalize_category_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    text = payload.get("text")
    if text:
        # Check for duplicates by exact text match within the same company
        duplicate_filter = {
            "text": text,
            "companyId": str(company_id)
        }
        total, exists = category_store.list_entries(limit=1, simple_filter=duplicate_filter)
        if total > 0:
            return jsonify({
                "error": "duplicate_entry",
                "message": f"Category with name '{text}' already exists in this company.",
                "existing_categoryId": exists[0].get("external_id")
            }), 409
    return _vector_upsert_response(
        category_store,
        label="category create",
        payload=payload,
        transform=lambda out: {"categoryId": out.get("external_id")},
    )

@app.route("/api/v10/categories/<category_id>", methods=["GET"])
@app.route("/categories/<category_id>", methods=["GET"])
@app.route("/embedding/categories/<category_id>", methods=["GET"])
def get_category(category_id: str):
    if not category_id:
        return jsonify({"error": "category_id is required"}), 400

    use_external = _should_use_external_id()
    ids, id_kind = _resolve_ids_for_identifier(category_store, category_id, use_external)
    if not ids:
        return jsonify({"error": "not_found"}), 404

    if id_kind == "external_id":
        entries = [category_store.get_entry(fid) for fid in ids]
        entries = [entry for entry in entries if entry]
        if len(entries) == 1:
            mapped = _map_category_entry(entries[0], include_attributes=True)
            return jsonify(mapped)
        mapped = []
        for e in entries:
            obj = _map_category_entry(e, include_attributes=True)
            if obj:
                mapped.append(obj)
        return jsonify({"items": mapped, "count": len(mapped)})

    entry = category_store.get_entry(ids[0])
    if not entry:
        return jsonify({"error": "not_found"}), 404
    mapped_entry = _map_category_entry(entry, include_attributes=True)
    return jsonify(mapped_entry)

@app.route("/api/v10/categories/<category_id>", methods=["PUT", "PATCH"])
@app.route("/categories/<category_id>", methods=["PUT", "PATCH"])
@app.route("/embedding/categories/<category_id>", methods=["PUT", "PATCH"])
def update_category(category_id: str):
    if not category_id:
        return jsonify({"error": "category_id is required"}), 400

    use_external = _should_use_external_id()
    ids, id_kind = _resolve_ids_for_identifier(category_store, category_id, use_external)
    if not ids:
        return jsonify({"error": "not_found"}), 404
    if id_kind == "external_id" and len(ids) > 1:
        return jsonify({"error": "multiple_matches", "count": len(ids)}), 409

    resolved_id = ids[0]
    existing_entry = category_store.get_entry(resolved_id)
    if not existing_entry:
        return jsonify({"error": "not_found"}), 404

    payload = request.get_json() or {}
    
    # companyId validasyonu
    company_id = payload.get("companyId") or payload.get("company_id") or payload.get("companyid")
    user_id = payload.get("userId") or payload.get("user_id") or payload.get("userid")
    existing_company_id = existing_entry.get("companyId")
    
    # Eğer mevcut kategorinin companyId'si varsa, gönderilen companyId eşleşmeli
    if existing_company_id and company_id and str(existing_company_id) != str(company_id):
        return jsonify({"error": "forbidden", "message": "Bu kategoriyi düzenleme yetkiniz yok."}), 403
    target_company_id = company_id or existing_company_id
    ok, reason = _validate_company_user(str(target_company_id) if target_company_id is not None else None, str(user_id) if user_id is not None else None)
    if not ok:
        status = 400 if reason in {"companyId is required", "userId is required"} else 403
        return jsonify({"error": reason}), status
    
    payload["id"] = payload.get("id") or resolved_id
    try:
        payload = _normalize_category_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return _vector_upsert_response(
        category_store,
        label="category update",
        payload=payload,
        transform=lambda out: {"categoryId": out.get("external_id")},
    )

@app.route("/api/v10/categories/<category_id>", methods=["DELETE"])
@app.route("/categories/<category_id>", methods=["DELETE"])
@app.route("/embedding/categories/<category_id>", methods=["DELETE"])
def delete_category(category_id: str):
    if not category_id:
        return jsonify({"error": "category_id is required"}), 400

    # companyId validasyonu (query parameter veya header)
    company_id = request.args.get("companyId") or request.args.get("company_id") or request.args.get("companyid")

    use_external = _should_use_external_id()
    ids, id_kind = _resolve_ids_for_identifier(category_store, category_id, use_external)
    if not ids:
        return jsonify({"error": "not_found"}), 404
    
    external_ids: list[str | None] = []
    # Her ID için companyId kontrolü
    if company_id:
        for fid in ids:
            entry = category_store.get_entry(fid)
            if entry:
                existing_company_id = entry.get("companyId")
                if existing_company_id and str(existing_company_id) != str(company_id):
                    return jsonify({"error": "forbidden", "message": "Bu kategoriyi silme yetkiniz yok."}), 403
                external_ids.append(entry.get("external_id"))
            else:
                external_ids.append(None)
    else:
        for fid in ids:
            entry = category_store.get_entry(fid)
            external_ids.append((entry or {}).get("external_id"))
    
    try:
        removed = category_store.delete_by_ids(ids)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    clean_external = [eid for eid in external_ids if eid is not None]
    return jsonify({"removed": removed, "categoryIds": clean_external, "identifierType": id_kind})


@app.route("/api/v10/attributes/upsert", methods=["POST"])
def upsert_attribute_vector():
    """
    Insert or update an attribute vector in its dedicated FAISS index.

    JSON body:
        categoryId (str, optional): Link attribute to a category; stored under metadata.categoryId.
        companyId (str, required): Firma kimliği; metadata yerine üst seviyede tutulur.
        userId (str, required): Şirket sahibi kullanıcı.
    """
    payload = request.get_json() or {}
    company_id = payload.get("companyId") or payload.get("company_id") or payload.get("companyid")
    user_id = payload.get("userId") or payload.get("user_id") or payload.get("userid")
    if not company_id:
        return jsonify({"error": "companyId is required"}), 400
    ok, reason = _validate_company_user(str(company_id), str(user_id) if user_id is not None else None)
    if not ok:
        status = 400 if reason in {"companyId is required", "userId is required"} else 403
        return jsonify({"error": reason}), status
    try:
        payload = _normalize_attribute_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return _vector_upsert_response(
        attribute_store,
        label="attribute upsert",
        payload=payload,
        transform=lambda out: {"attributeId": out.get("external_id")},
    )


@app.route("/api/v10/attributes/search", methods=["POST"])
def search_attribute_vector():
    """
    Run a similarity search on the attribute FAISS index.
    """
    return _vector_search_response(attribute_store, label="attribute search", transform=_map_attribute_entry)

@app.route("/api/v10/attributes", methods=["GET"])
def list_attributes():
    limit = _parse_int(request.args.get("limit"), 100, minimum=0)
    offset = _parse_int(request.args.get("offset"), 0, minimum=0)
    filter_arg = _parse_filter_arg(request.args.get("filter"))
    if filter_arg == "__invalid__":
        return jsonify({"error": "filter must be valid JSON"}), 400
    if filter_arg is not None and not isinstance(filter_arg, dict):
        return jsonify({"error": "filter must be an object"}), 400

    external_id = request.args.get("external_id") or request.args.get("externalId")
    if external_id is not None:
        filter_arg = dict(filter_arg or {})
        filter_arg["external_id"] = str(external_id)

    text = request.args.get("text")
    if text is not None:
        filter_arg = dict(filter_arg or {})
        filter_arg["text"] = str(text)

    # companyId filter desteği
    company_id = request.args.get("companyId") or request.args.get("company_id") or request.args.get("companyid")
    if company_id is not None:
        filter_arg = dict(filter_arg or {})
        filter_arg["companyId"] = str(company_id)

    # categoryId filter desteği
    category_id = request.args.get("categoryId") or request.args.get("category_id") or request.args.get("categoryid")
    if category_id is not None:
        filter_arg = dict(filter_arg or {})
        filter_arg["metadata.categoryId"] = str(category_id)

    # eski format filtre anahtarı gelirse normalize et
    if filter_arg and "metadata.companyId" in filter_arg:
        filter_arg["companyId"] = str(filter_arg.pop("metadata.companyId"))

    total, items = attribute_store.list_entries(
        offset=offset,
        limit=limit,
        simple_filter=filter_arg,
    )
    mapped = [_map_attribute_entry(item) for item in items]
    mapped = [m for m in mapped if m]
    return jsonify({"total": total, "offset": offset, "limit": limit, "items": mapped})

@app.route("/api/v10/attributes", methods=["POST"])
def create_attribute():
    payload = request.get_json() or {}
    
    # companyId zorunluluğu
    company_id = payload.get("companyId") or payload.get("company_id") or payload.get("companyid")
    user_id = payload.get("userId") or payload.get("user_id") or payload.get("userid")
    if not company_id:
        return jsonify({"error": "companyId is required"}), 400
    ok, reason = _validate_company_user(str(company_id), str(user_id) if user_id is not None else None)
    if not ok:
        status = 400 if reason in {"companyId is required", "userId is required"} else 403
        return jsonify({"error": reason}), status
    
    try:
        payload = _normalize_attribute_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    text = payload.get("text")
    if text:
        # Check for duplicates by exact text match within the same company
        duplicate_filter = {
            "text": text,
            "companyId": str(company_id)
        }
        total, exists = attribute_store.list_entries(limit=1, simple_filter=duplicate_filter)
        if total > 0:
            return jsonify({
                "error": "duplicate_entry",
                "message": f"Attribute with name '{text}' already exists in this company.",
                "existing_attributeId": exists[0].get("external_id")
            }), 409

    return _vector_upsert_response(
        attribute_store,
        label="attribute create",
        payload=payload,
        transform=lambda out: {"attributeId": out.get("external_id")},
    )

@app.route("/api/v10/attributes/<attribute_id>", methods=["GET"])
def get_attribute(attribute_id: str):
    if not attribute_id:
        return jsonify({"error": "attribute_id is required"}), 400

    use_external = _should_use_external_id()
    ids, id_kind = _resolve_ids_for_identifier(attribute_store, attribute_id, use_external)
    if not ids:
        return jsonify({"error": "not_found"}), 404

    if id_kind == "external_id":
        entries = [attribute_store.get_entry(fid) for fid in ids]
        entries = [entry for entry in entries if entry]
        if len(entries) == 1:
            return jsonify(_map_attribute_entry(entries[0]))
        mapped = [_map_attribute_entry(e) for e in entries]
        mapped = [m for m in mapped if m]
        return jsonify({"items": mapped, "count": len(mapped)})

    entry = attribute_store.get_entry(ids[0])
    if not entry:
        return jsonify({"error": "not_found"}), 404
    return jsonify(_map_attribute_entry(entry))

@app.route("/api/v10/attributes/<attribute_id>", methods=["PUT", "PATCH"])
def update_attribute(attribute_id: str):
    if not attribute_id:
        return jsonify({"error": "attribute_id is required"}), 400

    use_external = _should_use_external_id()
    ids, id_kind = _resolve_ids_for_identifier(attribute_store, attribute_id, use_external)
    if not ids:
        return jsonify({"error": "not_found"}), 404
    if id_kind == "external_id" and len(ids) > 1:
        return jsonify({"error": "multiple_matches", "count": len(ids)}), 409

    resolved_id = ids[0]
    existing_entry = attribute_store.get_entry(resolved_id)
    if not existing_entry:
        return jsonify({"error": "not_found"}), 404

    payload = request.get_json() or {}
    
    # companyId validasyonu
    company_id = payload.get("companyId") or payload.get("company_id") or payload.get("companyid")
    user_id = payload.get("userId") or payload.get("user_id") or payload.get("userid")
    existing_company_id = existing_entry.get("companyId")
    
    # Eğer mevcut attribute'un companyId'si varsa, gönderilen companyId eşleşmeli
    if existing_company_id and company_id and str(existing_company_id) != str(company_id):
        return jsonify({"error": "forbidden", "message": "Bu attribute'ü düzenleme yetkiniz yok."}), 403
    target_company_id = company_id or existing_company_id
    ok, reason = _validate_company_user(str(target_company_id) if target_company_id is not None else None, str(user_id) if user_id is not None else None)
    if not ok:
        status = 400 if reason in {"companyId is required", "userId is required"} else 403
        return jsonify({"error": reason}), status
    
    payload["id"] = payload.get("id") or resolved_id
    try:
        payload = _normalize_attribute_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return _vector_upsert_response(
        attribute_store,
        label="attribute update",
        payload=payload,
        transform=lambda out: {"attributeId": out.get("external_id")},
    )

@app.route("/api/v10/attributes/<attribute_id>", methods=["DELETE"])
def delete_attribute(attribute_id: str):
    if not attribute_id:
        return jsonify({"error": "attribute_id is required"}), 400

    # companyId validasyonu (query parameter)
    company_id = request.args.get("companyId") or request.args.get("company_id") or request.args.get("companyid")

    use_external = _should_use_external_id()
    ids, id_kind = _resolve_ids_for_identifier(attribute_store, attribute_id, use_external)
    if not ids:
        return jsonify({"error": "not_found"}), 404
    
    external_ids: list[str | None] = []
    # Her ID için companyId kontrolü
    if company_id:
        for fid in ids:
            entry = attribute_store.get_entry(fid)
            if entry:
                existing_company_id = entry.get("companyId")
                if existing_company_id and str(existing_company_id) != str(company_id):
                    return jsonify({"error": "forbidden", "message": "Bu attribute'ü silme yetkiniz yok."}), 403
                external_ids.append(entry.get("external_id"))
            else:
                external_ids.append(None)
    else:
        for fid in ids:
            entry = attribute_store.get_entry(fid)
            external_ids.append((entry or {}).get("external_id"))
    
    try:
        removed = attribute_store.delete_by_ids(ids)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    clean_external = [eid for eid in external_ids if eid is not None]
    return jsonify({"removed": removed, "attributeIds": clean_external, "identifierType": id_kind})


@app.route("/api/v10/content/search", methods=["POST"])
def search_content_chunks():
    """
    Run a similarity search on the chunk/content FAISS index created via `/api/v10/content/index`.

    JSON body:
        text / vector (optional): Query input; vector skips encoding
        k (int, optional): Number of results to return (default 5).
        filter (dict, optional): Metadata filters applied to stored chunks.
        radius (int, optional): Reconstruct surrounding chunks for more context.
    """
    payload = request.get_json() or {}
    return _chunk_search_response(payload)


@app.route("/api/v10/content/index/deactivate", methods=["POST"])
def deactivate_document():
    """
    Soft-disable a document's embeddings without touching FAISS.

    JSON body:
        documentId (str, required)
        companyId (str, optional): Used to update contentdocuments.index.state
    """
    payload = request.get_json() or {}
    document_id = _get_payload_value(payload, "documentId", "document_id", "documentid")
    company_id = _get_payload_value(payload, "companyId", "company_id", "companyid")
    if not document_id:
        return jsonify({"error": "documentId is required"}), 400

    # Mark embedding_documents as disabled
    try:
        chunk_store.update_document_status(str(document_id), status="disabled")
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"failed to update document status: {exc}"}), 500

    _deactivate_index_state(company_id, str(document_id), state="disabled")
    return jsonify({"documentId": str(document_id), "state": "disabled"})


@app.route("/api/v10/content/document/<doc_id>/reconstruct", methods=["GET"])
def reconstruct_document(doc_id: str):
    """
    Rebuild and return the concatenated text for a document from its stored chunks.
    """
    doc_info = chunk_store.get_documents_by_ids([doc_id]).get(doc_id)
    if doc_info and str(doc_info.get("status")).lower() in {"disabled", "removed"}:
        return jsonify({"error": "document_disabled", "documentId": doc_id, "status": doc_info.get("status")}), 409

    chunks = chunk_store.get_chunks_by_doc(doc_id)
    if not chunks:
        return jsonify({"error": "not_found", "documentId": doc_id}), 404

    text = _reconstruct_from_chunks(chunks)
    return jsonify(
        {
            "documentId": doc_id,
            "status": (doc_info or {}).get("status"),
            "chunkCount": len(chunks),
            "text": text,
        }
    )


@app.route("/api/v10/content/index/remove", methods=["POST"])
def remove_document():
    """
    Hard-delete a document's embeddings from FAISS and Mongo.

    JSON body:
        documentId (str, required)
        companyId (str, optional): Used to update contentdocuments.index.state
    """
    payload = request.get_json() or {}
    document_id = _get_payload_value(payload, "documentId", "document_id", "documentid")
    company_id = _get_payload_value(payload, "companyId", "company_id", "companyid")
    if not document_id:
        return jsonify({"error": "documentId is required"}), 400

    doc_id_str = str(document_id)
    chunks = chunk_store.get_chunks_by_doc(doc_id_str)
    faiss_ids = sorted({int(c["faiss_id"]) for c in chunks if "faiss_id" in c})

    removed_count = 0
    try:
        chunk_engine.remove_ids(faiss_ids)
        removed_count = len(faiss_ids)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"failed to remove from FAISS: {exc}"}), 500

    # Clean Mongo collections
    deleted_chunks = chunk_store.delete_chunks_by_doc(doc_id_str)
    chunk_store.delete_document(doc_id_str)

    _deactivate_index_state(company_id, doc_id_str, state="removed")

    return jsonify(
        {
            "documentId": doc_id_str,
            "faissRemoved": removed_count,
            "chunksDeleted": deleted_chunks,
            "state": "removed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.route("/api/v10/content/index", methods=["POST"])
def queue_content_index():
    """
    Queue a content document for indexing.

    JSON body:
        companyId (str, required)
        documentId (str, optional - generates UUID if omitted)
        uploadId (str, optional)
        text (str, optional)
        url (str, optional)
        indexOptions / options (dict, optional)
        metadata (dict, optional)
        trigger (str, optional)
        userId (str, optional)
    """
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON payload"}), 400

    company_id = _get_payload_value(payload, "companyId", "company_id", "companyid")
    if not company_id:
        return jsonify({"error": "companyId is required"}), 400
    company_id = str(company_id)

    document_id = _get_payload_value(payload, "documentId", "document_id", "documentid")
    if document_id:
        document_id = str(document_id)
    else:
        document_id = str(uuid.uuid4())

    user_id = _get_payload_value(payload, "userId", "user_id", "userid")
    if user_id is not None:
        user_id = str(user_id)

    trigger = _get_payload_value(payload, "trigger") or "api"
    job_id = _get_payload_value(payload, "jobId", "job_id")
    if job_id:
        job_id = str(job_id)
    else:
        job_id = str(uuid.uuid4())
    app.logger.info(
        "Content index request received (companyId=%s documentId=%s jobId=%s)",
        company_id,
        document_id,
        job_id,
    )

    raw_options = payload.get("indexOptions") or payload.get("options") or {}
    try:
        options = normalize_index_options(
            raw_options,
            default_chunk_size=DEFAULT_INDEX_CHUNK_SIZE,
            default_chunk_overlap=DEFAULT_INDEX_CHUNK_OVERLAP,
            default_min_chars=DEFAULT_INDEX_MIN_CHARS,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"invalid index options: {exc}"}), 400

    metadata_payload = payload.get("metadata") or {}
    if metadata_payload and not isinstance(metadata_payload, dict):
        return jsonify({"error": "metadata must be an object"}), 400

    existing_doc = content_store.get_document(company_id, document_id)
    metadata = {}
    if isinstance(existing_doc, dict) and isinstance(existing_doc.get("metadata"), dict):
        metadata.update(existing_doc["metadata"])
    metadata.update(metadata_payload)
    metadata.setdefault("companyId", company_id)
    metadata.setdefault("documentId", document_id)

    upload_id = _get_payload_value(payload, "uploadId", "upload_id", "uploadid")
    if not upload_id and isinstance(existing_doc, dict):
        upload_id = _get_payload_value(existing_doc, "uploadId", "upload_id", "uploadid")
    if upload_id:
        upload_id = str(upload_id)
        metadata.setdefault("uploadId", upload_id)

    text = payload.get("text")
    if text is not None and not isinstance(text, str):
        return jsonify({"error": "text must be a string"}), 400
    url = payload.get("url")
    if url is not None and not isinstance(url, str):
        return jsonify({"error": "url must be a string"}), 400
    if not url and isinstance(existing_doc, dict):
        existing_url = existing_doc.get("url")
        if isinstance(existing_url, str) and existing_url.strip():
            url = existing_url

    source = None
    if upload_id:
        source = {"type": "upload", "uploadId": upload_id}
    elif text:
        cleaned_text = text.strip()
        if not cleaned_text:
            return jsonify({"error": "text cannot be empty"}), 400
        source = {"type": "text", "text": text}
        metadata.setdefault("textSize", len(text))
    elif url:
        trimmed_url = url.strip()
        if not trimmed_url:
            return jsonify({"error": "url cannot be empty"}), 400
        source = {"type": "import_url", "url": trimmed_url}
        metadata.setdefault("url", trimmed_url)
    elif isinstance(existing_doc, dict) and isinstance(existing_doc.get("source"), dict):
        source = dict(existing_doc["source"])
    else:
        return jsonify({"error": "uploadId, text, or url is required for new documents"}), 400

    source_type = source.get("type") or source.get("source")
    if source_type:
        source["type"] = str(source_type).lower()
        source["source"] = source["type"]

    doc_type = options.get("scope")
    if not doc_type and isinstance(existing_doc, dict):
        doc_type = existing_doc.get("docType") or existing_doc.get("type")
    if doc_type:
        source.setdefault("docType", doc_type)
        metadata.setdefault("scope", doc_type)

    title = payload.get("title") or metadata.get("title")
    if not title and isinstance(existing_doc, dict):
        title = existing_doc.get("title") or existing_doc.get("name")

    try:
        doc = content_store.upsert_document_with_source(
            company_id=company_id,
            document_id=document_id,
            source=source,
            metadata=metadata,
            options=options,
            job_id=job_id,
            user_id=user_id,
            trigger=trigger,
            title=title,
            doc_type=doc_type,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"failed to prepare document: {exc}"}), 500

    try:
        content_store.append_log_entry(
            company_id=company_id,
            document_id=document_id,
            job_id=job_id,
            level="info",
            message="Index job queued via API.",
            state="queued",
            user_id=user_id,
            trigger=trigger,
            details={"source": source.get("type"), "options": options},
        )
    except Exception:
        # Logging failure should not block the request; it will be visible in stdout.
        pass

    message = {
        "companyId": company_id,
        "documentIds": [document_id],
        "options": options,
        "trigger": trigger,
        "jobId": job_id,
    }
    if user_id:
        message["userId"] = user_id
    app.logger.info(
        "Publishing content index job (queue=%s companyId=%s documentId=%s jobId=%s)",
        getattr(job_publisher, "queue_name", None),
        company_id,
        document_id,
        job_id,
    )

    publish_retries = max(1, int(CONTENT_INDEX_PUBLISH_RETRIES or 1))
    publish_delay = float(CONTENT_INDEX_PUBLISH_RETRY_DELAY_SECONDS or 0)
    last_exc: Exception | None = None
    for attempt in range(1, publish_retries + 1):
        try:
            job_publisher.publish(message)
            last_exc = None
            app.logger.info(
                "Content index job published (queue=%s companyId=%s documentId=%s jobId=%s)",
                getattr(job_publisher, "queue_name", None),
                company_id,
                document_id,
                job_id,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < publish_retries and publish_delay > 0:
                app.logger.warning(
                    "Failed to enqueue indexing job (attempt %s/%s); retrying in %ss: %s",
                    attempt,
                    publish_retries,
                    publish_delay,
                    exc,
                )
                time.sleep(publish_delay)

    if last_exc is not None:
        error_msg = f"failed to enqueue indexing job: {last_exc}"
        content_store.update_index_fields(
            company_id=company_id,
            document_id=document_id,
            state="failed",
            error=error_msg,
            job_id=job_id,
            trigger=trigger,
            options=options,
            user_id=user_id,
        )
        content_store.append_log_entry(
            company_id=company_id,
            document_id=document_id,
            job_id=job_id,
            level="error",
            message=error_msg,
            state="failed",
            user_id=user_id,
            trigger=trigger,
        )
        return jsonify({"error": error_msg}), 503

    index_state = doc.get("index", {}) if isinstance(doc, dict) else {}
    response = {
        "queued": True,
        "documentId": document_id,
        "jobId": job_id,
        "state": index_state.get("state", "queued"),
        "options": options,
    }
    return jsonify(response), 202
@app.route("/api/v10/ingest/markdown", methods=["POST"])
def ingest_markdown():
    """
    Ingest markdown content by chunking and embedding it into FAISS.

    JSON body:
        url (str, required): Source URL of the markdown.
        raw_markdown (str, required): Markdown document content.
        options (dict, optional):
            doc_type (str): Logical grouping label (default 'service').
            target_chars (int): Desired chunk size (default 1100).
            overlap_chars (int): Character overlap between chunks (default 180).
    """
    try:
        data = request.get_json() or {}
        url = (data.get("url") or "").strip()
        raw_md = data.get("raw_markdown") or ""
        opts = data.get("options") or {}
        doc_type = (opts.get("doc_type") or "service").lower()
        target = int(opts.get("target_chars") or 1100)
        overlap = int(opts.get("overlap_chars") or 180)

        result = store.ingest_markdown(
            url=url,
            raw_markdown=raw_md,
            doc_type=doc_type,
            target_chars=target,
            overlap_chars=overlap,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": f"ingest failed: {str(e)}"}), 400


@app.route("/api/v10/uploads/<upload_id>/text", methods=["GET"])
def get_upload_text(upload_id):
    """
    Fetch plain text extracted from the uploaded document on S3.

    Path params:
        upload_id (str): Unique identifier for the upload record.
    """
    try:
        upload_doc = upload_store.get_upload_by_id(upload_id)
    except UploadNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    file_doc = upload_store.get_file_by_upload_id(upload_id)
    try:
        bucket, key, filename = _resolve_upload_s3_location(upload_doc, file_doc)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        document = document_loader.fetch_text(key, bucket=bucket)
    except DocumentDownloadError as exc:
        return jsonify({"error": f"download failed: {exc}"}), 502
    except DocumentParseError as exc:
        return jsonify({"error": f"parse failed: {exc}"}), 422
    except Exception as exc:
        return jsonify({"error": f"unexpected error: {exc}"}), 500

    payload = {
        "upload_id": upload_id,
        "bucket": document.bucket,
        "key": document.key,
        "filename": document.filename or filename,
        "content_type": document.content_type,
        "text": document.text,
    }
    return jsonify(payload)

if __name__ == "__main__":
    port = int(os.getenv("EMBED_PORT", "5003"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
