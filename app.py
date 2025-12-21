import os
import uuid
from datetime import datetime, timezone

import numpy as np
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
from keycloak_service import get_keycloak_service, KeycloakError, KeycloakTokenError

load_dotenv()


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


CHUNK_INDEX_PATH = _path_from_env("INDEX_PATH", "FAISS_INDEX_PATH", default="faiss.index")
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
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
CATEGORY_MODEL_NAME = os.getenv("CATEGORY_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
ATTRIBUTE_MODEL_NAME = os.getenv("ATTRIBUTE_MODEL_NAME", CATEGORY_MODEL_NAME)
CATEGORY_META_DB_NAME = os.getenv("CATEGORY_META_DB_NAME")
CATEGORY_META_COLLECTION = os.getenv("CATEGORY_META_COLLECTION", "category_faiss_metadata")
ATTRIBUTE_META_DB_NAME = os.getenv("ATTRIBUTE_META_DB_NAME")
ATTRIBUTE_META_COLLECTION = os.getenv("ATTRIBUTE_META_COLLECTION", "attribute_faiss_metadata")
DEFAULT_INDEX_CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE") or 1200)
DEFAULT_INDEX_CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP") or 200)
DEFAULT_INDEX_MIN_CHARS = int(os.getenv("EMBED_MIN_CHARS") or 80)
REQUIRE_KEYCLOAK_AUTH = (os.getenv("REQUIRE_KEYCLOAK_AUTH") or "true").strip().lower() not in {"0", "false", "no", "off"}

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
chunk_engine = EmbeddingEngine(model_name=MODEL_NAME, index_path=CHUNK_INDEX_PATH)
chunk_store = MongoStore()
def _should_require_auth() -> bool:
    if not REQUIRE_KEYCLOAK_AUTH:
        return False
    if request.method == "OPTIONS":
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


def _vector_upsert_response(target_store, label: str = "upsert", payload: dict | None = None):
    try:
        payload = payload if payload is not None else (request.get_json() or {})
        text = payload.get("text")
        vector = payload.get("vector")
        external_id = payload.get("external_id")
        metadata = payload.get("metadata") or {}
        int_id = payload.get("id")
        out = target_store.upsert_vector(text, vector, external_id, metadata, int_id)
        return jsonify(out)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"{label} failed: {exc}"}), 400


def _vector_search_response(target_store, label: str = "search", payload: dict | None = None):
    try:
        payload = payload if payload is not None else (request.get_json() or {})
        text = payload.get("text")
        vector = payload.get("vector")
        k = int(payload.get("k") or 5)
        filt = payload.get("filter") or {}
        results = target_store.search(text, vector, k, filt)
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
    size = int(store.index.ntotal) if store.index is not None else 0
    return jsonify({"message": "tinnten-embedding up", "index_size": size})

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
        return _vector_search_response(category_store, label="category search", payload=payload)
    if target in {"attribute", "attributes"}:
        return _vector_search_response(attribute_store, label="attribute search", payload=payload)

    # Fallback to legacy/vector search against the general index
    return _vector_search_response(store, label="vector search", payload=payload)


@app.route("/api/v10/categories/upsert", methods=["POST"])
def upsert_category_vector():
    """
    Insert or update a category vector in its dedicated FAISS index.

    JSON body:
        parentId (str, optional): Parent category identifier; stored under metadata.parentId.
    """
    payload = request.get_json() or {}
    meta_payload = payload.get("metadata") or {}
    if meta_payload and not isinstance(meta_payload, dict):
        return jsonify({"error": "metadata must be an object"}), 400

    parent_id = payload.get("parentId") or payload.get("parent_id") or payload.get("parent")
    if parent_id is not None:
        meta_payload = dict(meta_payload)
        meta_payload["parentId"] = str(parent_id)

    payload["metadata"] = meta_payload
    return _vector_upsert_response(category_store, label="category upsert", payload=payload)


@app.route("/api/v10/categories/search", methods=["POST"])
def search_category_vector():
    """
    Run a similarity search on the category FAISS index.
    """
    return _vector_search_response(category_store, label="category search")


@app.route("/api/v10/attributes/upsert", methods=["POST"])
def upsert_attribute_vector():
    """
    Insert or update an attribute vector in its dedicated FAISS index.

    JSON body:
        categoryId (str, optional): Link attribute to a category; stored under metadata.categoryId.
    """
    payload = request.get_json() or {}
    meta_payload = payload.get("metadata") or {}
    if meta_payload and not isinstance(meta_payload, dict):
        return jsonify({"error": "metadata must be an object"}), 400

    category_id = payload.get("categoryId") or payload.get("category_id") or payload.get("category")
    if category_id is not None:
        meta_payload = dict(meta_payload)
        meta_payload["categoryId"] = str(category_id)

    payload["metadata"] = meta_payload
    return _vector_upsert_response(attribute_store, label="attribute upsert", payload=payload)


@app.route("/api/v10/attributes/search", methods=["POST"])
def search_attribute_vector():
    """
    Run a similarity search on the attribute FAISS index.
    """
    return _vector_search_response(attribute_store, label="attribute search")


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

    try:
        job_publisher.publish(message)
    except Exception as exc:  # noqa: BLE001
        error_msg = f"failed to enqueue indexing job: {exc}"
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
