import os
from functools import wraps
from typing import Any, Dict, Optional

import numpy as np
from flask import Flask, jsonify, request, g
from flask_cors import CORS
from dotenv import load_dotenv

from keycloak_service import KeycloakError, KeycloakTokenError, get_keycloak_service
from services.embedding_engine import EmbeddingEngine
from services.mongo_store import MongoStore
from services.rabbit_publisher import RabbitPublisher
from services.upload_store import UploadStore, UploadNotFoundError

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

keycloak = get_keycloak_service()

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
CHUNK_MODEL_NAME = os.getenv("CHUNK_MODEL_NAME", MODEL_NAME)
INDEX_PATH = os.getenv("CHUNK_INDEX_PATH") or os.getenv("FAISS_INDEX_PATH") or "faiss.index"
DEFAULT_CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE") or 1200)
DEFAULT_CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP") or 200)
DEFAULT_MIN_CHARS = int(os.getenv("EMBED_MIN_CHARS") or 80)

embedding_engine = EmbeddingEngine(model_name=CHUNK_MODEL_NAME, index_path=INDEX_PATH)
mongo_store = MongoStore()
rabbit_publisher = RabbitPublisher()
upload_store = UploadStore()


def require_service_token(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        authorization = request.headers.get("Authorization")
        try:
            g.keycloak_token = keycloak.validate_bearer_header(authorization)
        except KeycloakTokenError as exc:
            return jsonify({"error": "unauthorized", "message": str(exc)}), 401
        except KeycloakError as exc:
            return jsonify({"error": "keycloak_error", "message": str(exc)}), 500
        return fn(*args, **kwargs)

    return wrapper


@app.route("/", methods=["GET"])
def root():
    embedding_engine.reload_if_updated()
    return jsonify(
        {
            "message": "tinnten-embedding up",
            "index_size": embedding_engine.count(),
        }
    )


@app.route("/api/v10/llm/vector", methods=["POST"])
@require_service_token
def generate_vector():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided in the request body."}), 400
    vec = embedding_engine.encode_single(text)
    return jsonify({"vector": vec[0].tolist()})


@app.route("/api/v10/vector/ingest-web", methods=["POST"])
@require_service_token
def ingest_web():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    url = (data.get("url") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    if not url:
        return jsonify({"error": "url is required"}), 400

    chunk_size = int(data.get("chunk_size") or data.get("target_chars") or DEFAULT_CHUNK_SIZE)
    chunk_overlap = int(data.get("chunk_overlap") or data.get("overlap_chars") or DEFAULT_CHUNK_OVERLAP)
    min_chars = int(data.get("min_chars") or DEFAULT_MIN_CHARS)
    if chunk_size <= 0:
        return jsonify({"error": "chunk_size must be positive"}), 400
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        return jsonify({"error": "chunk_overlap must be >=0 and smaller than chunk_size"}), 400
    if min_chars <= 0:
        return jsonify({"error": "min_chars must be positive"}), 400

    metadata = data.get("metadata") or {}
    metadata.setdefault("url", url)
    metadata.setdefault("source", "web")

    doc_id = mongo_store.create_document(
        doc_id=data.get("doc_id"),
        doc_type="web",
        source="web",
        metadata=metadata,
    )

    message = {
        "ingest_type": "web",
        "doc_id": doc_id,
        "doc_type": "web",
        "text": text,
        "url": url,
        "metadata": metadata,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "min_chars": min_chars,
    }

    try:
        rabbit_publisher.publish(message)
    except Exception as exc:
        mongo_store.update_document_status(doc_id, status="failed", error=str(exc))
        return jsonify({"error": "failed_to_enqueue", "message": str(exc)}), 500

    return jsonify({"doc_id": doc_id, "status": "queued"})


@app.route("/api/v10/vector/ingest-upload", methods=["POST"])
@require_service_token
def ingest_upload():
    data = request.get_json() or {}
    upload_id = (data.get("upload_id") or data.get("uploadId") or "").strip()
    if not upload_id:
        return jsonify({"error": "upload_id is required"}), 400

    try:
        upload_doc = upload_store.get_upload_by_id(upload_id)
    except UploadNotFoundError as exc:
        return jsonify({"error": "upload_not_found", "message": str(exc)}), 404

    metadata = data.get("metadata") or {}
    metadata.setdefault("upload_id", upload_id)
    metadata.setdefault("upload_type", upload_doc.get("uploadType"))

    chunk_size = int(data.get("chunk_size") or data.get("target_chars") or DEFAULT_CHUNK_SIZE)
    chunk_overlap = int(data.get("chunk_overlap") or data.get("overlap_chars") or DEFAULT_CHUNK_OVERLAP)
    min_chars = int(data.get("min_chars") or DEFAULT_MIN_CHARS)
    if chunk_size <= 0:
        return jsonify({"error": "chunk_size must be positive"}), 400
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        return jsonify({"error": "chunk_overlap must be >=0 and smaller than chunk_size"}), 400
    if min_chars <= 0:
        return jsonify({"error": "min_chars must be positive"}), 400

    doc_id = mongo_store.create_document(
        doc_id=data.get("doc_id"),
        doc_type="document",
        source="upload",
        metadata=metadata,
    )

    upload_store.update_upload_status(
        upload_id,
        index_status="pending",
        is_file_opened=False,
        file_open_error=None,
        embedding_doc_id=doc_id,
    )

    message = {
        "ingest_type": "upload",
        "doc_id": doc_id,
        "doc_type": "document",
        "upload_id": upload_id,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "min_chars": min_chars,
        "metadata": metadata,
    }

    try:
        rabbit_publisher.publish(message)
    except Exception as exc:
        mongo_store.update_document_status(doc_id, status="failed", error=str(exc))
        upload_store.update_upload_status(
            upload_id,
            index_status="failed",
            is_file_opened=False,
            file_open_error=str(exc),
        )
        return jsonify({"error": "failed_to_enqueue", "message": str(exc)}), 500

    return jsonify({"doc_id": doc_id, "status": "queued"})


@app.route("/api/v10/vector/search", methods=["POST"])
@require_service_token
def search_vector():
    data = request.get_json() or {}
    k = int(data.get("k") or 5)
    if k <= 0:
        return jsonify({"error": "k must be positive"}), 400

    vector = data.get("vector")
    if vector is None:
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Provide either 'text' or 'vector'."}), 400
        query_vec = embedding_engine.encode([text], batch_size=1)
    else:
        arr = np.array(vector, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            return jsonify({"error": "vector must be 1D or 2D array"}), 400
        query_vec = arr.astype(np.float32, copy=False)

    try:
        scores, ids = embedding_engine.search(query_vec, k)
    except RuntimeError:
        return jsonify({"results": []})

    faiss_ids = [int(i) for i in ids[0] if i != -1]
    chunk_docs = mongo_store.get_chunks_by_faiss_ids(faiss_ids)
    filters = data.get("filter") or {}

    results = []
    for idx, score in zip(ids[0], scores[0]):
        if idx == -1:
            continue
        chunk = chunk_docs.get(int(idx))
        if not chunk:
            continue
        if not _passes_filters(chunk, filters):
            continue
        results.append(
            {
                "id": int(idx),
                "score": float(score),
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "text": chunk.get("text"),
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
                "doc_type": chunk.get("doc_type"),
                "source": chunk.get("source"),
                "metadata": chunk.get("metadata", {}),
            }
        )

    return jsonify({"results": results})


@app.route("/api/v10/vector/document/<doc_id>", methods=["GET"])
@require_service_token
def document_status(doc_id: str):
    doc = mongo_store.get_document(doc_id)
    if not doc:
        return jsonify({"error": "not_found"}), 404

    response = _serialize_document(doc)
    include_chunks = request.args.get("include_chunks", "").lower() in {"1", "true", "yes"}
    if include_chunks:
        chunks = mongo_store.get_chunks_by_doc(doc_id)
        response["chunks"] = [_serialize_chunk(chunk) for chunk in chunks]

    return jsonify(response)


def _passes_filters(chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if key.startswith("metadata."):
            sub = key.split(".", 1)[1]
            current = (chunk.get("metadata") or {}).get(sub)
        else:
            current = chunk.get(key)
        if current != expected:
            return False
    return True


def _serialize_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in doc.items() if k not in {"_id"}}
    if "created_at" in out and hasattr(out["created_at"], "isoformat"):
        out["created_at"] = out["created_at"].isoformat()
    if "updated_at" in out and hasattr(out["updated_at"], "isoformat"):
        out["updated_at"] = out["updated_at"].isoformat()
    return out


def _serialize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in chunk.items() if k not in {"_id"}}
    if "created_at" in out and hasattr(out["created_at"], "isoformat"):
        out["created_at"] = out["created_at"].isoformat()
    return out


if __name__ == "__main__":
    port = int(os.getenv("EMBED_PORT", "5003"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
