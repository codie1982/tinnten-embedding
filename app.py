import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from vector_store import EmbeddingIndex
from services import (
    DocumentLoader,
    DocumentDownloadError,
    DocumentParseError,
    UploadStore,
    UploadNotFoundError,
    ContentDocumentStore,
    normalize_index_options,
)
from services.rabbit_publisher import RabbitPublisher

load_dotenv()

INDEX_PATH = os.getenv("INDEX_PATH", "/data/faiss.index")
META_PATH  = os.getenv("META_PATH",  "/data/meta.json")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
DEFAULT_INDEX_CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE") or 1200)
DEFAULT_INDEX_CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP") or 200)
DEFAULT_INDEX_MIN_CHARS = int(os.getenv("EMBED_MIN_CHARS") or 80)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Tek bir instance yeterli
store = EmbeddingIndex(
    model_name=MODEL_NAME,
    index_path=INDEX_PATH,
    meta_path=META_PATH,
)
upload_store = UploadStore()
document_loader = DocumentLoader()
content_store = ContentDocumentStore()
job_publisher = RabbitPublisher()


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
        id (int, optional): Existing FAISS ID to overwrite; auto-assigned otherwise.
    """
    try:
        p = request.get_json() or {}
        text = p.get("text")
        vector = p.get("vector")
        external_id = p.get("external_id")
        metadata = p.get("metadata") or {}
        int_id = p.get("id")
        out = store.upsert_vector(text, vector, external_id, metadata, int_id)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": f"upsert failed: {str(e)}"}), 400

@app.route("/api/v10/vector/search", methods=["POST"])
def search_vector():
    """
    Run a similarity search against the FAISS index.

    JSON body:
        text (str, optional): Query text to encode if vector omitted.
        vector (list[float], optional): Pre-computed query vector.
        k (int, optional): Number of results to return (default 5).
        filter (dict, optional): Metadata filters applied to stored entries.
    """
    try:
        p = request.get_json() or {}
        text = p.get("text")
        vector = p.get("vector")
        k = int(p.get("k") or 5)
        filt = p.get("filter") or {}
        results = store.search(text, vector, k, filt)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": f"search failed: {str(e)}"}), 400


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
    if upload_id:
        upload_id = str(upload_id)
        metadata.setdefault("uploadId", upload_id)

    text = payload.get("text")
    if text is not None and not isinstance(text, str):
        return jsonify({"error": "text must be a string"}), 400
    url = payload.get("url")
    if url is not None and not isinstance(url, str):
        return jsonify({"error": "url must be a string"}), 400

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
        doc_type = existing_doc.get("docType")
    if doc_type:
        source.setdefault("docType", doc_type)
        metadata.setdefault("scope", doc_type)

    title = payload.get("title") or metadata.get("title")
    if not title and isinstance(existing_doc, dict):
        title = existing_doc.get("title")

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
