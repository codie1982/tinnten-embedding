import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from vector_store import EmbeddingIndex

load_dotenv()

INDEX_PATH = os.getenv("INDEX_PATH", "/data/faiss.index")
META_PATH  = os.getenv("META_PATH",  "/data/meta.json")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Tek bir instance yeterli
store = EmbeddingIndex(
    model_name=MODEL_NAME,
    index_path=INDEX_PATH,
    meta_path=META_PATH,
)

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

if __name__ == "__main__":
    port = int(os.getenv("EMBED_PORT", "5003"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
