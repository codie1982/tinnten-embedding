#!/usr/bin/env python3
"""Search the websearch FAISS index built from tur_Latn shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv



load_dotenv()

DEFAULT_MODEL = os.environ.get(
    "WEBSEARCH_MODEL",
    os.environ.get("CHUNK_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = os.environ.get("WEBSEARCH_INDEX_PATH") or str(BASE_DIR / "websearch.index")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the websearch FAISS index.")
    parser.add_argument("query", help="Search query text.")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return.")
    parser.add_argument("--radius", type=int, default=0, help="Neighboring chunk radius to expand results.")
    parser.add_argument("--index-path", "-i", type=Path, default=Path(DEFAULT_INDEX_PATH), help="Path to the FAISS index.")
    parser.add_argument("--model-name", "-m", default=DEFAULT_MODEL, help="SentenceTransformer model name.")
    parser.add_argument("--text-width", type=int, default=360, help="Max characters to show for each result.")
    parser.add_argument(
        "--filter",
        default=None,
        help="Optional JSON filter (e.g. '{\"metadata.url\": \"https://example.com\"}').",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON results.")
    return parser.parse_args()


def _short_text(value: Optional[str], width: int) -> str:
    if not value:
        return ""
    collapsed = " ".join(value.split())
    return textwrap.shorten(collapsed, width=width, placeholder=" ...")


def _passes_chunk_filters(chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for key, val in (filters or {}).items():
        if key.startswith("metadata."):
            sub = key.split(".", 1)[1]
            if (chunk.get("metadata") or {}).get(sub) != val:
                return False
        else:
            if chunk.get(key) != val:
                return False
    return True


def _reconstruct_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return ""
    ordered = sorted(
        chunks,
        key=lambda c: (
            int(c.get("char_start") or c.get("chunk_index") or 0),
            int(c.get("chunk_index") or 0),
        ),
    )
    buffer: List[str] = []
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


def _parse_filter(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --filter JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--filter must be a JSON object.")
    return parsed


def _faiss_search_subprocess(
    query: str,
    *,
    model_name: str,
    index_path: Path,
    k: int,
) -> tuple[List[float], List[int]]:
    payload = json.dumps(
        {
            "query": query,
            "model": model_name,
            "index_path": str(index_path),
            "k": int(k),
        }
    )
    code = (
        "import json, sys;"
        "import numpy as np;"
        "import faiss;"
        "from sentence_transformers import SentenceTransformer;"
        "payload=json.loads(sys.stdin.read());"
        "model=SentenceTransformer(payload['model']);"
        "vec=model.encode([payload['query']], normalize_embeddings=True);"
        "vec=np.asarray(vec, dtype=np.float32);"
        "faiss.normalize_L2(vec);"
        "index=faiss.read_index(payload['index_path']);"
        "scores, ids = index.search(vec, int(payload['k']));"
        "out={'scores': scores[0].tolist(), 'ids': [int(i) for i in ids[0]]};"
        "print(json.dumps(out))"
    )
    try:
        output = subprocess.check_output(
            [sys.executable, "-c", code],
            input=payload,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Search process failed: {exc}") from exc
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse search output: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("Search output is not a JSON object.")
    scores = data.get("scores") or []
    ids = data.get("ids") or []
    if not isinstance(scores, list) or not isinstance(ids, list):
        raise SystemExit("Search output is invalid.")
    return scores, ids


def _search_index(
    query: str,
    *,
    k: int,
    radius: int,
    filters: Dict[str, Any],
    model_name: str,
    index_path: Path,
) -> List[Dict[str, Any]]:
    if k <= 0:
        raise SystemExit("--k must be positive.")
    if radius < 0:
        raise SystemExit("--radius must be >= 0.")

    scores, ids = _faiss_search_subprocess(
        query,
        model_name=model_name,
        index_path=index_path,
        k=k,
    )
    faiss_ids = [int(i) for i in ids if i != -1]
    from services.mongo_store import MongoStore

    store = MongoStore()
    chunk_docs = store.get_chunks_by_faiss_ids(faiss_ids)
    doc_ids = {c.get("doc_id") for c in chunk_docs.values() if c}
    doc_status_map = store.get_documents_by_ids(doc_ids)
    doc_chunk_cache: Dict[str, List[Dict[str, Any]]] = {}

    results: List[Dict[str, Any]] = []
    for idx, score in zip(ids, scores):
        if idx == -1:
            continue
        chunk = chunk_docs.get(int(idx))
        if not chunk:
            continue
        doc_info = doc_status_map.get(chunk.get("doc_id"))
        if doc_info and str(doc_info.get("status")).lower() in {"disabled", "removed"}:
            continue
        if filters and not _passes_chunk_filters(chunk, filters):
            continue

        combined_text = chunk.get("text")
        if radius > 0:
            doc_id = str(chunk.get("doc_id") or "")
            chunk_index = chunk.get("chunk_index") if "chunk_index" in chunk else chunk.get("index")
            if doc_id and chunk_index is not None:
                if doc_id not in doc_chunk_cache:
                    doc_chunk_cache[doc_id] = store.get_chunks_by_doc(doc_id)
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
    return results


def main() -> None:
    args = _parse_args()
    query = args.query.strip()
    if not query:
        raise SystemExit("Query text is required.")

    filters = _parse_filter(args.filter)
    results = _search_index(
        query,
        k=args.k,
        radius=args.radius,
        filters=filters,
        model_name=args.model_name,
        index_path=args.index_path,
    )

    if args.json:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
        return

    if not results:
        print("Sonuc bulunamadi.")
        return

    for i, item in enumerate(results, start=1):
        meta = item.get("metadata") or {}
        url = meta.get("url") or ""
        text_snippet = _short_text(item.get("text"), width=args.text_width)
        print(f"{i}. score={item['score']:.4f} id={item['id']} doc_id={item.get('doc_id')}")
        if url:
            print(f"   url: {url}")
        print(f"   text: {text_snippet}\n")

if __name__ == "__main__":
    main()
