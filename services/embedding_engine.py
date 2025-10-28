"""
Shared embedding + FAISS index management utilities.
"""
from __future__ import annotations

import os
import threading
from typing import Iterable, List, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """
    Wraps a SentenceTransformer model together with a FAISS index.
    Safe to share across threads via internal locking.
    """

    def __init__(
        self,
        model_name: str,
        index_path: str = "faiss.index",
    ) -> None:
        self.model_name = model_name
        self.index_path = index_path
        self._lock = threading.RLock()
        self._model = SentenceTransformer(self.model_name)
        self._index: faiss.IndexIDMap2 | None = None
        self._dimension: int | None = None
        self._index_mtime: float | None = None
        self._load_index()

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of strings into a normalised numpy array.
        """
        if not texts:
            raise ValueError("texts cannot be empty")
        embeddings = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vec, dtype=np.float32).reshape(1, -1)

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    def add_embeddings(self, embeddings: np.ndarray, ids: Sequence[int]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if len(ids) != embeddings.shape[0]:
            raise ValueError("ids length must match embeddings rows")
        with self._lock:
            self._ensure_index(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            id_array = np.asarray(list(ids), dtype=np.int64)
            self._index.add_with_ids(embeddings, id_array)
            self._save_index()

    def search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if query_vectors.ndim != 2:
            raise ValueError("query_vectors must be 2D")
        if k <= 0:
            raise ValueError("k must be positive")
        with self._lock:
            self.reload_if_updated_locked()
            if self._index is None or self._index.ntotal == 0:
                raise RuntimeError("FAISS index is empty.")
            if self._index.d != query_vectors.shape[1]:
                raise ValueError(f"Query dim {query_vectors.shape[1]} != index dim {self._index.d}")
            faiss.normalize_L2(query_vectors)
            scores, ids = self._index.search(query_vectors, k)
            return scores, ids

    def count(self) -> int:
        with self._lock:
            return int(self._index.ntotal) if self._index is not None else 0

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _load_index(self) -> None:
        if not os.path.exists(self.index_path):
            return
        with self._lock:
            index = faiss.read_index(self.index_path)
            if not isinstance(index, faiss.IndexIDMap):
                # Wrap plain indices to support explicit IDs
                index = faiss.IndexIDMap(index)
            self._index = faiss.downcast_index(index)
            self._dimension = self._index.d
            try:
                self._index_mtime = os.path.getmtime(self.index_path)
            except OSError:
                self._index_mtime = None

    def _save_index(self) -> None:
        if self._index is None:
            return
        tmp = f"{self.index_path}.tmp"
        faiss.write_index(self._index, tmp)
        os.replace(tmp, self.index_path)
        try:
            self._index_mtime = os.path.getmtime(self.index_path)
        except OSError:
            self._index_mtime = None

    def _ensure_index(self, dimension: int) -> None:
        if self._index is None:
            base = faiss.IndexFlatIP(dimension)
            self._index = faiss.IndexIDMap2(base)
            self._dimension = dimension
            return
        if self._index.d != dimension:
            raise ValueError(f"Embedding dimension mismatch: {dimension} != {self._index.d}")

    def reload_if_updated(self) -> None:
        """
        Public entrypoint that reloads the FAISS index from disk if it has changed.
        """
        with self._lock:
            self.reload_if_updated_locked()

    def reload_if_updated_locked(self) -> None:
        """
        Internal helper; caller must hold `_lock`.
        """
        if not os.path.exists(self.index_path):
            return
        try:
            mtime = os.path.getmtime(self.index_path)
        except OSError:
            return
        if self._index is None or self._index_mtime is None or mtime > self._index_mtime:
            index = faiss.read_index(self.index_path)
            if not isinstance(index, faiss.IndexIDMap):
                index = faiss.IndexIDMap(index)
            self._index = faiss.downcast_index(index)
            self._dimension = self._index.d
            self._index_mtime = mtime
