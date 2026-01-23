"""
Shared embedding + FAISS index management utilities.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Iterable, Iterator, List, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import fcntl  # type: ignore
except Exception:  # noqa: BLE001
    fcntl = None


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
        self.auto_reset_on_error = (os.getenv("FAISS_AUTO_RESET_ON_ERROR") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.auto_reset_on_dim_mismatch = (os.getenv("FAISS_AUTO_RESET_ON_DIM_MISMATCH") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.max_index_dimension = int(os.getenv("FAISS_MAX_INDEX_DIMENSION") or 16384)
        self._lock = threading.RLock()
        self._model = SentenceTransformer(self.model_name)
        self._index: faiss.IndexIDMap2 | None = None
        self._dimension: int | None = None
        self._model_dimension: int | None = None
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
    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        """
        Best-effort cross-process lock for index writers.

        This protects the `.tmp` write+replace sequence and prevents lost updates when
        multiple ingest workers run concurrently on the same shared filesystem.
        """
        if fcntl is None:
            yield
            return
        lock_path = f"{self.index_path}.lock"
        lock_dir = os.path.dirname(lock_path)
        if lock_dir:
            try:
                os.makedirs(lock_dir, exist_ok=True)
            except OSError:
                pass
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def add_embeddings(self, embeddings: np.ndarray, ids: Sequence[int]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if len(ids) != embeddings.shape[0]:
            raise ValueError("ids length must match embeddings rows")
        with self._lock:
            with self._write_lock():
                self.reload_if_updated_locked()
                self._maybe_refresh_index(expected_dim=embeddings.shape[1])
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
                # Fallback to a fresh on-disk read if the in-memory index was invalidated.
                disk_index = self._read_index_from_disk()
                if disk_index is not None:
                    if disk_index.d != query_vectors.shape[1]:
                        raise ValueError(f"Query dim {query_vectors.shape[1]} != index dim {disk_index.d}")
                    if disk_index.ntotal > 0:
                        faiss.normalize_L2(query_vectors)
                        return disk_index.search(query_vectors, k)
                raise RuntimeError("FAISS index is empty.")
            if self._index.d <= 0 or self._index.d > self.max_index_dimension:
                disk_index = self._read_index_from_disk()
                if disk_index is not None and disk_index.d == query_vectors.shape[1] and disk_index.ntotal > 0:
                    faiss.normalize_L2(query_vectors)
                    return disk_index.search(query_vectors, k)
                reason = f"invalid FAISS index dimension: {self._index.d}"
                print(f"[embedding-engine] {reason}", flush=True)
                if self.auto_reset_on_error:
                    with self._write_lock():
                        self._quarantine_index_file(reason=reason)
                    self._index = None
                    self._dimension = None
                    self._index_mtime = None
                raise RuntimeError("FAISS index is empty.")
            if self._index.d != query_vectors.shape[1]:
                raise ValueError(f"Query dim {query_vectors.shape[1]} != index dim {self._index.d}")
            faiss.normalize_L2(query_vectors)
            scores, ids = self._index.search(query_vectors, k)
            return scores, ids

    def count(self) -> int:
        with self._lock:
            return int(self._index.ntotal) if self._index is not None else 0

    def remove_ids(self, ids: Sequence[int]) -> int:
        """
        Remove the supplied FAISS IDs from the index and persist the updated index.
        Returns the number of IDs requested for removal (not the number actually present).
        """
        id_list = [int(i) for i in ids if i is not None]
        if not id_list:
            return 0
        with self._lock:
            with self._write_lock():
                self.reload_if_updated_locked()
                if self._index is None:
                    return 0
                id_array = np.asarray(id_list, dtype=np.int64)
                try:
                    self._index.remove_ids(id_array)
                finally:
                    self._save_index()
        return len(id_list)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def model_dimension(self) -> int:
        with self._lock:
            if self._model_dimension is not None:
                return self._model_dimension
            getter = getattr(self._model, "get_sentence_embedding_dimension", None)
            if callable(getter):
                self._model_dimension = int(getter())
            else:
                self._model_dimension = int(self.encode_single("dimension_probe").shape[1])
            return self._model_dimension

    def index_dimension(self) -> int | None:
        with self._lock:
            return int(self._dimension) if self._dimension is not None else None

    def _quarantine_index_file(self, *, reason: str) -> None:
        """
        Move the current index file aside so a fresh index can be created.
        """
        if not os.path.exists(self.index_path):
            return
        suffix = time.strftime("%Y%m%d-%H%M%S")
        target = f"{self.index_path}.corrupt.{suffix}"
        try:
            os.replace(self.index_path, target)
        except OSError:
            return
        print(f"[embedding-engine] quarantined invalid index {self.index_path} -> {target} ({reason})", flush=True)

    def _load_index(self) -> None:
        if not os.path.exists(self.index_path):
            return
        with self._lock:
            try:
                index = faiss.read_index(self.index_path)
                if not isinstance(index, faiss.IndexIDMap):
                    # Wrap plain indices to support explicit IDs
                    index = faiss.IndexIDMap(index)
                self._index = faiss.downcast_index(index)
                self._dimension = int(self._index.d)
                if self._dimension <= 0 or self._dimension > self.max_index_dimension:
                    raise ValueError(f"invalid FAISS index dimension: {self._dimension}")
                model_dim = self.model_dimension()
                if model_dim != self._dimension:
                    msg = (
                        f"FAISS index dim mismatch for {self.index_path}: "
                        f"index_dim={self._dimension} model_dim={model_dim} model={self.model_name}"
                    )
                    print(f"[embedding-engine] {msg}", flush=True)
                    if self.auto_reset_on_dim_mismatch:
                        self._quarantine_index_file(reason=msg)
                        self._index = None
                        self._dimension = None
                        self._index_mtime = None
                        return
                try:
                    self._index_mtime = os.path.getmtime(self.index_path)
                except OSError:
                    self._index_mtime = None
            except Exception as exc:  # noqa: BLE001
                # If the index file is corrupt or not a FAISS index, avoid crashing the service.
                # Optionally move it aside so ingest can rebuild a fresh one.
                print(f"[embedding-engine] failed to load index {self.index_path}: {exc}", flush=True)
                if self.auto_reset_on_error or "invalid FAISS index dimension" in str(exc):
                    self._quarantine_index_file(reason=str(exc))
                self._index = None
                self._dimension = None
                self._index_mtime = None

    def _save_index(self) -> None:
        if self._index is None:
            return
        tmp = f"{self.index_path}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
        try:
            faiss.write_index(self._index, tmp)
            os.replace(tmp, self.index_path)
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
        try:
            self._index_mtime = os.path.getmtime(self.index_path)
        except OSError:
            self._index_mtime = None

    def _read_index_from_disk(self) -> faiss.IndexIDMap2 | None:
        if not os.path.exists(self.index_path):
            return None
        try:
            index = faiss.read_index(self.index_path)
        except Exception:
            return None
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
        return faiss.downcast_index(index)

    def _maybe_refresh_index(self, expected_dim: int) -> None:
        if self._index is None:
            return
        dim = int(getattr(self._index, "d", 0) or 0)
        needs_reload = dim <= 0 or dim > self.max_index_dimension or dim != expected_dim
        if needs_reload:
            disk_index = self._read_index_from_disk()
            if disk_index is not None:
                self._index = disk_index
                self._dimension = int(disk_index.d)
                try:
                    self._index_mtime = os.path.getmtime(self.index_path)
                except OSError:
                    self._index_mtime = None
                dim = int(self._index.d)
        if self._index is None:
            return
        if dim <= 0 or dim > self.max_index_dimension:
            reason = f"invalid FAISS index dimension: {dim}"
            print(f"[embedding-engine] {reason}", flush=True)
            if self.auto_reset_on_error:
                self._quarantine_index_file(reason=reason)
                self._index = None
                self._dimension = None
                self._index_mtime = None
            return
        if dim != expected_dim:
            reason = (
                f"FAISS index dim mismatch for {self.index_path}: index_dim={dim} expected={expected_dim}"
            )
            print(f"[embedding-engine] {reason}", flush=True)
            if self.auto_reset_on_dim_mismatch:
                self._quarantine_index_file(reason=reason)
                self._index = None
                self._dimension = None
                self._index_mtime = None
            else:
                raise ValueError(f"Embedding dimension mismatch: {expected_dim} != {dim}")

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
            try:
                index = faiss.read_index(self.index_path)
                if not isinstance(index, faiss.IndexIDMap):
                    index = faiss.IndexIDMap(index)
                self._index = faiss.downcast_index(index)
                self._dimension = int(self._index.d)
                if self._dimension <= 0 or self._dimension > self.max_index_dimension:
                    raise ValueError(f"invalid FAISS index dimension: {self._dimension}")
                model_dim = self.model_dimension()
                if model_dim != self._dimension:
                    msg = (
                        f"FAISS index dim mismatch for {self.index_path}: "
                        f"index_dim={self._dimension} model_dim={model_dim} model={self.model_name}"
                    )
                    print(f"[embedding-engine] {msg}", flush=True)
                    if self.auto_reset_on_dim_mismatch:
                        self._quarantine_index_file(reason=msg)
                        self._index = None
                        self._dimension = None
                        self._index_mtime = None
                        return
                self._index_mtime = mtime
            except Exception as exc:  # noqa: BLE001
                print(f"[embedding-engine] failed to reload index {self.index_path}: {exc}", flush=True)
                if self.auto_reset_on_error or "invalid FAISS index dimension" in str(exc):
                    self._quarantine_index_file(reason=str(exc))
                self._index = None
                self._dimension = None
                self._index_mtime = None
