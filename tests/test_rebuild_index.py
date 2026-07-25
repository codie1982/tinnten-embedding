"""
Firma index'i yeniden inşa testleri.

Soft delete FAISS'e dokunmadığı için yetim vektörler birikir ve kimlikleri
kaybolur — dolayısıyla onları "toplamak" imkânsızdır. Tek çare, hâlâ Mongo'da
karşılığı olan id'lerden temiz bir index kurmaktır. Buradaki testler o kurtarma
yolunun gerçekten alan geri kazandırdığını ve sessizce her şeyi silmediğini doğrular.
"""
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

DIM = 8


def _engine(path):
    from services.embedding_engine import EmbeddingEngine

    EmbeddingEngine._model_cache.clear()
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = DIM
    with patch("services.embedding_engine.SentenceTransformer", return_value=model):
        return EmbeddingEngine(model_name="test-model", index_path=path)


def _vecs(n):
    arr = np.random.rand(n, DIM).astype("float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def test_rebuild_drops_orphans_and_keeps_live_vectors():
    with tempfile.TemporaryDirectory() as tmp:
        eng = _engine(os.path.join(tmp, "c.index"))
        eng.add_embeddings(_vecs(5), [1, 2, 3, 4, 5])

        # Mongo'da yalnızca 1, 3, 5 hayatta (2 ve 4 soft delete ile düştü).
        out = eng.rebuild_from_ids([1, 3, 5], dry_run=False)

        assert out["ok"] is True
        assert out["before"]["ntotal"] == 5
        assert out["after"]["ntotal"] == 3
        assert out["reclaimed"]["vectors"] == 2
        assert out["missingReconstruct"] == 0
        assert eng.count() == 3


def test_rebuild_dry_run_does_not_touch_index():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.index")
        eng = _engine(path)
        eng.add_embeddings(_vecs(4), [1, 2, 3, 4])
        before_mtime = os.path.getmtime(path)

        out = eng.rebuild_from_ids([1, 2], dry_run=True)

        assert out["dryRun"] is True
        assert out["after"]["ntotal"] == 2      # projeksiyon
        assert eng.count() == 4                  # gerçek index dokunulmadı
        assert os.path.getmtime(path) == before_mtime


def test_rebuild_preserves_vector_values_not_just_ids():
    """reconstruct kopyası: aynı id AYNI vektörü taşımalı (yeniden normalize yok)."""
    with tempfile.TemporaryDirectory() as tmp:
        eng = _engine(os.path.join(tmp, "c.index"))
        vecs = _vecs(3)
        eng.add_embeddings(vecs, [7, 8, 9])
        original = eng._index.reconstruct(8).copy()

        eng.rebuild_from_ids([7, 8, 9], dry_run=False)

        np.testing.assert_allclose(eng._index.reconstruct(8), original, rtol=1e-6)


def test_rebuild_counts_ids_missing_from_faiss():
    """Mongo'da olup FAISS'te olmayan id bir ALARMdır, sessizce yutulmamalı."""
    with tempfile.TemporaryDirectory() as tmp:
        eng = _engine(os.path.join(tmp, "c.index"))
        eng.add_embeddings(_vecs(2), [1, 2])

        out = eng.rebuild_from_ids([1, 2, 404], dry_run=False)

        assert out["missingReconstruct"] == 1
        assert out["after"]["ntotal"] == 2


def test_rebuild_refuses_to_wipe_everything():
    """
    Kaynak liste boş gelirse (ör. Mongo sorgusu hatalı) index'i sıfırlamak
    sessiz bir felakettir — açıkça izin verilmedikçe reddedilmeli.
    """
    with tempfile.TemporaryDirectory() as tmp:
        eng = _engine(os.path.join(tmp, "c.index"))
        eng.add_embeddings(_vecs(3), [1, 2, 3])

        out = eng.rebuild_from_ids([], dry_run=False)

        assert out["ok"] is False
        assert out["reason"] == "refusing_empty_rebuild"
        assert eng.count() == 3  # dokunulmadı

        forced = eng.rebuild_from_ids([], dry_run=False, allow_empty=True)
        assert forced["ok"] is True
        assert eng.count() == 0
