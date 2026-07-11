"""
EmbeddingEngine.search self-heal: canlı bellek index'i bozulup TÜM -1 dönerse
(disk sağlamken) — "index dolu ama arama 0" arızası — diskten yeniden okuyup
tekrar dener. Restart beklemeden kendini onarır.
"""
import gc
import threading
from unittest.mock import MagicMock

import numpy as np

from services.embedding_engine import EmbeddingEngine


def test_reload_keeps_faiss_owner_alive(tmp_path):
    """The downcast SWIG view must not outlive its owning index wrapper."""
    import faiss

    path = tmp_path / "owned.index"
    source = faiss.IndexIDMap2(faiss.IndexFlatIP(2))
    vectors = np.asarray([[1.0, 0.0]], dtype="float32")
    source.add_with_ids(vectors, np.asarray([42], dtype="int64"))
    faiss.write_index(source, str(path))

    eng = EmbeddingEngine.__new__(EmbeddingEngine)
    eng.index_path = str(path)
    eng._backing_index = None
    loaded = eng._read_index_from_disk()
    gc.collect()

    assert eng._backing_index is not None
    assert loaded.d == 2
    scores, ids = loaded.search(vectors, 1)
    assert int(ids[0][0]) == 42


def _bare_engine():
    eng = EmbeddingEngine.__new__(EmbeddingEngine)  # __init__/model yüklemesini atla
    eng._lock = threading.RLock()
    eng.max_index_dimension = 16384
    eng.index_path = "/nonexistent/faiss.index"
    eng._dimension = 768
    eng._index_mtime = None
    eng.reload_if_updated_locked = lambda: None  # no-op (mtime reload'u devre dışı)
    return eng


def test_search_self_heals_on_empty_when_disk_has_data(mocker):
    eng = _bare_engine()
    # Bellek index'i DOLU görünür (.d/.ntotal ok) ama search TÜM -1 döner (bozuk C++ obj).
    mem = MagicMock()
    mem.d = 768
    mem.ntotal = 5
    mem.search.return_value = (np.array([[0.0, 0.0]]), np.array([[-1, -1]]))
    eng._index = mem
    # Disk index: sağlam → gerçek sonuç döner.
    disk = MagicMock()
    disk.d = 768
    disk.ntotal = 8115
    disk.search.return_value = (np.array([[0.8, 0.7]]), np.array([[42, 43]]))
    mocker.patch.object(eng, "_read_index_from_disk", return_value=disk)

    scores, ids = eng.search(np.zeros((1, 768), dtype="float32"), 2)
    assert list(ids[0]) == [42, 43]   # disk'ten gelen gerçek sonuç servis edildi
    assert eng._index is disk          # bozuk bellek index'i disk ile değiştirildi
    disk.search.assert_called_once()


def test_search_no_reread_when_results_present(mocker):
    eng = _bare_engine()
    mem = MagicMock()
    mem.d = 768
    mem.ntotal = 8115
    mem.search.return_value = (np.array([[0.9]]), np.array([[7]]))
    eng._index = mem
    reread = mocker.patch.object(eng, "_read_index_from_disk")

    scores, ids = eng.search(np.zeros((1, 768), dtype="float32"), 1)
    assert list(ids[0]) == [7]
    reread.assert_not_called()  # sonuç var → gereksiz disk okuması YAPILMAZ


def test_search_stays_empty_when_disk_also_empty(mocker):
    """Gerçekten eşleşme yoksa (disk de boş/eşleşmez) sonuç boş kalır — sonsuz
    yeniden-okuma YOK (tek deneme)."""
    eng = _bare_engine()
    mem = MagicMock()
    mem.d = 768
    mem.ntotal = 3
    mem.search.return_value = (np.array([[0.0]]), np.array([[-1]]))
    eng._index = mem
    disk = MagicMock()
    disk.d = 768
    disk.ntotal = 0  # disk de boş
    mocker.patch.object(eng, "_read_index_from_disk", return_value=disk)

    scores, ids = eng.search(np.zeros((1, 768), dtype="float32"), 1)
    assert list(ids[0]) == [-1]        # boş kaldı (disk boş → adopt edilmedi)
    assert eng._index is mem            # disk ntotal=0 → değiştirilmedi
