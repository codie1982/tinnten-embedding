"""
Migration güvenlik kilitleri.

Bu script'in en tehlikeli arıza modu SESSİZ olanıydı: `_prepare_reconstruct`
var olmayan `make_direct_map()`i çağırıp hatayı yutuyor, `_build_company_index`
de her reconstruct hatasını yutuyordu. Düz bir IndexIDMap ile çalıştırıldığında
sonuç: HER FİRMA İÇİN BOŞ index dosyası + `exit 0` ("başarılı").

Buradaki testler o yolun artık gürültülü şekilde durduğunu doğrular.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from migrate_global_to_per_company_faiss import (  # noqa: E402
    MigrationAbort,
    _build_company_index,
    _prepare_reconstruct,
)

DIM = 4


def _idmap2(n=3):
    import faiss

    idx = faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))
    idx.add_with_ids(
        np.random.rand(n, DIM).astype("float32"),
        np.arange(10, 10 + n, dtype="int64"),
    )
    return idx


def _plain_idmap(n=3):
    import faiss

    idx = faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
    idx.add_with_ids(
        np.random.rand(n, DIM).astype("float32"),
        np.arange(10, 10 + n, dtype="int64"),
    )
    return idx


def test_prepare_reconstruct_accepts_idmap2():
    """Motorun yarattığı tip (IndexIDMap2) sorunsuz geçmeli."""
    _prepare_reconstruct(_idmap2())


def test_prepare_reconstruct_aborts_on_plain_idmap():
    """Düz IndexIDMap id ile reconstruct edemez → sessizce devam ETMEMELİ."""
    with pytest.raises(MigrationAbort):
        _prepare_reconstruct(_plain_idmap())


def test_prepare_reconstruct_aborts_on_empty_index():
    import faiss

    with pytest.raises(MigrationAbort):
        _prepare_reconstruct(faiss.IndexIDMap2(faiss.IndexFlatIP(DIM)))


def test_build_company_index_refuses_to_write_empty():
    """Beklenen chunk'ların hiçbiri okunamadıysa BOŞ index yazılmamalı."""
    with pytest.raises(MigrationAbort):
        _build_company_index(_idmap2(), [999, 1000], DIM)


def test_build_company_index_aborts_on_partial_loss_unless_allowed():
    """Kısmi kayıp da varsayılan olarak durdurur; --allow-missing ile zorlanabilir."""
    src = _idmap2()
    with pytest.raises(MigrationAbort):
        _build_company_index(src, [10, 11, 999], DIM)

    idx, written, missing = _build_company_index(src, [10, 11, 999], DIM, True)
    assert (written, missing) == (2, 1)
    assert idx.ntotal == 2
