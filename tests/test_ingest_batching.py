"""
Toplu (batch) ingest testleri.

Neden var: firma FAISS index'i canlıda ~200MB ve her `add_embeddings` /
`remove_ids` dosyanın TAMAMINI yeniden yazıyordu → doküman başına 2 tam yazım,
ingest'in tek darboğazı. `batch_writes` bu yazımları tek seferde topluyor.

Buradaki testler iki sözleşmeyi kilitler:
  1) Blok içinde diske YAZILMAZ, çıkışta TEK yazım olur (kazancın kendisi).
  2) Ack'ler blok ÇIKIŞINDAN SONRA verilir (dayanıklılık): erken ack, çıkıştan
     önceki bir çökmede Mongo'da "tamamlandı" görünen ama index'te bulunmayan
     chunk bırakırdı.
"""
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import faiss
import numpy as np


DIM = 8


def _new_engine(index_path: str):
    from services.embedding_engine import EmbeddingEngine

    EmbeddingEngine._model_cache.clear()
    with patch("services.embedding_engine.SentenceTransformer") as MockST:
        MockST.side_effect = lambda name: MagicMock(name=f"model::{name}")
        return EmbeddingEngine(model_name="batch-test-model", index_path=index_path)


def _vec(seed: int) -> np.ndarray:
    return (np.ones((1, DIM), dtype=np.float32) * float(seed))


# ----------------------------------------------------------------------
# 1) Engine: yazım erteleme
# ----------------------------------------------------------------------
def test_batch_writes_saves_index_once(tmp_path):
    """Blok içindeki N mutasyon → TEK diske yazım."""
    engine = _new_engine(str(tmp_path / "company" / "c1.index"))

    with patch.object(engine, "_save_index", wraps=engine._save_index) as save_spy:
        with engine.batch_writes():
            engine.add_embeddings(_vec(1), [101])
            engine.add_embeddings(_vec(2), [102])
            engine.add_embeddings(_vec(3), [103])
        assert save_spy.call_count == 1


def test_without_batch_every_mutation_saves(tmp_path):
    """Karşılaştırma (bugünkü davranış): batch yokken her mutasyon yazıyor."""
    engine = _new_engine(str(tmp_path / "company" / "c2.index"))

    with patch.object(engine, "_save_index", wraps=engine._save_index) as save_spy:
        engine.add_embeddings(_vec(1), [201])
        engine.add_embeddings(_vec(2), [202])
        assert save_spy.call_count == 2


def test_batch_defers_disk_write_until_exit(tmp_path):
    """Blok İÇİNDE disk değişmemeli; çıkışta hepsi kalıcı olmalı."""
    index_path = str(tmp_path / "company" / "c3.index")
    engine = _new_engine(index_path)

    with engine.batch_writes():
        engine.add_embeddings(_vec(1), [301])
        engine.add_embeddings(_vec(2), [302])
        # Henüz çıkmadık: diskte ya dosya yok ya da boş.
        on_disk = faiss.read_index(index_path).ntotal if os.path.exists(index_path) else 0
        assert on_disk == 0, "batch bitmeden diske yazılmamalı"

    assert faiss.read_index(index_path).ntotal == 2, "çıkışta tüm vektörler kalıcı olmalı"


def test_nested_batch_writes_saves_once(tmp_path):
    """İç içe blok tek yazım yapar ve kilitte kendini kilitlemez (deadlock yok)."""
    engine = _new_engine(str(tmp_path / "company" / "c4.index"))

    with patch.object(engine, "_save_index", wraps=engine._save_index) as save_spy:
        with engine.batch_writes():
            engine.add_embeddings(_vec(1), [401])
            with engine.batch_writes():
                engine.add_embeddings(_vec(2), [402])
            assert save_spy.call_count == 0, "iç blok çıkışı erken yazmamalı"
        assert save_spy.call_count == 1


def test_remove_ids_inside_batch_is_deferred(tmp_path):
    """Stale temizliği (remove_ids) de aynı yazımda toplanmalı."""
    index_path = str(tmp_path / "company" / "c5.index")
    engine = _new_engine(index_path)
    engine.add_embeddings(_vec(1), [501])  # batch dışı: hemen yazılır

    with patch.object(engine, "_save_index", wraps=engine._save_index) as save_spy:
        with engine.batch_writes():
            engine.add_embeddings(_vec(2), [502])
            engine.remove_ids([501])
        assert save_spy.call_count == 1

    assert faiss.read_index(index_path).ntotal == 1


def test_failed_save_keeps_batch_dirty(tmp_path):
    """Kayıt patlarsa hata yukarı gider ve 'kirli' bayrağı düşmez.

    Düşseydi vektörler bellekte kalır, bir daha hiç kaydedilmezdi.
    """
    engine = _new_engine(str(tmp_path / "company" / "c6.index"))

    with patch.object(engine, "_save_index", side_effect=OSError("disk full")):
        try:
            with engine.batch_writes():
                engine.add_embeddings(_vec(1), [601])
        except OSError:
            pass
        else:
            raise AssertionError("save hatası çağırana ulaşmalıydı")

    assert engine._batch_dirty is True
    assert engine._batch_depth == 0


# ----------------------------------------------------------------------
# 2) Worker: gruplama ve ack sırası
# ----------------------------------------------------------------------
def _make_worker(mocker):
    mocker.patch("workers.ingest_worker.EmbeddingEmailEvents")
    mocker.patch("workers.ingest_worker.EmbeddingErrorLogger")
    from workers.ingest_worker import IngestWorker

    worker = IngestWorker()
    worker.channel = MagicMock()
    worker.channel.is_open = True
    return worker


def _wire(worker, mocker, events, *, fail_on=None):
    """Engine'i sahte batch context'iyle, işlemeyi olay kaydıyla değiştirir."""

    @contextmanager
    def fake_batch():
        events.append("batch:enter")
        yield
        events.append("batch:exit")

    engine = MagicMock()
    engine.batch_writes.side_effect = fake_batch
    mocker.patch.object(worker, "_engine_for_company", return_value=engine)

    def process(payload):
        doc = payload["documentIds"][0]
        if fail_on and doc == fail_on:
            raise RuntimeError(f"bozuk doküman: {doc}")
        events.append(f"process:{doc}")

    mocker.patch.object(worker, "_process_payload", side_effect=process)
    mocker.patch.object(worker, "_report_message_failure")
    worker.channel.basic_ack.side_effect = lambda delivery_tag: events.append(f"ack:{delivery_tag}")
    worker.channel.basic_nack.side_effect = lambda delivery_tag, requeue: events.append(
        f"nack:{delivery_tag}:requeue={requeue}"
    )
    return engine


def test_acks_happen_only_after_batch_exit(mocker):
    """DAYANIKLILIK SÖZLEŞMESİ: ack'ler index diske indikten SONRA."""
    worker = _make_worker(mocker)
    events = []
    _wire(worker, mocker, events)

    worker._pending = [
        (1, {"companyId": "A", "documentIds": ["d1"]}),
        (2, {"companyId": "A", "documentIds": ["d2"]}),
    ]
    worker._flush_pending()

    assert events == [
        "batch:enter",
        "process:d1",
        "process:d2",
        "batch:exit",
        "ack:1",
        "ack:2",
    ]


def test_messages_are_grouped_per_company(mocker):
    """Farklı firmalar ayrı bloklarda işlenir (her firmanın index'i ayrı dosya)."""
    worker = _make_worker(mocker)
    events = []
    _wire(worker, mocker, events)

    worker._pending = [
        (1, {"companyId": "A", "documentIds": ["a1"]}),
        (2, {"companyId": "B", "documentIds": ["b1"]}),
        (3, {"companyId": "A", "documentIds": ["a2"]}),
    ]
    worker._flush_pending()

    assert events.count("batch:enter") == 2, "firma başına bir blok"
    # A grubu birlikte işlenmeli (a1 ve a2 aynı blokta, arada exit olmadan)
    a1, a2 = events.index("process:a1"), events.index("process:a2")
    assert "batch:exit" not in events[a1:a2]


def test_failing_message_does_not_drop_siblings(mocker):
    """Tek mesajın hatası grubu düşürmez: kendisi nack'lenir, kardeşleri ack."""
    worker = _make_worker(mocker)
    events = []
    _wire(worker, mocker, events, fail_on="bad")

    worker._pending = [
        (1, {"companyId": "A", "documentIds": ["ok1"]}),
        (2, {"companyId": "A", "documentIds": ["bad"]}),
        (3, {"companyId": "A", "documentIds": ["ok2"]}),
    ]
    worker._flush_pending()

    assert "process:ok1" in events and "process:ok2" in events
    assert "nack:2:requeue=False" in events
    assert "ack:1" in events and "ack:3" in events
    assert "ack:2" not in events


def test_batch_save_failure_requeues_successful_messages(mocker):
    """Toplu kayıt patlarsa başarılı mesajlar ack DEĞİL, requeue edilir.

    Ack'lenseydi vektörleri diske inmemiş dokümanlar sessizce kaybolurdu.
    """
    worker = _make_worker(mocker)
    events = []

    @contextmanager
    def exploding_batch():
        events.append("batch:enter")
        yield
        raise OSError("disk full")

    engine = MagicMock()
    engine.batch_writes.side_effect = exploding_batch
    mocker.patch.object(worker, "_engine_for_company", return_value=engine)
    mocker.patch.object(
        worker, "_process_payload", side_effect=lambda p: events.append(f"process:{p['documentIds'][0]}")
    )
    mocker.patch.object(worker, "_log_worker_error")
    worker.channel.basic_ack.side_effect = lambda delivery_tag: events.append(f"ack:{delivery_tag}")
    worker.channel.basic_nack.side_effect = lambda delivery_tag, requeue: events.append(
        f"nack:{delivery_tag}:requeue={requeue}"
    )

    worker._pending = [
        (1, {"companyId": "A", "documentIds": ["d1"]}),
        (2, {"companyId": "A", "documentIds": ["d2"]}),
    ]
    worker._flush_pending()

    assert "ack:1" not in events and "ack:2" not in events
    assert "nack:1:requeue=True" in events
    assert "nack:2:requeue=True" in events


def test_non_content_messages_bypass_batching(mocker):
    """embedded_chunks/legacy mesajları tamponlanmaz, anında işlenip ack'lenir."""
    worker = _make_worker(mocker)
    events = []
    _wire(worker, mocker, events)
    mocker.patch.object(worker, "_process_payload", side_effect=lambda p: events.append("processed"))

    method = MagicMock()
    method.delivery_tag = 7
    worker._handle_message(worker.channel, method, MagicMock(), b'{"payload_type":"embedded_chunks","doc_id":"x"}')

    assert worker._pending == [], "tamponlanmamalı"
    assert events == ["processed", "ack:7"]


def test_content_messages_are_buffered_until_batch_is_full(mocker):
    """content-index mesajları grup dolana kadar bekletilir."""
    worker = _make_worker(mocker)
    worker.batch_max_messages = 3
    events = []
    _wire(worker, mocker, events)

    method = MagicMock()
    method.delivery_tag = 1
    body = b'{"companyId":"A","documentIds":["d1"]}'
    worker._handle_message(worker.channel, method, MagicMock(), body)

    assert len(worker._pending) == 1
    assert events == [], "grup dolmadan işlenmemeli"
