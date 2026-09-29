from unittest.mock import MagicMock, patch

import mongomock
import numpy as np
import pytest

from services.index_quota import index_quota_operation_key, mutate_index_quota
from services.mongo_store import MongoStore
from services.tinnten_server_client import (
    QuotaExceededError,
    QuotaServiceError,
    TinntenServerClient,
)
from workers.ingest_worker import IngestWorker, RetryableIngestLockError


def _response(status, payload):
    response = MagicMock(status_code=status, content=b"{}", text=str(payload))
    response.json.return_value = payload
    return response


def test_index_operation_key_is_stable_and_metric_scoped():
    first = index_quota_operation_key("C1", "D1", "V1", "documents")
    assert first == index_quota_operation_key("C1", "D1", "V1", "documents")
    assert first != index_quota_operation_key("C1", "D1", "V1", "vectorBytes")
    assert first != index_quota_operation_key(
        "C1", "D1", "V1", "documents", purpose="remove"
    )


def test_embedding_owner_metadata_supports_server_reconciliation():
    client = MagicMock()
    with patch("services.index_quota.get_tinnten_server_client", return_value=client):
        mutate_index_quota(
            metric="documents",
            action="reserve",
            company_id="C1",
            document_id="D1",
            revision="job:J1",
            amount=1,
            metadata={"vectorDimension": 1536},
        )

    payload = client.mutate_quota.call_args.kwargs
    assert payload["owner_type"] == "embedding_document"
    assert payload["owner_id"] == "D1"
    assert payload["metadata"] == {
        "vectorDimension": 1536,
        "companyId": "C1",
        "documentId": "D1",
        "revision": "job:J1",
        "purpose": "swap",
        "indexIdentity": "company:C1",
    }


def test_quota_client_uses_index_metric_endpoint_and_integer_bytes():
    client = TinntenServerClient.__new__(TinntenServerClient)
    client.base_url = "http://server"
    client.timeout = 3
    client._build_headers = MagicMock(return_value={"Authorization": "Bearer token"})
    with patch(
        "services.tinnten_server_client.requests.post",
        return_value=_response(200, {"success": True, "data": {"state": "reserved"}}),
    ) as post:
        client.mutate_quota(
            section="index",
            metric="vectorBytes",
            action="reserve",
            company_id="C1",
            operation_key="index-swap:abc",
            amount=3072,
            bootstrap_used=4096,
        )

    assert post.call_args.args[0] == (
        "http://server/api/v10/internal/quotas/index/vectorBytes/reserve"
    )
    assert post.call_args.kwargs["json"]["amount"] == 3072
    assert post.call_args.kwargs["json"]["bootstrapUsed"] == 4096


def test_quota_client_sends_bootstrap_for_first_release():
    client = TinntenServerClient.__new__(TinntenServerClient)
    client.base_url = "http://server"
    client.timeout = 3
    client._build_headers = MagicMock(return_value={"Authorization": "Bearer token"})
    with patch(
        "services.tinnten_server_client.requests.post",
        return_value=_response(200, {"success": True, "data": {"state": "released"}}),
    ) as post:
        client.mutate_quota(
            section="index",
            metric="vectorBytes",
            action="release",
            company_id="C1",
            operation_key="index-remove:abc",
            amount=1024,
            bootstrap_used=8192,
        )

    assert post.call_args.args[0].endswith("/index/vectorBytes/release")
    assert post.call_args.kwargs["json"]["bootstrapUsed"] == 8192


def test_quota_client_does_not_treat_generic_reserve_conflict_as_limit():
    client = TinntenServerClient.__new__(TinntenServerClient)
    client.base_url = "http://server"
    client.timeout = 3
    client._build_headers = MagicMock(return_value={})
    with patch(
        "services.tinnten_server_client.requests.post",
        return_value=_response(
            409,
            {"message": "drift", "data": {"code": "QUOTA_COUNTER_DRIFT"}},
        ),
    ), pytest.raises(QuotaServiceError):
        client.mutate_quota(
            section="index",
            metric="documents",
            action="reserve",
            company_id="C1",
            operation_key="index-swap:abc",
            amount=1,
        )


def test_active_index_usage_counts_only_active_version_float32_bytes():
    store = MongoStore.__new__(MongoStore)
    client = mongomock.MongoClient()
    store.documents = client.db.embedding_documents
    store.chunks = client.db.embedding_chunks
    store.documents.insert_many(
        [
            {"doc_id": "D1", "company_id": "C1", "status": "ready", "active_ingest_version": "v2"},
            {"doc_id": "D2", "company_id": "C1", "status": "disabled", "active_ingest_version": "v1"},
        ]
    )
    store.chunks.insert_many(
        [
            {"doc_id": "D1", "company_id": "C1", "ingest_version": "v1", "embedding_dimension": 99},
            {"doc_id": "D1", "company_id": "C1", "ingest_version": "v2", "embedding_dimension": 4},
            {"doc_id": "D1", "company_id": "C1", "ingest_version": "v2", "embedding_dimension": 4},
            {"doc_id": "D2", "company_id": "C1", "ingest_version": "v1", "embedding_dimension": 8},
        ]
    )

    assert store.get_company_active_index_usage("C1") == {
        "documents": 1,
        "vectorBytes": 32,
    }


def test_document_removal_quota_intent_survives_status_change_until_cleared():
    store = MongoStore.__new__(MongoStore)
    store.documents = mongomock.MongoClient().db.embedding_documents
    store.documents.insert_one({"doc_id": "D1", "status": "ready"})
    snapshots = [
        {
            "company_id": "C1",
            "revision": "version:v1",
            "documents": 1,
            "vectorBytes": 32,
        }
    ]

    pending = store.stage_document_quota_release(
        "D1", purpose="remove", snapshots=snapshots
    )
    store.update_document_status("D1", status="removed")

    assert pending["purpose"] == "remove"
    assert store.get_document("D1")["quota_release_pending"]["snapshots"] == snapshots
    assert store.clear_document_quota_release("D1", purpose="remove") is True
    assert "quota_release_pending" not in store.get_document("D1")


def test_pre_swap_quota_intent_is_durable_before_remote_reservation():
    store = MongoStore.__new__(MongoStore)
    store.documents = mongomock.MongoClient().db.embedding_documents
    store.documents.insert_one({"doc_id": "D1", "status": "processing"})
    context = {
        "company_id": "C1",
        "doc_id": "D1",
        "revision": "job:J1",
        "deltas": {"documents": 1, "vectorBytes": 32},
        "planned": ["documents", "vectorBytes"],
        "reserved": [],
    }

    pending = store.stage_index_quota_swap_pending("D1", context)

    assert pending["phase"] == "reserving"
    assert pending["planned"] == ["documents", "vectorBytes"]
    assert store.get_document("D1")["quota_swap_pending"]["revision"] == "job:J1"


def test_document_removal_releases_with_pre_removal_company_bootstrap(
    app_with_mocks, monkeypatch
):
    import app as embedding_app

    quota_store = MagicMock()
    quota_store.mark_document_index_quota_released.return_value = True
    monkeypatch.setattr(embedding_app, "chunk_store", quota_store)
    intent = {
        "purpose": "remove",
        "snapshots": [
            {
                "company_id": "C1",
                "revision": "version:v1",
                "documents": 1,
                "vectorBytes": 32,
                "bootstrap": {"documents": 7, "vectorBytes": 8192},
            }
        ],
    }

    with patch.object(embedding_app, "mutate_index_quota") as mutate:
        result = embedding_app._release_document_index_quota("D1", intent)

    assert result["ok"] is True
    assert [call.kwargs["bootstrap_used"] for call in mutate.call_args_list] == [
        7,
        8192,
    ]
    quota_store.mark_document_index_quota_released.assert_called_once_with("D1")
    quota_store.clear_document_quota_release.assert_called_once_with(
        "D1", purpose="remove"
    )


def test_disabled_document_chunks_are_not_treated_as_active_quota():
    store = MongoStore.__new__(MongoStore)
    client = mongomock.MongoClient()
    store.documents = client.db.embedding_documents
    store.chunks = client.db.embedding_chunks
    store.documents.insert_one(
        {"doc_id": "D1", "status": "disabled", "active_ingest_version": "v1"}
    )
    store.chunks.insert_one(
        {"doc_id": "D1", "ingest_version": "v1", "embedding_dimension": 4}
    )

    assert store.get_active_chunks_by_doc("D1") == []
    store.documents.update_one(
        {"doc_id": "D1"},
        {"$set": {"status": "processing", "index_quota_released": True}},
    )
    assert store.get_active_chunks_by_doc("D1") == []


def _quota_worker(store):
    worker = IngestWorker.__new__(IngestWorker)
    worker.index_quota_enabled = True
    worker.quota_fail_open = False
    worker._get_store = MagicMock(return_value=store)
    return worker


def test_swap_quota_revision_is_stable_for_job_or_vector_payload():
    embeddings = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert IngestWorker._index_quota_revision(
        embeddings, expected_job_id="job-1"
    ) == "job:job-1"
    assert IngestWorker._index_quota_revision(
        embeddings
    ) == IngestWorker._index_quota_revision(embeddings.copy())
    assert IngestWorker._index_quota_revision(embeddings) != (
        IngestWorker._index_quota_revision(embeddings + 1)
    )


def test_swap_reserves_growth_before_faiss_and_commits_after_activation():
    store = MagicMock()
    store.get_active_chunks_by_doc.return_value = []
    store.logical_vector_bytes.return_value = 0
    store.get_company_active_index_usage.return_value = {
        "documents": 5,
        "vectorBytes": 1000,
    }
    store.reserve_faiss_ids.return_value = [10, 11]
    store.set_active_ingest_version.return_value = True
    store.get_faiss_ids_by_doc_except_version.return_value = []
    store.delete_chunks_by_doc_except_version.return_value = 0
    worker = _quota_worker(store)
    engine = MagicMock()
    embeddings = np.zeros((2, 6), dtype=np.float32)

    with patch("workers.ingest_worker.mutate_index_quota") as mutate:
        worker._swap_in_chunks(
            doc_id="D1",
            company_id="C1",
            engine=engine,
            embeddings=embeddings,
            build_chunk_docs=lambda ids, version: [
                {"doc_id": "D1", "faiss_id": fid, "ingest_version": version}
                for fid in ids
            ],
            chunk_count=2,
        )

    actions = [call.kwargs["action"] for call in mutate.call_args_list]
    metrics = [call.kwargs["metric"] for call in mutate.call_args_list]
    assert actions == ["reserve", "reserve", "commit", "commit"]
    assert metrics == ["documents", "vectorBytes", "documents", "vectorBytes"]
    assert mutate.call_args_list[1].kwargs["amount"] == 48  # 2 * 6 * float32
    store.stage_index_quota_swap_pending.assert_called_once()
    assert store.set_active_ingest_version.call_count == 1
    engine.add_embeddings.assert_called_once()


def test_quota_block_happens_before_faiss_mutation():
    store = MagicMock()
    worker = _quota_worker(store)
    worker._prepare_index_quota_change = MagicMock(
        side_effect=QuotaExceededError("full", data={"metric": "vectorBytes"})
    )
    engine = MagicMock()

    with pytest.raises(QuotaExceededError):
        worker._swap_in_chunks(
            doc_id="D1",
            company_id="C1",
            engine=engine,
            embeddings=np.zeros((1, 4), dtype=np.float32),
            build_chunk_docs=lambda _ids, _version: [],
            chunk_count=1,
        )

    store.reserve_faiss_ids.assert_not_called()
    engine.add_embeddings.assert_not_called()


def test_ambiguous_activation_never_cancels_or_deletes_the_persisted_version():
    store = MagicMock()
    store.get_document.side_effect = [
        {},
        {
            "active_ingest_version": "new-version",
            "quota_swap_pending": {
                "phase": "activated",
                "revision": "job:J1",
            },
        },
    ]
    store.get_active_chunks_by_doc.return_value = []
    store.logical_vector_bytes.return_value = 0
    store.get_company_active_index_usage.return_value = {
        "documents": 0,
        "vectorBytes": 0,
    }
    store.reserve_faiss_ids.return_value = [10]
    store.set_active_ingest_version.side_effect = RuntimeError("ack lost")
    worker = _quota_worker(store)
    engine = MagicMock()

    with patch("workers.ingest_worker.uuid.uuid4") as uuid4, patch(
        "workers.ingest_worker.mutate_index_quota"
    ) as mutate:
        uuid4.return_value.hex = "new-version"
        with pytest.raises(RetryableIngestLockError):
            worker._swap_in_chunks(
                doc_id="D1",
                company_id="C1",
                engine=engine,
                embeddings=np.zeros((1, 4), dtype=np.float32),
                build_chunk_docs=lambda ids, version: [
                    {
                        "doc_id": "D1",
                        "faiss_id": ids[0],
                        "ingest_version": version,
                    }
                ],
                chunk_count=1,
                expected_job_id="J1",
            )

    assert [call.kwargs["action"] for call in mutate.call_args_list] == [
        "reserve",
        "reserve",
    ]
    engine.remove_ids.assert_not_called()
    store.delete_chunks_by_doc_version.assert_not_called()


def test_successful_smaller_swap_releases_only_logical_byte_delta():
    store = MagicMock()
    old_chunks = [{"embedding_dimension": 6} for _ in range(3)]
    store.get_active_chunks_by_doc.return_value = old_chunks
    store.logical_vector_bytes.return_value = 72
    store.get_company_active_index_usage.return_value = {
        "documents": 4,
        "vectorBytes": 4096,
    }
    store.reserve_faiss_ids.return_value = [20, 21]
    store.set_active_ingest_version.return_value = True
    store.get_faiss_ids_by_doc_except_version.return_value = [1, 2, 3]
    store.delete_chunks_by_doc_except_version.return_value = 3
    worker = _quota_worker(store)
    engine = MagicMock()

    with patch("workers.ingest_worker.mutate_index_quota") as mutate:
        worker._swap_in_chunks(
            doc_id="D1",
            company_id="C1",
            engine=engine,
            embeddings=np.zeros((2, 6), dtype=np.float32),
            build_chunk_docs=lambda ids, version: [
                {"doc_id": "D1", "faiss_id": fid, "ingest_version": version}
                for fid in ids
            ],
            chunk_count=2,
        )

    mutate.assert_called_once()
    assert mutate.call_args.kwargs["action"] == "release"
    assert mutate.call_args.kwargs["metric"] == "vectorBytes"
    assert mutate.call_args.kwargs["amount"] == 24
    assert mutate.call_args.kwargs["bootstrap_used"] == 4096
    assert mutate.call_args.kwargs["purpose"] == "swap-shrink"


def test_pending_post_swap_quota_is_recovered_idempotently():
    context = {
        "company_id": "C1",
        "doc_id": "D1",
        "revision": "job:J1",
        "deltas": {"documents": 1, "vectorBytes": -16},
        "reserved": ["documents"],
        "metadata": {},
    }
    store = MagicMock()
    store.get_document.return_value = {"quota_swap_pending": context}
    worker = _quota_worker(store)

    with patch("workers.ingest_worker.mutate_index_quota") as mutate:
        worker._recover_pending_index_quota("D1")

    assert [call.kwargs["action"] for call in mutate.call_args_list] == [
        "commit",
        "release",
    ]
    store.clear_index_quota_swap_pending.assert_called_once_with("D1", "job:J1")


def test_pending_pre_swap_quota_cancels_every_planned_metric_without_usage_release():
    context = {
        "phase": "reserving",
        "company_id": "C1",
        "doc_id": "D1",
        "revision": "job:J1",
        "deltas": {"documents": 1, "vectorBytes": 32},
        "planned": ["documents", "vectorBytes"],
        # The process may have died before it could append either result.
        "reserved": [],
        "bootstrap": {"documents": 2, "vectorBytes": 64},
        "metadata": {},
    }
    store = MagicMock()
    store.get_document.return_value = {"quota_swap_pending": context}
    worker = _quota_worker(store)

    with patch("workers.ingest_worker.mutate_index_quota") as mutate:
        worker._recover_pending_index_quota("D1")

    assert [call.kwargs["action"] for call in mutate.call_args_list] == [
        "cancel-reservation",
        "cancel-reservation",
    ]
    assert [call.kwargs["metric"] for call in mutate.call_args_list] == [
        "documents",
        "vectorBytes",
    ]
    store.clear_index_quota_swap_pending.assert_called_once_with("D1", "job:J1")


def test_pending_quota_document_scan_is_bounded_and_includes_both_intents():
    store = MongoStore.__new__(MongoStore)
    store.documents = mongomock.MongoClient().db.embedding_documents
    store.documents.insert_many(
        [
            {"doc_id": "D1", "quota_swap_pending": {"revision": "r1"}},
            {
                "doc_id": "D2",
                "quota_release_pending": {"purpose": "remove", "snapshots": []},
            },
            {"doc_id": "D3", "status": "ready"},
        ]
    )

    first = store.list_pending_index_quota_documents(limit=1)
    all_pending = store.list_pending_index_quota_documents(limit=10)

    assert len(first) == 1
    assert {row["doc_id"] for row in all_pending} == {"D1", "D2"}


def test_periodic_recovery_replays_swap_and_document_release_intents():
    swap_context = {
        "company_id": "C1",
        "doc_id": "D1",
        "revision": "job:J1",
        "deltas": {"documents": 1, "vectorBytes": 0},
        "reserved": ["documents"],
        "bootstrap": {"documents": 2},
        "metadata": {},
    }
    release_intent = {
        "purpose": "remove",
        "snapshots": [
            {
                "company_id": "C1",
                "revision": "version:v1",
                "documents": 1,
                "vectorBytes": 32,
                "bootstrap": {"documents": 3, "vectorBytes": 96},
            }
        ],
    }
    documents = {
        "D1": {"doc_id": "D1", "quota_swap_pending": swap_context},
        "D2": {"doc_id": "D2", "quota_release_pending": release_intent},
    }
    store = MagicMock()
    store.list_pending_index_quota_documents.return_value = list(documents.values())
    store.get_document.side_effect = lambda doc_id: documents.get(doc_id, {})
    store.clear_index_quota_swap_pending.side_effect = lambda doc_id, _revision: (
        documents[doc_id].pop("quota_swap_pending", None) is not None
    )
    store.mark_document_index_quota_released.return_value = True
    store.clear_document_quota_release.side_effect = lambda doc_id, **_kwargs: (
        documents[doc_id].pop("quota_release_pending", None) is not None
    )
    worker = _quota_worker(store)

    with patch("workers.ingest_worker.mutate_index_quota") as mutate:
        summary = worker.recover_pending_index_quotas_once(limit=25)

    assert summary == {
        "documents": 2,
        "swapRecovered": 1,
        "releaseRecovered": 1,
        "failed": 0,
    }
    store.list_pending_index_quota_documents.assert_called_once_with(25)
    assert [(call.kwargs["action"], call.kwargs["metric"]) for call in mutate.call_args_list] == [
        ("commit", "documents"),
        ("release", "documents"),
        ("release", "vectorBytes"),
    ]
    assert mutate.call_args_list[1].kwargs["bootstrap_used"] == 3
    assert mutate.call_args_list[2].kwargs["bootstrap_used"] == 96
    store.mark_document_index_quota_released.assert_called_once_with("D2")


def test_periodic_release_failure_keeps_durable_intent_for_next_pass():
    release_intent = {
        "purpose": "disable",
        "snapshots": [
            {
                "company_id": "C1",
                "revision": "version:v1",
                "documents": 1,
                "vectorBytes": 0,
            }
        ],
    }
    document = {"doc_id": "D1", "quota_release_pending": release_intent}
    store = MagicMock()
    store.list_pending_index_quota_documents.return_value = [document]
    store.get_document.return_value = document
    worker = _quota_worker(store)

    with patch(
        "workers.ingest_worker.mutate_index_quota",
        side_effect=QuotaServiceError("offline"),
    ):
        summary = worker.recover_pending_index_quotas_once(limit=10)

    assert summary["failed"] == 1
    assert summary["releaseRecovered"] == 0
    store.mark_document_index_quota_released.assert_not_called()
    store.clear_document_quota_release.assert_not_called()
    store.touch_pending_index_quota_document.assert_called_once_with("D1")


def test_index_quota_recovery_loop_runs_a_startup_pass():
    worker = IngestWorker.__new__(IngestWorker)
    worker._stop_event = MagicMock()
    worker._stop_event.is_set.return_value = False
    worker._stop_event.wait.return_value = True
    worker.recover_pending_index_quotas_once = MagicMock(
        return_value={
            "documents": 0,
            "swapRecovered": 0,
            "releaseRecovered": 0,
            "failed": 0,
        }
    )

    worker._index_quota_recovery_loop()

    worker.recover_pending_index_quotas_once.assert_called_once_with()
    worker._stop_event.wait.assert_called_once_with(60.0)
