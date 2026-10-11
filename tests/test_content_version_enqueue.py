"""File history enqueue must preserve Mongo identity and the current head."""
import mongomock
import pytest
from bson import ObjectId
from services.content_store import ContentDocumentStore, IndexJobConflictError


def store_with_head():
    store = ContentDocumentStore.__new__(ContentDocumentStore)
    store.documents = mongomock.MongoClient()["tinnten-test"]["contentdocuments"]
    row = {"_id": ObjectId(), "companyId": None, "documentId": None,
           "companyid": None, "currentVersionId": ObjectId(),
           "index": {"attempt": 4, "jobId": "job-4", "state": "indexed"}}
    store.documents.insert_one(row)
    return store, row


def enqueue(store, row, **changes):
    args = dict(company_id=None, document_id=str(row["_id"]),
                source={"type": "text", "text": "current content"},
                metadata={"versionId": str(row["currentVersionId"])},
                options={}, job_id="job-4", attempt=4)
    args.update(changes)
    return store.upsert_document_with_source(**args)


@pytest.mark.parametrize("changes", [
    {"job_id": "old-job", "attempt": 3},
    {"job_id": "other-job", "attempt": 4},
    {"job_id": "unversioned-job", "attempt": None},
])
def test_personal_mongoose_row_rejects_stale_enqueue_without_duplicate(changes):
    store, row = store_with_head()
    with pytest.raises(IndexJobConflictError):
        enqueue(store, row, **changes)
    assert store.documents.count_documents({}) == 1
    assert store.documents.find_one({"_id": row["_id"]}) == row


def test_valid_enqueue_keeps_personal_mongoose_identity():
    store, row = store_with_head()
    for job_id, attempt in [("job-4", 4), ("job-5", 5)]:
        updated = enqueue(store, row, job_id=job_id, attempt=attempt)
        assert updated["_id"] == row["_id"]
        assert updated["documentId"] is None
        assert updated["index"]["jobId"] == job_id
        assert store.documents.count_documents({}) == 1


def test_old_version_cannot_enqueue_even_with_a_larger_attempt():
    store, row = store_with_head()
    with pytest.raises(IndexJobConflictError):
        enqueue(store, row, job_id="old-head", attempt=10,
                metadata={"versionId": str(ObjectId())})
    assert store.documents.find_one({"_id": row["_id"]}) == row


def test_head_change_between_read_and_write_rejects_enqueue(monkeypatch):
    store, row = store_with_head()
    original = store.documents.update_one
    newer = ObjectId()
    def change_head(query, update, **kwargs):
        original({"_id": row["_id"]}, {"$set": {"currentVersionId": newer}})
        return original(query, update, **kwargs)
    monkeypatch.setattr(store.documents, "update_one", change_head)
    with pytest.raises(IndexJobConflictError):
        enqueue(store, row, job_id="delayed", attempt=10)
    assert store.documents.count_documents({}) == 1
    assert store.documents.find_one({"_id": row["_id"]})["index"] == row["index"]
