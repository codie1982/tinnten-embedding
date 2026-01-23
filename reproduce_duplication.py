
import os
import sys
import uuid
import numpy as np
import shutil
from dotenv import load_dotenv

# Ensure we can import from project root
sys.path.append(os.getcwd())
load_dotenv()

from services.mongo_store import MongoStore, CHUNKS_COLL
from services.embedding_engine import EmbeddingEngine
from workers.ingest_worker import IngestWorker

# Test configuration
TEST_DOC_ID = "test-doc-repro-" + uuid.uuid4().hex
INDEX_PATH = "test_repro.index"

def cleanup():
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    if os.path.exists(INDEX_PATH + ".lock"):
        os.remove(INDEX_PATH + ".lock")
    
    store = MongoStore()
    store.delete_chunks_by_doc(TEST_DOC_ID)
    store.delete_document(TEST_DOC_ID)

def run_test():
    cleanup()
    
    print(f"--- Starting Reproduction Test (Doc ID: {TEST_DOC_ID}) ---")
    
    worker = IngestWorker()
    worker.index_path = INDEX_PATH
    # Force initialize engine
    worker._get_engine() 
    
    store = MongoStore()
    
    # Create fake chunks and embeddings
    chunks = [
        {"text": "foo", "char_start": 0, "char_end": 3, "index": 0},
        {"text": "bar", "char_start": 4, "char_end": 7, "index": 1},
    ]
    embeddings = [
        [0.1] * 768,
        [0.2] * 768
    ]
    
    payload = {
        "payload_type": "embedded_chunks",
        "doc_id": TEST_DOC_ID,
        "doc_type": "test",
        "source": "test",
        "metadata": {"test": True},
        "chunks": chunks,
        "embeddings": embeddings,
        "index_path": INDEX_PATH
    }
    
    # 1. First Ingestion
    print(">>> 1. First Ingestion")
    worker._process_embedded_chunks(payload)
    
    chunks_1 = store.get_chunks_by_doc(TEST_DOC_ID)
    idx_1 = worker._get_engine()
    count_1 = idx_1.count()
    print(f"   Chunks in DB: {len(chunks_1)}")
    print(f"   Vectors in Index: {count_1}")
    
    assert len(chunks_1) == 2
    assert count_1 == 2
    
    # 2. Second Ingestion (Duplicate)
    print(">>> 2. Second Ingestion (Duplicate)")
    # Simulate create_web_index behavior: it resets status to 'queued'
    store.update_document_status(TEST_DOC_ID, status="queued")
    
    worker._process_embedded_chunks(payload)
    
    chunks_2 = store.get_chunks_by_doc(TEST_DOC_ID)
    idx_2 = worker._get_engine()
    count_2 = idx_2.count()
    print(f"   Chunks in DB: {len(chunks_2)}")
    print(f"   Vectors in Index: {count_2}")
    
    if count_2 > count_1:
         print("FAILURE: Duplicate vectors passed to index!")
    else:
         print("SUCCESS: Vectors were replaced correctly.")

    if len(chunks_2) > 2:
        print("FAILURE: Duplicate chunks in MongoDB!")

if __name__ == "__main__":
    try:
        run_test()
    finally:
        cleanup()
