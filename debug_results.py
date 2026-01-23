
import os
import sys
import json
from dotenv import load_dotenv

# Ensure we can import app and services
sys.path.append("/home/codie/Documents/developer/tinnten/tinnten-embedding")
load_dotenv()

from app import _websearch_faiss_search, WEBSEARCH_MODEL_NAME, WEBSEARCH_INDEX_PATH
from services.mongo_store import MongoStore

print(f"WEBSEARCH_MODEL_NAME: {WEBSEARCH_MODEL_NAME}")
print(f"WEBSEARCH_INDEX_PATH: {WEBSEARCH_INDEX_PATH}")
print(f"MONGO_URI: {os.getenv('MONGO_URI')}")
print(f"EMBED_DB_NAME (config): {os.getenv('EMBED_DB_NAME')}")

# 1. Search FAISS
try:
    scores, ids = _websearch_faiss_search("test query", 5, WEBSEARCH_MODEL_NAME, WEBSEARCH_INDEX_PATH)
    print("Search successful!")
    print(f"Scores: {scores}")
    print(f"IDs: {ids}")
except Exception as e:
    print(f"Search failed: {e}")
    sys.exit(1)

faiss_ids = [int(i) for i in ids if i != -1]
if not faiss_ids:
    print("No valid FAISS IDs found.")
    sys.exit(0)

# 2. Lookup in Mongo
store = MongoStore()
print(f"Connected to Mongo. DB: {store.chunk_db.name}, Coll: {store.chunks.name}")

cursor = store.chunks.find({"faiss_id": {"$in": faiss_ids}})
chunk_docs = {int(doc["faiss_id"]): doc for doc in cursor}

print(f"Found {len(chunk_docs)} docs in Mongo matching FAISS IDs.")
print(f"Requested IDs: {faiss_ids}")
print(f"Found IDs: {list(chunk_docs.keys())}")

for fid in faiss_ids:
    if fid not in chunk_docs:
        print(f"MISSING: FAISS ID {fid} not found in Mongo.")
    else:
        doc = chunk_docs[fid]
        doc_id = doc.get("doc_id")
        
        # Check document status
        doc_info = store.documents.find_one({"doc_id": doc_id})
        status = doc_info.get("status") if doc_info else "NOT_FOUND"
        
        print(f"FOUND: ID {fid} -> DocID: {doc_id} | Status: {status}")
