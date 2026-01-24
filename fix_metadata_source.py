import os
from dotenv import load_dotenv
from pymongo import MongoClient
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db():
    from init.db import get_database
    # Assuming standard names if not set, but using existing helpers is better if possible.
    # We'll just rely on MongoStore logic effectively by connecting similarly.
    # Or just use the env vars
    return get_database("tinnten-embedding"), get_database("tinnten")

def fix_metadata():
    chunk_db, doc_db = get_db()
    
    chunks_coll = chunk_db["embedding_chunks"]
    docs_coll = doc_db["embedding_documents"]
    
    # 1. Update embedding_chunks (top-level source)
    # Target: source != "web" (or specifically deu_Latn, but user said make them web)
    # We'll target specifically deu_Latn to be safe, or anything that looks like a dataset.
    # User said "metada.source.deu_Latn olanlar".
    
    target_source = "deu_Latn"
    new_source = "web"
    
    logger.info(f"Updating chunks and documents with source='{target_source}' to '{new_source}'...")
    
    # Update chunks top-level source
    res = chunks_coll.update_many(
        {"source": target_source},
        {"$set": {"source": new_source}}
    )
    logger.info(f"Chunks (top-level source): Matched {res.matched_count}, Modified {res.modified_count}")

    # Update chunks metadata.source
    res = chunks_coll.update_many(
        {"metadata.source": target_source},
        {"$set": {"metadata.source": new_source}}
    )
    logger.info(f"Chunks (metadata.source): Matched {res.matched_count}, Modified {res.modified_count}")

    # Update documents metadata.source
    res = docs_coll.update_many(
        {"metadata.source": target_source},
        {"$set": {"metadata.source": new_source}}
    )
    logger.info(f"Documents (metadata.source): Matched {res.matched_count}, Modified {res.modified_count}")

    # Also update documents top-level source if it exists
    res = docs_coll.update_many(
        {"source": target_source},
        {"$set": {"source": new_source}}
    )
    logger.info(f"Documents (top-level source): Matched {res.matched_count}, Modified {res.modified_count}")

if __name__ == "__main__":
    fix_metadata()
