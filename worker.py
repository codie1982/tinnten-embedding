import logging
import threading
import time

from workers.ingest_worker import IngestWorker

logger = logging.getLogger("tinnten.embedding.worker")

_worker_lock = threading.Lock()
_worker_thread: threading.Thread | None = None


def _run_ingest_worker() -> None:
    worker = IngestWorker()
    worker.start()


def start_all_workers() -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            logger.info("Ingest worker is already running.")
            return

        _worker_thread = threading.Thread(
            target=_run_ingest_worker,
            name="IngestWorker",
            daemon=True,
        )
        _worker_thread.start()
        logger.info("Started ingest worker thread.")


if __name__ == "__main__":
    start_all_workers()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping workers...")
