import logging
import threading
import time

from workers.ingest_worker import IngestWorker

logger = logging.getLogger("tinnten.embedding.worker")

_worker_lock = threading.Lock()
_worker_thread: threading.Thread | None = None

# Retry constants
_RESTART_DELAY_SECONDS = 5
_MAX_RESTART_DELAY_SECONDS = 60
_RESTART_BACKOFF_FACTOR = 2


def _run_ingest_worker_with_retry() -> None:
    """Run the ingest worker with automatic restart on crash."""
    delay = _RESTART_DELAY_SECONDS
    while True:
        try:
            logger.info("Ingest worker starting...")
            worker = IngestWorker()
            worker.start()
            # If start() returns without exception, worker stopped cleanly
            logger.info("Ingest worker stopped cleanly.")
            break
        except Exception as exc:
            logger.error(
                "Ingest worker crashed: %s — restarting in %ss...",
                exc,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * _RESTART_BACKOFF_FACTOR, _MAX_RESTART_DELAY_SECONDS)


def _watchdog_loop() -> None:
    """Periodically checks if the ingest worker thread is alive; restarts if not."""
    global _worker_thread
    while True:
        time.sleep(10)
        with _worker_lock:
            if _worker_thread is None:
                continue
            if not _worker_thread.is_alive():
                logger.warning("Ingest worker thread died — restarting...")
                _worker_thread = threading.Thread(
                    target=_run_ingest_worker_with_retry,
                    name="IngestWorker",
                    daemon=True,
                )
                _worker_thread.start()
                logger.info("Ingest worker restarted by watchdog.")


def start_all_workers() -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            logger.info("Ingest worker is already running.")
            return

        _worker_thread = threading.Thread(
            target=_run_ingest_worker_with_retry,
            name="IngestWorker",
            daemon=True,
        )
        _worker_thread.start()
        logger.info("Started ingest worker thread.")

    # Start watchdog in background
    watchdog = threading.Thread(target=_watchdog_loop, name="WorkerWatchdog", daemon=True)
    watchdog.start()
    logger.info("Started worker watchdog thread.")


if __name__ == "__main__":
    start_all_workers()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping workers...")
