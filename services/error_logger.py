"""
Append-only NDJSON error logger for embedding service.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_ERROR_LOG_PATH = os.getenv("EMBEDDING_ERROR_LOG_PATH") or "./logs/embedding-errors.ndjson"


class EmbeddingErrorLogger:
    _file_lock = threading.Lock()

    def __init__(self, file_path: Optional[str] = None, component: str = "api") -> None:
        self.file_path = file_path or DEFAULT_ERROR_LOG_PATH
        self.component = str(component or "api")

    def append(self, entry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": "error",
            "service": "tinnten-embedding",
            "component": self.component,
        }
        if isinstance(entry, dict):
            payload.update(entry)

        absolute_path = Path(self.file_path)
        if not absolute_path.is_absolute():
            absolute_path = Path.cwd() / absolute_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(payload, ensure_ascii=True) + "\n"
        with self._file_lock:
            with absolute_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        return payload
