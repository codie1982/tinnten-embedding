"""Indexability checks for parser-backed file sources.

The crawler and inline web paths are intentionally outside this module's
policy.  A crawl may legitimately yield an empty/filtered page and has its own
retry/lifecycle semantics.  Uploaded files, on the other hand, must never be
reported as indexed when parsing produced no usable chunk input.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FILE_BACKED_SOURCES = frozenset({"upload", "local_file"})
PDF_CONTENT_TYPES = frozenset({"application/pdf", "application/x-pdf"})

# XLSX/PPTX loaders add navigation markers even when the document contains no
# cell/shape text.  Those markers describe the container, not user content.
_PARSER_SCAFFOLD_LINE = re.compile(
    r"^\s*\[(?:sheet|slide(?:\s+\d+)?)\][^\n]*$", re.IGNORECASE
)
_STRUCTURAL_EXTENSIONS = frozenset({"xlsx", "pptx"})
_KNOWN_EXTENSIONS = frozenset(
    {"txt", "md", "csv", "json", "xml", "yaml", "yml", "html", "htm", "pdf", "doc", "docx", "xlsx", "pptx"}
)
_ZERO_WIDTH_CHARS = frozenset({"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"})


@dataclass(frozen=True, slots=True)
class ExtractionMetrics:
    extracted_chars: int
    meaningful_chars: int
    min_chars: int


class NonIndexableFileContentError(RuntimeError):
    """A parsed file cannot produce a valid embedding chunk."""

    def __init__(
        self,
        reason: str,
        *,
        metrics: ExtractionMetrics,
        filename: str = "",
        content_type: str = "",
    ) -> None:
        # Keep the persisted/callback error stable and machine-readable.  Counts
        # live in index.stats and the structured worker log.
        super().__init__(reason)
        self.reason = reason
        self.metrics = metrics
        self.filename = filename
        self.content_type = content_type


def is_file_backed_source(source: Any) -> bool:
    return str(source or "").strip().lower() in FILE_BACKED_SOURCES


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalized_extension(value: Any) -> str:
    raw = str(value or "").strip().lower().lstrip(".")
    return raw if raw in _KNOWN_EXTENSIONS else ""


def _suffix_from_path(value: Any) -> str:
    raw = str(value or "").strip().split("#", 1)[0].split("?", 1)[0]
    return _normalized_extension(Path(raw).suffix)


def _resolve_extension(metadata: Mapping[str, Any]) -> str:
    """Resolve the parser's actual extension before user-facing filenames.

    ``detectedExt`` is populated by the server's byte-level detector and is
    therefore more authoritative than a claimed/original filename.  Older
    queue messages may only carry parser, filename, or storage-key metadata;
    keep those fallbacks so PDF/OCR classification survives legacy records.
    """
    preflight = metadata.get("indexPreflight")
    preflight = preflight if isinstance(preflight, Mapping) else {}
    parse_result = metadata.get("parseResult")
    parse_result = parse_result if isinstance(parse_result, Mapping) else {}

    for value in (
        metadata.get("detectedExt"),
        metadata.get("detected_ext"),
        preflight.get("detectedExt"),
        metadata.get("extension"),
        metadata.get("ext"),
        preflight.get("extension"),
    ):
        extension = _normalized_extension(value)
        if extension:
            return extension

    for value in (
        preflight.get("parser"),
        parse_result.get("parser"),
        metadata.get("parser"),
    ):
        extension = _normalized_extension(value)
        if extension:
            return extension

    for key in (
        "filename",
        "fileName",
        "name",
        "originalName",
        "originalname",
        "key",
        "storageKey",
        "s3Key",
        "localPath",
        "path",
        "url",
    ):
        extension = _suffix_from_path(metadata.get(key))
        if extension:
            return extension
    return ""


def _meaningful_text(text: Any, *, extension: str = "") -> str:
    value = text if isinstance(text, str) else ""
    if extension in _STRUCTURAL_EXTENSIONS:
        value = "\n".join(
            line for line in value.splitlines() if not _PARSER_SCAFFOLD_LINE.match(line)
        )
    normalized = "".join(
        " " if char == "\x00" else "" if char in _ZERO_WIDTH_CHARS else char
        for char in value
    )
    normalized = " ".join(normalized.split())
    # Keep the worker gate in parity with the server's /[\p{L}\p{N}]/gu
    # signal: punctuation, separators and replacement glyphs are not content.
    return "".join(char for char in normalized if char.isalnum())


def _is_pdf(metadata: Mapping[str, Any]) -> bool:
    content_type = _metadata_value(
        metadata, "contentType", "content_type", "mimeType", "mimetype"
    ).split(";", 1)[0].strip().lower()
    return _resolve_extension(metadata) == "pdf" or content_type in PDF_CONTENT_TYPES


def _ocr_was_applied(metadata: Mapping[str, Any]) -> bool:
    """Distinguish an OCR request from evidence that OCR actually ran."""
    for key in ("ocrApplied", "ocr_applied", "ocrPerformed", "ocr_performed"):
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
    parse_result = metadata.get("parseResult")
    if isinstance(parse_result, Mapping):
        return _ocr_was_applied(parse_result)
    return False


def validate_file_extraction(
    *,
    source: Any,
    text: Any,
    metadata: Mapping[str, Any] | None,
    min_chars: int,
    ocr_enabled: bool,
) -> ExtractionMetrics:
    """Validate parser output for uploaded/local files.

    Non-file sources are deliberately a no-op.  For file sources this raises a
    stable reason code before the embedding engine or either vector store is
    touched.
    """
    raw_text = text if isinstance(text, str) else ""
    meta = metadata if isinstance(metadata, Mapping) else {}
    normalized = raw_text.strip()
    meaningful = _meaningful_text(normalized, extension=_resolve_extension(meta))
    metrics = ExtractionMetrics(
        extracted_chars=len(raw_text),
        meaningful_chars=len(meaningful),
        min_chars=max(1, int(min_chars or 1)),
    )
    if not is_file_backed_source(source):
        return metrics

    filename = _metadata_value(meta, "filename", "fileName", "name")
    content_type = _metadata_value(
        meta, "contentType", "content_type", "mimeType", "mimetype"
    )

    if metrics.meaningful_chars == 0:
        # `options.ocr` is only a request flag. The current loader does not run
        # OCR itself, so an empty PDF remains actionable as `ocr_required`
        # unless upstream metadata explicitly proves OCR was performed.
        reason = (
            "ocr_required"
            if _is_pdf(meta) and not _ocr_was_applied(meta)
            else "no_extractable_text"
        )
        raise NonIndexableFileContentError(
            reason,
            metrics=metrics,
            filename=filename,
            content_type=content_type,
        )

    return metrics


def validate_file_chunk_result(
    *,
    source: Any,
    chunk_count: Any,
    metrics: ExtractionMetrics,
    metadata: Mapping[str, Any] | None,
) -> None:
    """Defensive second gate for a parser output that yielded zero chunks."""
    if not is_file_backed_source(source) or int(chunk_count or 0) > 0:
        return
    meta = metadata if isinstance(metadata, Mapping) else {}
    raise NonIndexableFileContentError(
        "insufficient_text",
        metrics=metrics,
        filename=_metadata_value(meta, "filename", "fileName", "name"),
        content_type=_metadata_value(
            meta, "contentType", "content_type", "mimeType", "mimetype"
        ),
    )
