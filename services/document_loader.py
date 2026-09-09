"""
Utilities for downloading documents from S3 and extracting their textual contents.
"""
from __future__ import annotations

import io
import codecs
import os
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from botocore.exceptions import ClientError
from docx import Document
from pypdf import PdfReader

from init.aws import get_aws_config, get_s3_client


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
}

TEXTUAL_MIME_PREFIXES = ("text/", "application/json", "application/xml")
DOCX_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
XLSX_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
}
PPTX_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-powerpoint",
}


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer limit without allowing a bad env to disable it."""
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# These are deliberately a second line of defence. The API's file-intelligence
# worker validates the same documents before enqueueing, but legacy messages and
# retries can still make the embedding worker parse the original object itself.
MAX_FILE_SIZE_BYTES = _positive_int_env(
    "EMBEDDING_DOCUMENT_MAX_FILE_SIZE_BYTES", 100 * 1024 * 1024
)
MAX_EXTRACTED_TEXT_CHARS = _positive_int_env(
    "EMBEDDING_DOCUMENT_MAX_EXTRACTED_TEXT_CHARS", 500_000
)
OFFICE_MAX_FILE_SIZE_BYTES = _positive_int_env(
    "EMBEDDING_DOCUMENT_OFFICE_MAX_FILE_SIZE_BYTES", 50 * 1024 * 1024
)
OFFICE_MAX_ENTRIES = _positive_int_env(
    "EMBEDDING_DOCUMENT_OFFICE_MAX_ENTRIES", 2_000
)
OFFICE_MAX_TOTAL_UNCOMPRESSED_BYTES = _positive_int_env(
    "EMBEDDING_DOCUMENT_OFFICE_MAX_TOTAL_UNCOMPRESSED_BYTES", 64 * 1024 * 1024
)
OFFICE_MAX_ENTRY_SIZE_BYTES = _positive_int_env(
    "EMBEDDING_DOCUMENT_OFFICE_MAX_ENTRY_SIZE_BYTES", 16 * 1024 * 1024
)
PDF_MAX_PAGES = _positive_int_env("EMBEDDING_DOCUMENT_PDF_MAX_PAGES", 1_000)
PDF_MAX_DECOMPRESSED_STREAM_BYTES = _positive_int_env(
    "EMBEDDING_DOCUMENT_PDF_MAX_DECOMPRESSED_STREAM_BYTES", 16 * 1024 * 1024
)
DOCX_MAX_PARAGRAPHS = _positive_int_env(
    "EMBEDDING_DOCUMENT_DOCX_MAX_PARAGRAPHS", 100_000
)
XLSX_MAX_FILE_SIZE_BYTES = _positive_int_env(
    "EMBEDDING_DOCUMENT_XLSX_MAX_FILE_SIZE_BYTES", 25 * 1024 * 1024
)
XLSX_MAX_WORKSHEETS = _positive_int_env(
    "EMBEDDING_DOCUMENT_XLSX_MAX_WORKSHEETS", 100
)
XLSX_MAX_ROWS_PER_SHEET = _positive_int_env(
    "EMBEDDING_DOCUMENT_XLSX_MAX_ROWS_PER_SHEET", 10_000
)
XLSX_MAX_COLUMNS_PER_SHEET = _positive_int_env(
    "EMBEDDING_DOCUMENT_XLSX_MAX_COLUMNS_PER_SHEET", 512
)
XLSX_MAX_CELLS = _positive_int_env(
    "EMBEDDING_DOCUMENT_XLSX_MAX_CELLS", 500_000
)
PPTX_MAX_SLIDES = _positive_int_env(
    "EMBEDDING_DOCUMENT_PPTX_MAX_SLIDES", 500
)
PPTX_MAX_SHAPES = _positive_int_env(
    "EMBEDDING_DOCUMENT_PPTX_MAX_SHAPES", 50_000
)

_PPTX_SLIDE_ENTRY = re.compile(r"^ppt/slides/slide\d+\.xml$", re.IGNORECASE)


class _BoundedTextCollector:
    """Collect text without ever retaining more than the indexable text budget."""

    def __init__(self, limit: Optional[int] = None) -> None:
        self.limit = max(1, int(limit or MAX_EXTRACTED_TEXT_CHARS))
        self._parts: list[str] = []
        self._length = 0
        self.full = False

    def append(self, value: object) -> bool:
        text = str(value or "").strip()
        if not text or self.full:
            return not self.full

        separator = "\n" if self._parts else ""
        remaining = self.limit - self._length
        if remaining <= len(separator):
            self.full = True
            return False

        allowed = remaining - len(separator)
        fragment = text[:allowed]
        if separator:
            self._parts.append(separator)
            self._length += len(separator)
        if fragment:
            self._parts.append(fragment)
            self._length += len(fragment)
        self.full = len(fragment) < len(text) or self._length >= self.limit
        return not self.full

    def text(self) -> str:
        return "".join(self._parts).strip()


class _SimpleHTMLTextExtractor(HTMLParser):
    def __init__(self, *, max_chars: Optional[int] = None) -> None:
        super().__init__()
        self._collector = _BoundedTextCollector(max_chars)

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self._collector.append(data)

    @property
    def full(self) -> bool:
        return self._collector.full

    def text(self) -> str:
        return self._collector.text()


class DocumentDownloadError(RuntimeError):
    """Raised when the document cannot be downloaded from S3."""


class DocumentParseError(RuntimeError):
    """Raised when the downloaded document cannot be parsed."""


@dataclass(frozen=True, slots=True)
class DocumentContent:
    bucket: str
    key: str
    filename: str
    content_type: Optional[str]
    text: str


class DocumentLoader:
    """
    Fetches documents from S3 and extracts text depending on their type.
    """

    def __init__(self, *, bucket: Optional[str] = None) -> None:
        cfg = get_aws_config()
        self.bucket = bucket or cfg.bucket
        self.client = get_s3_client()

    def fetch_text(
        self,
        key: str,
        *,
        bucket: Optional[str] = None,
        local: bool = False,
        filename_hint: Optional[str] = None,
    ) -> DocumentContent:
        """
        Download a document from S3 (or read from local filesystem) and return its textual contents.
        """
        if local:
            return self._fetch_local(key)

        target_bucket = bucket or self.bucket
        try:
            response = self.client.get_object(Bucket=target_bucket, Key=key)
        except ClientError as exc:
            raise DocumentDownloadError(f"Failed to download {key!r} from S3: {exc}") from exc

        body = response.get("Body")
        if body is None:
            raise DocumentDownloadError(f"S3 object {key!r} has no response body")
        try:
            declared_length = _safe_nonnegative_int(response.get("ContentLength"))
            if declared_length is not None and declared_length > MAX_FILE_SIZE_BYTES:
                raise DocumentDownloadError(
                    f"Document exceeds the {MAX_FILE_SIZE_BYTES}-byte download limit"
                )
            data = _read_bounded(body, MAX_FILE_SIZE_BYTES)
        except DocumentDownloadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DocumentDownloadError(f"Failed to read {key!r} from S3: {exc}") from exc
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        content_type = response.get("ContentType")
        # Object keys are frequently UUIDs without an extension. Use the
        # upload record's original filename when supplied so parser selection
        # does not depend on an unreliable S3 Content-Type.
        filename = str(filename_hint or "").strip() or Path(key).name or key
        text = self._extract_text(data, filename, content_type)
        return DocumentContent(bucket=target_bucket, key=key, filename=filename, content_type=content_type, text=text)

    def _fetch_local(self, file_path: str) -> DocumentContent:
        """Read a document from local filesystem and extract text."""
        try:
            if Path(file_path).stat().st_size > MAX_FILE_SIZE_BYTES:
                raise DocumentDownloadError(
                    f"Document exceeds the {MAX_FILE_SIZE_BYTES}-byte download limit"
                )
            with open(file_path, "rb") as f:
                data = _read_bounded(f, MAX_FILE_SIZE_BYTES)
        except DocumentDownloadError:
            raise
        except OSError as exc:
            raise DocumentDownloadError(f"Failed to read local file {file_path!r}: {exc}") from exc

        filename = Path(file_path).name
        suffix = Path(filename).suffix.lower()
        content_type = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".html": "text/html", ".htm": "text/html",
            ".csv": "text/csv", ".txt": "text/plain", ".md": "text/markdown", ".json": "application/json",
        }.get(suffix)
        text = self._extract_text(data, filename, content_type)
        return DocumentContent(bucket="local", key=file_path, filename=filename, content_type=content_type, text=text)

    # ------------------------------------------------------------------
    # Text extraction helpers
    # ------------------------------------------------------------------
    def _extract_text(self, data: bytes, filename: str, content_type: Optional[str]) -> str:
        if not isinstance(data, bytes):
            raise DocumentParseError("Document payload must be bytes")
        if len(data) > MAX_FILE_SIZE_BYTES:
            raise DocumentParseError(
                f"Document exceeds the {MAX_FILE_SIZE_BYTES}-byte parse limit"
            )

        suffix = Path(filename).suffix.lower()
        if suffix and suffix not in SUPPORTED_EXTENSIONS:
            raise DocumentParseError(f"Unsupported file extension: {suffix}")

        if suffix in {
            ".txt",
            ".md",
            ".csv",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
        } or self._is_plain_text(content_type):
            return _decode_utf8_bounded(data)

        if suffix in {".html", ".htm"} or content_type in {"text/html", "application/xhtml+xml"}:
            return self._read_html(data)

        if suffix == ".pdf" or content_type == "application/pdf":
            return self._read_pdf(data)

        if suffix in {".docx"} or content_type in DOCX_MIME_TYPES:
            return self._read_docx(data)

        if suffix == ".xlsx" or content_type in XLSX_MIME_TYPES:
            return self._read_xlsx(data)

        if suffix == ".pptx" or content_type in PPTX_MIME_TYPES:
            return self._read_pptx(data)

        raise DocumentParseError("Unknown document format")

    @staticmethod
    def _is_plain_text(content_type: Optional[str]) -> bool:
        if not content_type:
            return False
        return content_type.startswith(TEXTUAL_MIME_PREFIXES)

    @staticmethod
    def _read_pdf(data: bytes) -> str:
        try:
            _configure_pypdf_decompression_limit()
            with io.BytesIO(data) as stream:
                reader = PdfReader(stream)
                page_count = len(reader.pages)
                if page_count > PDF_MAX_PAGES:
                    raise DocumentParseError(
                        f"PDF page limit exceeded ({page_count} > {PDF_MAX_PAGES})"
                    )

                collector = _BoundedTextCollector()
                for page in reader.pages:
                    collector.append(page.extract_text() or "")
                    if collector.full:
                        break
                return collector.text()
        except DocumentParseError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DocumentParseError(f"Failed to parse PDF: {exc}") from exc

    @staticmethod
    def _read_docx(data: bytes) -> str:
        try:
            _validate_office_archive(data, "DOCX")
            with io.BytesIO(data) as stream:
                document = Document(stream)
                paragraphs = document.paragraphs
                if len(paragraphs) > DOCX_MAX_PARAGRAPHS:
                    raise DocumentParseError(
                        "DOCX paragraph limit exceeded "
                        f"({len(paragraphs)} > {DOCX_MAX_PARAGRAPHS})"
                    )

                collector = _BoundedTextCollector()
                for paragraph in paragraphs:
                    collector.append(paragraph.text)
                    if collector.full:
                        break
                return collector.text()
        except DocumentParseError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DocumentParseError(f"Failed to parse DOCX: {exc}") from exc

    @staticmethod
    def _read_html(data: bytes) -> str:
        try:
            parser = _SimpleHTMLTextExtractor()
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            view = memoryview(data)
            for offset in range(0, len(view), 64 * 1024):
                parser.feed(decoder.decode(view[offset : offset + 64 * 1024], final=False))
                if parser.full:
                    break
            if not parser.full:
                parser.feed(decoder.decode(b"", final=True))
            parser.close()
            return parser.text()
        except Exception as exc:  # noqa: BLE001
            raise DocumentParseError(f"Failed to parse HTML: {exc}") from exc

    @staticmethod
    def _read_xlsx(data: bytes) -> str:
        try:
            from openpyxl import load_workbook  # type: ignore
        except ImportError as exc:
            raise DocumentParseError(
                "openpyxl package is required to parse .xlsx files."
            ) from exc

        workbook = None
        stream = None
        try:
            if len(data) > XLSX_MAX_FILE_SIZE_BYTES:
                raise DocumentParseError(
                    f"XLSX file size limit exceeded ({len(data)} > {XLSX_MAX_FILE_SIZE_BYTES})"
                )
            _validate_office_archive(data, "XLSX")
            stream = io.BytesIO(data)
            workbook = load_workbook(
                filename=stream,
                data_only=True,
                read_only=True,
                keep_links=False,
            )
            sheets = workbook.worksheets
            if len(sheets) > XLSX_MAX_WORKSHEETS:
                raise DocumentParseError(
                    f"XLSX worksheet limit exceeded ({len(sheets)} > {XLSX_MAX_WORKSHEETS})"
                )

            declared_cells = 0
            dimensions: list[tuple[object, int, int]] = []
            for sheet in sheets:
                row_count = max(0, int(getattr(sheet, "max_row", 0) or 0))
                column_count = max(0, int(getattr(sheet, "max_column", 0) or 0))
                if row_count > XLSX_MAX_ROWS_PER_SHEET:
                    raise DocumentParseError(
                        f"XLSX row limit exceeded in {sheet.title!r} "
                        f"({row_count} > {XLSX_MAX_ROWS_PER_SHEET})"
                    )
                if column_count > XLSX_MAX_COLUMNS_PER_SHEET:
                    raise DocumentParseError(
                        f"XLSX column limit exceeded in {sheet.title!r} "
                        f"({column_count} > {XLSX_MAX_COLUMNS_PER_SHEET})"
                    )
                declared_cells += row_count * column_count
                if declared_cells > XLSX_MAX_CELLS:
                    raise DocumentParseError(
                        f"XLSX cell limit exceeded ({declared_cells} > {XLSX_MAX_CELLS})"
                    )
                dimensions.append((sheet, row_count, column_count))

            collector = _BoundedTextCollector()
            for sheet, row_count, column_count in dimensions:
                collector.append(f"[sheet] {sheet.title}")
                if collector.full:
                    break
                if not row_count or not column_count:
                    continue
                for row in sheet.iter_rows(
                    min_row=1,
                    max_row=row_count,
                    min_col=1,
                    max_col=column_count,
                    values_only=True,
                ):
                    values = []
                    for cell in row:
                        value = "" if cell is None else str(cell).strip()
                        if value:
                            values.append(value)
                    if values:
                        collector.append(" | ".join(values))
                    if collector.full:
                        break
                if collector.full:
                    break
            return collector.text()
        except DocumentParseError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DocumentParseError(f"Failed to parse XLSX: {exc}") from exc
        finally:
            close = getattr(workbook, "close", None)
            if callable(close):
                close()
            if stream is not None:
                stream.close()

    @staticmethod
    def _read_pptx(data: bytes) -> str:
        try:
            from pptx import Presentation  # type: ignore
        except ImportError as exc:
            raise DocumentParseError(
                "python-pptx package is required to parse .pptx files."
            ) from exc

        try:
            names = _validate_office_archive(data, "PPTX")
            declared_slides = sum(1 for name in names if _PPTX_SLIDE_ENTRY.match(name))
            if declared_slides > PPTX_MAX_SLIDES:
                raise DocumentParseError(
                    f"PPTX slide limit exceeded ({declared_slides} > {PPTX_MAX_SLIDES})"
                )

            with io.BytesIO(data) as stream:
                presentation = Presentation(stream)
                slides = presentation.slides
                if len(slides) > PPTX_MAX_SLIDES:
                    raise DocumentParseError(
                        f"PPTX slide limit exceeded ({len(slides)} > {PPTX_MAX_SLIDES})"
                    )

                collector = _BoundedTextCollector()
                shape_count = 0
                for slide_idx, slide in enumerate(slides, start=1):
                    collector.append(f"[slide] {slide_idx}")
                    for shape in slide.shapes:
                        shape_count += 1
                        if shape_count > PPTX_MAX_SHAPES:
                            raise DocumentParseError(
                                f"PPTX shape limit exceeded ({shape_count} > {PPTX_MAX_SHAPES})"
                            )
                        text = getattr(shape, "text", None)
                        if isinstance(text, str) and text.strip():
                            collector.append(text)
                        if collector.full:
                            break
                    if collector.full:
                        break
                return collector.text()
        except DocumentParseError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DocumentParseError(f"Failed to parse PPTX: {exc}") from exc


def _safe_nonnegative_int(value: object) -> Optional[int]:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _read_bounded(stream: object, limit: int) -> bytes:
    read = getattr(stream, "read", None)
    if not callable(read):
        raise DocumentDownloadError("Document stream is not readable")
    data = read(limit + 1)
    if not isinstance(data, (bytes, bytearray)):
        raise DocumentDownloadError("Document stream did not return bytes")
    if len(data) > limit:
        raise DocumentDownloadError(f"Document exceeds the {limit}-byte download limit")
    return bytes(data)


def _decode_utf8_bounded(data: bytes) -> str:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    parts: list[str] = []
    length = 0
    view = memoryview(data)
    for offset in range(0, len(view), 64 * 1024):
        decoded = decoder.decode(view[offset : offset + 64 * 1024], final=False)
        remaining = MAX_EXTRACTED_TEXT_CHARS - length
        if remaining <= 0:
            break
        fragment = decoded[:remaining]
        parts.append(fragment)
        length += len(fragment)
        if len(fragment) < len(decoded):
            break
    else:
        tail = decoder.decode(b"", final=True)
        if tail and length < MAX_EXTRACTED_TEXT_CHARS:
            parts.append(tail[: MAX_EXTRACTED_TEXT_CHARS - length])
    return "".join(parts)


def _validate_office_archive(data: bytes, label: str) -> tuple[str, ...]:
    if len(data) > OFFICE_MAX_FILE_SIZE_BYTES:
        raise DocumentParseError(
            f"{label} file size limit exceeded ({len(data)} > {OFFICE_MAX_FILE_SIZE_BYTES})"
        )

    try:
        with io.BytesIO(data) as stream, zipfile.ZipFile(stream) as archive:
            entries = archive.infolist()
            if len(entries) > OFFICE_MAX_ENTRIES:
                raise DocumentParseError(
                    f"{label} archive entry limit exceeded ({len(entries)} > {OFFICE_MAX_ENTRIES})"
                )

            total_size = 0
            names: list[str] = []
            for entry in entries:
                names.append(entry.filename)
                if entry.is_dir():
                    continue
                if entry.flag_bits & 0x1:
                    raise DocumentParseError(f"Encrypted {label} archives are not supported")
                if entry.file_size > OFFICE_MAX_ENTRY_SIZE_BYTES:
                    raise DocumentParseError(
                        f"{label} archive entry size limit exceeded for {entry.filename!r}"
                    )
                total_size += entry.file_size
                if total_size > OFFICE_MAX_TOTAL_UNCOMPRESSED_BYTES:
                    raise DocumentParseError(
                        f"{label} archive uncompressed size limit exceeded"
                    )
            return tuple(names)
    except DocumentParseError:
        raise
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        raise DocumentParseError(f"Invalid {label} archive: {exc}") from exc


def _configure_pypdf_decompression_limit() -> None:
    """Clamp pypdf's process-wide zlib output ceiling when the version supports it."""
    try:
        from pypdf import filters  # type: ignore

        current = getattr(filters, "ZLIB_MAX_OUTPUT_LENGTH", None)
        if isinstance(current, int) and (current <= 0 or current > PDF_MAX_DECOMPRESSED_STREAM_BYTES):
            filters.ZLIB_MAX_OUTPUT_LENGTH = PDF_MAX_DECOMPRESSED_STREAM_BYTES
    except Exception:  # noqa: BLE001
        # Older pypdf releases do not expose this hardening hook. File and page
        # limits still apply, and parsing errors remain fail-closed.
        return
