"""
Utility helpers for chunking plain text into overlapping windows prior to embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    Represents a single chunk of text together with its positional metadata.
    """

    text: str
    index: int
    char_start: int
    char_end: int


def normalize_text(text: str) -> str:
    """
    Lightweight normalisation applied before chunking.
    Currently only strips leading/trailing whitespace and collapses CRLF.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chars: int = 40,
) -> List[Chunk]:
    """
    Split text into overlapping windows.

    Parameters
    ----------
    text:
        The input text to chunk.
    chunk_size:
        Target size for each chunk (in characters).
    overlap:
        Number of trailing characters to overlap between consecutive chunks.
    min_chars:
        Minimum character count required for a chunk to be kept.

    Returns
    -------
    List[Chunk]
        Ordered list of chunk metadata objects.
    """
    clean = normalize_text(text)
    if not clean:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    if step <= 0:
        step = 1

    chunks: List[Chunk] = []
    idx = 0
    for start in range(0, len(clean), step):
        end = min(len(clean), start + chunk_size)
        window = clean[start:end]
        if len(window) < min_chars:
            continue
        chunks.append(Chunk(text=window, index=idx, char_start=start, char_end=end))
        idx += 1
        if end == len(clean):
            break
    return chunks


def iter_chunk_text(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chars: int = 40,
) -> Iterable[Chunk]:
    """
    Generator variant of `chunk_text` that yields chunks lazily.
    """
    for chunk in chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chars=min_chars,
    ):
        yield chunk
