"""
Utility helpers for chunking plain text into overlapping windows prior to embedding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


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


def _iter_markdown_sections(clean: str) -> List[Tuple[List[str], int, int]]:
    """
    Markdown'ı ATX heading'lerine göre bölümlere ayırır. Her bölüm heading
    satırıyla başlar; heading path (üst başlıklar) stack ile izlenir. Fenced
    code-block (``` / ~~~) içindeki '#'ler heading SAYILMAZ. İlk heading'den
    önceki metin (preamble) path=[] ile ilk bölüm olur.

    Dönüş: [(heading_path, char_start, char_end), ...] — orijinal metindeki span'lar.
    """
    lines = clean.split("\n")
    offsets: List[int] = []
    p = 0
    for ln in lines:
        offsets.append(p)
        p += len(ln) + 1  # +1: '\n'
    total = len(clean)

    stack: List[Tuple[int, str]] = []
    sections: List[Tuple[List[str], int, int]] = []
    cur_start = 0
    cur_path: List[str] = []
    in_fence = False
    for i, ln in enumerate(lines):
        if _FENCE_RE.match(ln):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(ln)
        if not m:
            continue
        end = offsets[i]
        if end > cur_start:
            sections.append((list(cur_path), cur_start, end))
        level = len(m.group(1))
        title = m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        cur_path = [t for (_lvl, t) in stack]
        cur_start = offsets[i]
    sections.append((list(cur_path), cur_start, total))
    return sections


def chunk_markdown_structure(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chars: int = 40,
    title: Optional[str] = None,
    url: Optional[str] = None,
) -> List[Chunk]:
    """
    Yapı-farkında chunking: markdown heading sınırlarına saygı gösterir, ardışık
    küçük bölümleri ≤chunk_size pencerelere paketler, chunk_size'ı aşan bölümü
    karakter penceresine (offset korunarak) böler. Her chunk'ın başına sentetik
    context header eklenir: "«title» — «heading > path» («url»)". `char_start/
    char_end` GÖVDE span'ıdır (header sentetik, offset'e dahil değil).
    """
    clean = normalize_text(text)
    if not clean:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    def header_for(path: List[str]) -> str:
        parts: List[str] = []
        t = str(title or "").strip()
        if t:
            parts.append(t)
        if path:
            parts.append(" > ".join(p for p in path if p))
        base = " — ".join(parts)
        u = str(url or "").strip()
        if u:
            base = f"{base} ({u})" if base else u
        base = base.strip()[:160]
        return (base + "\n\n") if base else ""

    chunks: List[Chunk] = []
    idx = 0

    def emit(path: List[str], s: int, e: int) -> None:
        nonlocal idx
        body = clean[s:e].strip()
        if len(body) < min_chars:
            return
        head = header_for(path)
        if len(body) <= chunk_size:
            chunks.append(Chunk(text=head + body, index=idx, char_start=s, char_end=e))
            idx += 1
            return
        # Oversize bölüm → karakter penceresi (offset kaydırmalı), her pencereye header.
        for sub in chunk_text(body, chunk_size=chunk_size, overlap=overlap, min_chars=min_chars):
            chunks.append(
                Chunk(
                    text=head + sub.text,
                    index=idx,
                    char_start=s + sub.char_start,
                    char_end=s + sub.char_end,
                )
            )
            idx += 1

    # Ardışık bölümleri (orijinal metinde bitişik) ≤chunk_size pencerelere paketle.
    pack_start: Optional[int] = None
    pack_end = 0
    pack_path: List[str] = []
    for path, s, e in _iter_markdown_sections(clean):
        if pack_start is None:
            pack_start, pack_end, pack_path = s, e, path
        elif (e - pack_start) <= chunk_size:
            pack_end = e  # bitişik → paketi genişlet
        else:
            emit(pack_path, pack_start, pack_end)
            pack_start, pack_end, pack_path = s, e, path
    if pack_start is not None:
        emit(pack_path, pack_start, pack_end)
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
