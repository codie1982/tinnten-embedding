"""
Utility helpers for chunking plain text into overlapping windows prior to embedding.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup, Comment

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HTML_TAG_RE = re.compile(
    r"</?(?:html|head|body|main|article|section|aside|div|p|br|hr|h[1-6]|"
    r"span|a|ul|ol|li|dl|dt|dd|table|thead|tbody|tfoot|tr|td|th|caption|"
    r"strong|em|b|i|u|small|mark|code|pre|blockquote|figure|figcaption|"
    r"script|style|noscript|nav|footer|header|iframe|template|svg)\b[^>]*>",
    re.IGNORECASE,
)
_PAIRED_HTML_TAG_RE = re.compile(
    r"<([A-Za-z][A-Za-z0-9:_-]*)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "caption", "dd", "div",
    "dl", "dt", "figcaption", "figure", "footer", "form", "h1", "h2",
    "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav",
    "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot", "th",
    "thead", "tr", "ul",
}
_NON_CONTENT_TAGS = {"script", "style", "noscript", "iframe", "template", "svg"}
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PARAGRAPH_BOUNDARY_RE = re.compile(r"\n[ \t]*\n+")
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?…]+(?:[\"'”’»\)\]]*)[ \t]*(?:\n+|[ \t]+)")
_LINE_BOUNDARY_RE = re.compile(r"\n+")
_WORD_BOUNDARY_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    Represents a single chunk of text together with its positional metadata.
    """

    text: str
    index: int
    char_start: int
    char_end: int
    heading_path: Tuple[str, ...] = ()
    context_header: str = ""


def normalize_text(text: str) -> str:
    """
    Normalise and sanitise text immediately before chunking.

    In addition to newline/whitespace normalisation, remove residual HTML tags,
    comments and non-content elements. This is intentionally done at the shared
    chunker boundary so inline text, fetcher records, uploads and legacy callers
    receive the same protection. Plain text comparisons such as ``5 < 10`` are
    left untouched because parsing is only enabled when a known HTML tag exists.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    clean = text.replace("\r\n", "\n").replace("\r", "\n")
    clean = _ZERO_WIDTH_RE.sub("", clean)
    clean = _CONTROL_RE.sub("", clean)

    # Decode entities first so escaped fragments such as ``&lt;div&gt;`` are
    # detected and removed as well. Two passes cover commonly double-escaped
    # crawler output without looping indefinitely on malformed input.
    for _ in range(2):
        decoded = html.unescape(clean)
        if decoded == clean:
            break
        clean = decoded

    clean = _HTML_COMMENT_RE.sub("", clean)
    if _HTML_TAG_RE.search(clean) or _PAIRED_HTML_TAG_RE.search(clean):
        soup = BeautifulSoup(clean, "html.parser")
        for node in soup.find_all(string=lambda value: isinstance(value, Comment)):
            node.extract()
        for tag in soup.find_all(_NON_CONTENT_TAGS):
            tag.decompose()

        # Preserve meaningful layout while dropping markup. Markdown headings
        # already present outside HTML fragments remain unchanged.
        for heading in soup.find_all(re.compile(r"^h[1-6]$", re.IGNORECASE)):
            level = int(str(heading.name)[1])
            title = heading.get_text(" ", strip=True)
            heading.replace_with(f"\n{'#' * level} {title}\n" if title else "\n")
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for tag in soup.find_all(_BLOCK_TAGS):
            tag.insert_before("\n")
            tag.insert_after("\n")
        clean = soup.get_text(separator="", strip=False)

    # Best-effort removal for malformed known tags BeautifulSoup could not
    # consume. Unknown angle-bracket text is retained to avoid damaging prose.
    clean = _HTML_TAG_RE.sub("", clean)
    clean = clean.replace("\xa0", " ")
    clean = re.sub(r"[ \t]+\n", "\n", clean)
    clean = re.sub(r"\n[ \t]+", "\n", clean)
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


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


def _last_boundary_before(text: str, pattern: re.Pattern[str], start: int, end: int) -> Optional[int]:
    """Return the final boundary end in ``[start, end]`` for one separator level."""
    candidate: Optional[int] = None
    for match in pattern.finditer(text, start, end):
        if match.end() <= end:
            candidate = match.end()
    return candidate


def _recursive_chunk_end(text: str, start: int, hard_end: int, chunk_size: int) -> int:
    """Choose a semantic boundary without exceeding the configured hard limit."""
    if hard_end >= len(text):
        return len(text)

    # Avoid producing pathologically small chunks merely because a paragraph
    # separator occurs near the beginning of the window. This is the same
    # separator hierarchy commonly used by recursive text splitters, extended
    # with an explicit sentence boundary before the word fallback.
    preferred_floor = start + max(1, int(chunk_size * 0.40))
    for pattern in (
        _PARAGRAPH_BOUNDARY_RE,
        _SENTENCE_BOUNDARY_RE,
        _LINE_BOUNDARY_RE,
        _WORD_BOUNDARY_RE,
    ):
        candidate = _last_boundary_before(text, pattern, preferred_floor, hard_end)
        if candidate is not None and candidate > start:
            return candidate

    # If the preferred-fill window has no separator, preserve a word whenever
    # possible. A single token longer than chunk_size is the only case where a
    # hard character split is unavoidable.
    candidate = _last_boundary_before(text, _WORD_BOUNDARY_RE, start, hard_end)
    return candidate if candidate is not None and candidate > start else hard_end


def _recursive_overlap_start(text: str, chunk_start: int, chunk_end: int, overlap: int) -> int:
    """Start the next window at the nearest sentence (or word) boundary."""
    if overlap <= 0:
        return chunk_end
    desired = max(chunk_start + 1, chunk_end - overlap)

    sentence_boundaries = list(_SENTENCE_BOUNDARY_RE.finditer(text, chunk_start, chunk_end))
    sentence_candidates = [
        match.end()
        for match in sentence_boundaries
        if chunk_start < match.end() < chunk_end
    ]
    if sentence_candidates:
        return min(sentence_candidates, key=lambda value: abs(value - desired))

    # A single complete sentence cannot be overlapped without either repeating
    # the whole window or starting mid-sentence. Prefer semantic integrity and
    # omit overlap for that edge case.
    if sentence_boundaries:
        return chunk_end

    word_match = _WORD_BOUNDARY_RE.search(text, desired, chunk_end)
    if word_match is not None and word_match.end() < chunk_end:
        return word_match.end()
    return chunk_end


def chunk_text_recursive(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chars: int = 40,
) -> List[Chunk]:
    """Split text at paragraph, sentence, line and word boundaries.

    The configured ``chunk_size`` remains a hard character limit. Paragraphs
    and sentences are preferred; words are preserved when a sentence itself is
    too long. Only a single token longer than the limit is split by character.
    Overlap also starts on a sentence boundary when one is available.
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

    chunks: List[Chunk] = []
    start = 0
    while start < len(clean):
        hard_end = min(len(clean), start + chunk_size)
        end = _recursive_chunk_end(clean, start, hard_end, chunk_size)
        raw = clean[start:end]
        left_trimmed = raw.lstrip()
        leading = len(raw) - len(left_trimmed)
        body = left_trimmed.rstrip()
        body_start = start + leading
        body_end = body_start + len(body)
        if len(body) >= min_chars:
            chunks.append(Chunk(
                text=body,
                index=len(chunks),
                char_start=body_start,
                char_end=body_end,
            ))
        if end >= len(clean):
            break
        next_start = _recursive_overlap_start(clean, start, end, overlap)
        if next_start <= start:
            next_start = end
        start = next_start
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
            chunks.append(Chunk(
                text=head + body,
                index=idx,
                char_start=s,
                char_end=e,
                heading_path=tuple(path),
                context_header=head,
            ))
            idx += 1
            return
        # Oversize bölüm → cümle/kelime sınırı duyarlı recursive pencereler.
        raw = clean[s:e]
        leading = len(raw) - len(raw.lstrip())
        body_start = s + leading
        for sub in chunk_text_recursive(body, chunk_size=chunk_size, overlap=overlap, min_chars=min_chars):
            chunks.append(
                Chunk(
                    text=head + sub.text,
                    index=idx,
                    char_start=body_start + sub.char_start,
                    char_end=body_start + sub.char_end,
                    heading_path=tuple(path),
                    context_header=head,
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
