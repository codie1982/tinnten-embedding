"""
FAZ 4 — yapı-farkında chunking (chunk_markdown_structure).
"""
from services.chunker import chunk_markdown_structure, _iter_markdown_sections


def test_sections_respect_heading_hierarchy():
    md = "Önsöz metni burada.\n# Bölüm A\nA içeriği\n## Alt A1\nA1 içeriği\n# Bölüm B\nB içeriği"
    secs = _iter_markdown_sections(md)
    paths = [p for (p, _s, _e) in secs]
    assert paths[0] == []  # preamble
    assert paths[1] == ["Bölüm A"]
    assert paths[2] == ["Bölüm A", "Alt A1"]
    assert paths[3] == ["Bölüm B"]


def test_fenced_code_hash_is_not_heading():
    md = "# Gerçek Başlık\nmetin\n```\n# bu bir yorum, başlık DEĞİL\n```\ndevam"
    secs = _iter_markdown_sections(md)
    paths = [p for (p, _s, _e) in secs]
    # Yalnız bir gerçek başlık; fenced içindeki # sayılmadı
    assert paths.count(["Gerçek Başlık"]) == 1
    assert ["bu bir yorum, başlık DEĞİL"] not in paths


def test_context_header_prepended_with_title_and_url():
    md = "# Ürünler\nAyakkabı listesi burada yeterince uzun içerik."
    chunks = chunk_markdown_structure(
        md, chunk_size=1200, min_chars=5, title="ACME Mağaza", url="https://acme.com/p"
    )
    assert len(chunks) == 1
    assert chunks[0].text.startswith("ACME Mağaza — Ürünler (https://acme.com/p)")
    assert "Ayakkabı listesi" in chunks[0].text


def test_char_offsets_reference_body_not_header():
    md = "# H\n" + ("x" * 50)
    chunks = chunk_markdown_structure(md, chunk_size=1200, min_chars=5, title="T")
    c = chunks[0]
    # char_start/char_end orijinal metindeki gövde span'ı; header sentetik
    assert md[c.char_start:c.char_end].strip().startswith("# H")
    assert c.text.startswith("T — H")


def test_oversize_section_falls_back_to_char_windows_with_offsets():
    body = "y" * 3000
    md = "# Büyük\n" + body
    chunks = chunk_markdown_structure(md, chunk_size=1000, overlap=100, min_chars=10, title="T")
    assert len(chunks) >= 3  # 3000 char → birden çok pencere
    for c in chunks:
        assert c.text.startswith("T — Büyük")  # her pencerede context header
        # offset'ler orijinal metne denk geliyor
        assert 0 <= c.char_start < c.char_end <= len(md)


def test_small_sections_are_packed():
    md = "# A\na\n# B\nb\n# C\nc"  # üç minik bölüm
    chunks = chunk_markdown_structure(md, chunk_size=1200, min_chars=1, title="T")
    # Hepsi tek pakete sığar → 1 chunk (bitişik span)
    assert len(chunks) == 1
    assert "# A" in chunks[0].text and "# C" in chunks[0].text


def test_empty_and_non_string():
    assert chunk_markdown_structure("") == []
    assert chunk_markdown_structure("   \n  ") == []


def test_no_headings_behaves_like_single_section():
    md = "Başlıksız düz metin, yeterince uzun içerik burada var."
    chunks = chunk_markdown_structure(md, chunk_size=1200, min_chars=5, url="https://x.com")
    assert len(chunks) == 1
    assert chunks[0].text.endswith(md)  # header yalnız url
    # title/path yokken header bare url (parantezsiz) — doğru davranış
    assert chunks[0].text.startswith("https://x.com")
