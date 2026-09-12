"""
FAZ 4 — yapı-farkında chunking (chunk_markdown_structure).
"""
from services.chunker import (
    _iter_markdown_sections,
    chunk_markdown_structure,
    chunk_text,
    normalize_text,
)


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
    assert chunks[0].heading_path == ("Ürünler",)
    assert chunks[0].context_header.startswith("ACME Mağaza — Ürünler")


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


def test_reconstruction_uses_body_instead_of_synthetic_context_header(app_with_mocks):
    import app

    reconstructed = app._reconstruct_from_chunks([
        {"text": "Title — Section\n\nAlpha", "context_prefix_chars": len("Title — Section\n\n"), "char_start": 0, "char_end": 5, "chunk_index": 0},
        {"text": "Title — Section\n\nBeta", "context_prefix_chars": len("Title — Section\n\n"), "char_start": 6, "char_end": 10, "chunk_index": 1},
    ])
    assert reconstructed == "Alpha Beta"


def test_chunking_removes_residual_html_and_non_content_tags():
    source = (
        '<div class="content"><p>Merhaba <strong>dünya</strong></p>'
        '<script>window.secret = true</script><p>İkinci paragraf</p></div>'
    )

    chunks = chunk_text(source, chunk_size=1200, min_chars=1)

    assert len(chunks) == 1
    assert "Merhaba dünya" in chunks[0].text
    assert "İkinci paragraf" in chunks[0].text
    assert "<div" not in chunks[0].text
    assert "<strong" not in chunks[0].text
    assert "window.secret" not in chunks[0].text


def test_chunking_removes_escaped_html_fragments_and_decodes_entities():
    source = "&lt;p&gt;Fiyat&nbsp;&amp;&nbsp;stok&lt;/p&gt;"

    chunks = chunk_text(source, chunk_size=1200, min_chars=1)

    assert [chunk.text for chunk in chunks] == ["Fiyat & stok"]


def test_plain_angle_bracket_comparison_is_not_treated_as_html():
    source = "Koşul: 5 < 10 ve 10 > 5."

    assert normalize_text(source) == source


def test_markdown_structure_survives_residual_inline_html():
    source = "# Başlık\n\nMetin <span>önemli</span> içerik."

    chunks = chunk_markdown_structure(source, chunk_size=1200, min_chars=1)

    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Başlık",)
    assert "Metin önemli içerik." in chunks[0].text
    assert "<span>" not in chunks[0].text


def test_html_heading_becomes_markdown_heading_before_strategy_detection():
    source = "<h2>Ürün Bilgisi</h2><p>Dayanıklı ve hafif.</p>"

    clean = normalize_text(source)
    chunks = chunk_markdown_structure(clean, chunk_size=1200, min_chars=1)

    assert clean.startswith("## Ürün Bilgisi")
    assert chunks[0].heading_path == ("Ürün Bilgisi",)


def test_custom_paired_html_tags_are_removed():
    source = "Önce <product-card>Ürün açıklaması</product-card> sonra"

    assert normalize_text(source) == "Önce Ürün açıklaması sonra"
