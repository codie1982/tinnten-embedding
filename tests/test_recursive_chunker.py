import pytest

from services.chunker import chunk_text_recursive
from services.content_store import normalize_index_options


def test_recursive_chunker_prefers_sentence_boundaries():
    text = (
        "Birinci cümle yeterince uzundur ve kendi anlamını taşır. "
        "İkinci cümle de bağımsız bir düşünceyi eksiksiz biçimde anlatır. "
        "Üçüncü cümle son parçaya anlamlı bir bağlam ekler."
    )

    chunks = chunk_text_recursive(text, chunk_size=105, overlap=25, min_chars=1)

    assert len(chunks) >= 2
    assert all(len(chunk.text) <= 105 for chunk in chunks)
    assert all(chunk.text[-1] in ".!?…" for chunk in chunks[:-1])
    assert all(chunk.text == text[chunk.char_start:chunk.char_end] for chunk in chunks)


def test_recursive_chunker_preserves_words_until_a_token_exceeds_limit():
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima"

    chunks = chunk_text_recursive(text, chunk_size=24, overlap=5, min_chars=1)

    assert len(chunks) > 1
    assert all(not chunk.text.startswith(" ") and not chunk.text.endswith(" ") for chunk in chunks)
    assert all(chunk.text == text[chunk.char_start:chunk.char_end] for chunk in chunks)
    assert all(len(chunk.text) <= 24 for chunk in chunks)


def test_index_options_normalizes_and_validates_chunk_strategy():
    defaults = {"default_chunk_size": 900, "default_chunk_overlap": 120}
    resolved = normalize_index_options({"chunk_strategy": " Recursive "}, **defaults)
    assert resolved["chunkStrategy"] == "recursive"

    with pytest.raises(ValueError, match="chunkStrategy"):
        normalize_index_options({"chunkStrategy": "unknown"}, **defaults)
