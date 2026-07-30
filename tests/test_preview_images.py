"""Website-RAG önizleme görseli (og:image) query-time enrichment birim testleri.

Kapsam: `FetcherStore._pick_preview_image` (saf seçim mantığı) ve
`FetcherStore.preview_images_by_urls` (crawl_results batch lookup, mongomock ile).
FetcherStore.__init__ gerçek Mongo'ya bağlanıp ping attığından `__new__` ile
bypass edilir ve `crawl_results` doğrudan mongomock koleksiyonuna bağlanır.
"""
import mongomock

from services.fetcher_store import FetcherStore


# ── _pick_preview_image: og:image önceliği + yedekler ────────────────────────

def test_pick_prefers_og_image():
    extracted = {"og": {"og:image": "https://cdn.example.com/a.jpg", "og:title": "T"}}
    assert FetcherStore._pick_preview_image(extracted) == "https://cdn.example.com/a.jpg"


def test_pick_falls_back_to_secure_url_variant():
    extracted = {"og": {"og:image:secure_url": "https://cdn.example.com/secure.jpg"}}
    assert (
        FetcherStore._pick_preview_image(extracted)
        == "https://cdn.example.com/secure.jpg"
    )


def test_pick_falls_back_to_first_img_when_no_og():
    extracted = {"og": {}, "images": [{"src": "https://example.com/first.png", "alt": "x"}]}
    assert FetcherStore._pick_preview_image(extracted) == "https://example.com/first.png"


def test_pick_skips_blank_og_and_uses_gallery():
    extracted = {"og": {"og:image": "   "}, "images": [{"src": "https://example.com/g.png"}]}
    assert FetcherStore._pick_preview_image(extracted) == "https://example.com/g.png"


def test_pick_returns_none_when_no_image():
    assert FetcherStore._pick_preview_image({"og": {}, "images": []}) is None
    assert FetcherStore._pick_preview_image({}) is None


# ── preview_images_by_urls: crawl_results batch lookup ───────────────────────

def _store_with(rows):
    store = FetcherStore.__new__(FetcherStore)  # __init__ (gerçek Mongo) bypass
    col = mongomock.MongoClient().db.crawl_results
    if rows:
        col.insert_many(rows)
    store.crawl_results = col
    return store


def test_preview_images_by_urls_maps_url_to_image():
    store = _store_with([
        {"url": "https://site.com/a", "extracted": {"og": {"og:image": "https://cdn/a.jpg"}}},
        {"url": "https://site.com/b", "extracted": {"images": [{"src": "https://cdn/b.png"}]}},
    ])
    result = store.preview_images_by_urls(["https://site.com/a", "https://site.com/b"])
    assert result == {
        "https://site.com/a": "https://cdn/a.jpg",
        "https://site.com/b": "https://cdn/b.png",
    }


def test_preview_images_by_urls_omits_pages_without_image():
    store = _store_with([
        {"url": "https://site.com/a", "extracted": {"og": {"og:image": "https://cdn/a.jpg"}}},
        {"url": "https://site.com/noimg", "extracted": {"og": {}, "images": []}},
    ])
    result = store.preview_images_by_urls(["https://site.com/a", "https://site.com/noimg"])
    assert result == {"https://site.com/a": "https://cdn/a.jpg"}
    assert "https://site.com/noimg" not in result


def test_preview_images_by_urls_empty_input_returns_empty():
    store = _store_with([])
    assert store.preview_images_by_urls([]) == {}
    assert store.preview_images_by_urls(["   "]) == {}
