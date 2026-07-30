"""Uygulama sabitleri.

Env değişkeni yerine KODDA yönetilen özellik bayrakları ve sabitler burada
toplanır — deploy'da ayrı env ayarı gerektirmeden tek yerden açılıp kapanır.
Env'e bağlı ayarlar (URI, timeout vb.) yerlerinde kalır; buraya yalnız koddan
sabitlenen değerler yazılır.
"""


class FeatureFlags:
    """Özellik bayrakları (kod-yönetimli)."""

    # Website-RAG kaynak kartlarında og:image önizlemesi.
    # True → arama yanıtındaki web sonuçlarına query-time og:image enrichment
    # uygulanır: sonuç URL'lerine göre `crawl_results.extracted.og`'dan og:image
    # batch çekilip sonuç metadata'sına eklenir. Mevcut indexli içerik YENİDEN
    # INDEXLENMEDEN görsel alır. Salt-okuma + best-effort — fetcher DB erişilemezse
    # arama görselsiz sürer, yani True bırakmak güvenlidir.
    RAG_PREVIEW_IMAGES_ENABLED = True
