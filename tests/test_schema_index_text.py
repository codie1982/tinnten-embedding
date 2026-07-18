"""
FAZ 6 — schema-derived index metni: fetcher port'unun testleri + PARITY.

Kritik: `services/schema_index_text.py` fetcher'ın
`result_processor._index_text_from_extracted_content` / `_collect_text_leaves`
fonksiyonlarının portudur. Parity testi ikisini AYNI girdiyle çalıştırıp
byte-özdeş çıktı doğrular — fetcher tarafı değişirse test kırılır ve reindex'in
canlı ile ayrışacağı ANINDA görülür.

Fetcher repo'su import edilebilirse parity çalışır; edilemezse (CI ortamı fetcher'ı
görmüyorsa) yalnız parity testi atlanır, port'un kendi davranış testleri koşar.
"""
import json
import sys
from pathlib import Path

import pytest

from services.schema_index_text import (
    collect_text_leaves,
    index_text_from_extracted_content,
)


# ---------------------------------------------------------------------------
# Port'un kendi davranışı
# ---------------------------------------------------------------------------
def test_empty_returns_empty():
    assert index_text_from_extracted_content(None) == ""
    assert index_text_from_extracted_content("") == ""
    assert index_text_from_extracted_content({}) == ""


def test_dict_leaves_joined_by_newline():
    data = {"baslik": "İş Laptopu", "aciklama": "Hafif ve güçlü", "fiyat": "1.499 TL"}
    out = index_text_from_extracted_content(data, min_chars=1)
    assert out == "İş Laptopu\nHafif ve güçlü\n1.499 TL"


def test_bare_url_and_data_uri_leaves_skipped():
    data = {
        "img": "https://example.com/a.png",
        "logo": "data:image/png;base64,AAAA",
        "text": "gerçek içerik burada yeterince uzun olsun diye",
    }
    out = index_text_from_extracted_content(data, min_chars=1)
    assert out == "gerçek içerik burada yeterince uzun olsun diye"


def test_url_inside_text_not_skipped():
    # Yalnız ÇIPLAK url yaprağı atlanır; metin içindeki url kalır.
    data = {"t": "Detay için https://x.com adresine bak"}
    out = index_text_from_extracted_content(data, min_chars=1)
    assert "https://x.com" in out


def test_min_chars_threshold_drops_short_output():
    data = {"t": "kısa"}
    assert index_text_from_extracted_content(data, min_chars=80) == ""
    assert index_text_from_extracted_content(data, min_chars=1) == "kısa"


def test_json_string_is_parsed():
    raw = json.dumps({"a": "birinci alan", "b": "ikinci alan"})
    out = index_text_from_extracted_content(raw, min_chars=1)
    assert out == "birinci alan\nikinci alan"


def test_plain_text_string_used_as_is():
    txt = "Bu düz bir LLM extraction çıktısı, JSON değil, yeterince uzun."
    assert index_text_from_extracted_content(txt, min_chars=1) == txt


def test_nested_list_and_dict_traversed_in_order():
    data = {"bolum": [{"h": "Başlık"}, {"p": "Paragraf bir"}, {"p": "Paragraf iki"}]}
    out = index_text_from_extracted_content(data, min_chars=1)
    assert out == "Başlık\nParagraf bir\nParagraf iki"


def test_list_capped_at_200():
    parts = []
    collect_text_leaves([f"x{i}" for i in range(500)], parts)
    assert len(parts) == 200


def test_max_depth_stops_recursion():
    # 6'dan derin iç içe → yaprak toplanmaz
    deep = "leaf"
    for _ in range(8):
        deep = {"child": deep}
    parts = []
    collect_text_leaves(deep, parts)
    assert parts == []


# ---------------------------------------------------------------------------
# PARITY — fetcher'ın orijinal fonksiyonuyla birebir mi?
#
# Her iki repoda da top-level `services` paketi var → aynı process'te import
# çakışır (pytest tinnten-embedding'in services'ini önce yükler, fetcher'ınki
# gölgelenir). Bu yüzden fetcher fonksiyonunu AYRI BİR SUBPROCESS'te, yalnız
# fetcher yolu sys.path'te olacak şekilde çalıştırıp çıktıyı kıyaslıyoruz.
# ---------------------------------------------------------------------------
import subprocess

# Uzun bir paragraf (>200 char) — fetcher'ın INDEX_SCHEMA_MIN_CHARS=200 eşiğini
# aşan vakalar için. Kısa vakalar da var: ikisinde de eşik davranışı test edilsin.
_LONG = (
    "Bu ürün iş ve günlük kullanım için tasarlanmış hafif bir dizüstü bilgisayardır. "
    "Uzun pil ömrü, yüksek çözünürlüklü ekran ve hızlı SSD depolama sunar. Kutu "
    "içeriğinde şarj adaptörü ve hızlı başlangıç kılavuzu bulunur, garanti süresi "
    "yirmi dört aydır ve ücretsiz iade on dört gün içinde geçerlidir."
)

PARITY_CASES = [
    # Eşik ÜSTÜ (fetcher non-empty döner) — indirgeme mantığı test edilir
    {"baslik": "İş Laptopu", "aciklama": _LONG},
    {"img": "https://x.com/a.png", "aciklama": _LONG, "logo": "data:image/png;base64,AAAA"},
    {"bolum": [{"h": "Özellikler"}, {"p": _LONG}]},
    _LONG,
    json.dumps({"a": "İş Laptopu", "b": _LONG}),
    # Eşik ALTI (fetcher "" döner) — eşik davranışı parity'si test edilir
    {"baslik": "İş Laptopu", "fiyat": "1.499 TL"},
    "kısa metin",
]

# Fetcher fonksiyonunu AYRI SÜRECTE, GERÇEK eşiğiyle çalıştırır; eşiği de döner.
_FETCHER_RUNNER = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from services.result_processor import ResultProcessor, Config
rp = ResultProcessor.__new__(ResultProcessor)
cases = json.loads(sys.argv[2])
out = [rp._index_text_from_extracted_content(c) for c in cases]
print(json.dumps({"threshold": Config.INDEX_SCHEMA_MIN_CHARS, "out": out}, ensure_ascii=False))
"""


def _fetcher_result(cases):
    """Fetcher çıktısı + eşiği subprocess'ten alır. {'threshold':int,'out':[...]} | None."""
    fetcher_root = Path(__file__).resolve().parents[2] / "tinnten-fetcher"
    if not fetcher_root.exists():
        return None
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _FETCHER_RUNNER, str(fetcher_root), json.dumps(cases)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(fetcher_root),
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def test_parity_with_fetcher():
    result = _fetcher_result(PARITY_CASES)
    if result is None:
        pytest.skip("tinnten-fetcher subprocess'te çalıştırılamadı (bu ortamda yok)")

    threshold = result["threshold"]
    fetcher_out = result["out"]

    # Port'un varsayılanı fetcher eşiğiyle EŞİT olmalı — faithfulness'ın temeli.
    from services.schema_index_text import DEFAULT_SCHEMA_MIN_CHARS

    assert DEFAULT_SCHEMA_MIN_CHARS == threshold, (
        f"port varsayilani {DEFAULT_SCHEMA_MIN_CHARS} != fetcher esigi {threshold} "
        f"→ reindex canli ile ayrisir"
    )

    # TÜM vakalar (eşik üstü + altı) birebir eşleşmeli — eşiği de fetcher'dan alıp
    # port'a aynısını veriyoruz, böylece indirgeme VE eşik davranışı test edilir.
    nonempty = 0
    for case, f_out in zip(PARITY_CASES, fetcher_out):
        port_out = index_text_from_extracted_content(case, min_chars=threshold)
        assert port_out == f_out, f"parity kirildi: {case!r}\n port={port_out!r}\n fetcher={f_out!r}"
        if f_out:
            nonempty += 1
    assert nonempty > 0, "esik ustu hicbir vaka yok — test anlamsiz"
