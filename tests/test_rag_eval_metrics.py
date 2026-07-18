"""
RAG v2 Faz 1 — eval metriklerinin birim testleri.

`scripts/rag_eval.py` bir paket değil (eval araçları prod image'ına girmesin diye
`services/` altına konmadı), bu yüzden modül dosya yolundan yüklenir. Metrikler
saf fonksiyon: mongomock/mock gerekmez, ağ yok.
"""
import importlib.util
import math
import sys
from pathlib import Path

import pytest


def _load_rag_eval():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rag_eval.py"
    spec = importlib.util.spec_from_file_location("rag_eval", path)
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` çözümleme sırasında sys.modules[cls.__module__]'e bakar →
    # exec_module'dan ÖNCE kaydetmezsek AttributeError verir.
    sys.modules["rag_eval"] = module
    spec.loader.exec_module(module)
    return module


rag_eval = _load_rag_eval()


# ---------------------------------------------------------------------------
# normalize_url — etiket ile crawl metadata'sı arasındaki gürültüyü eler
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://example.com/a", "example.com/a"),
        ("http://example.com/a", "example.com/a"),          # şema önemsiz
        ("https://www.example.com/a", "example.com/a"),      # www önemsiz
        ("https://example.com/a/", "example.com/a"),         # sondaki / önemsiz
        ("https://EXAMPLE.com/a", "example.com/a"),          # host case-insensitive
        ("example.com/a", "example.com/a"),                  # şemasız
        ("https://example.com/a#bolum", "example.com/a"),    # fragment atılır
        ("https://example.com/a?x=1", "example.com/a?x=1"),  # query KORUNUR
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_url(raw, expected):
    assert rag_eval.normalize_url(raw) == expected


def test_normalize_url_distinguishes_different_paths():
    assert rag_eval.normalize_url("https://example.com/a") != rag_eval.normalize_url(
        "https://example.com/b"
    )


# ---------------------------------------------------------------------------
# recall@k
# ---------------------------------------------------------------------------
def test_recall_perfect():
    targets = {"doc:a", "doc:b"}
    hits = [{"doc:a"}, {"doc:b"}]
    assert rag_eval.recall_at_k(hits, targets, 10) == 1.0


def test_recall_partial():
    targets = {"doc:a", "doc:b"}
    hits = [{"doc:a"}, set()]
    assert rag_eval.recall_at_k(hits, targets, 10) == 0.5


def test_recall_zero_when_nothing_relevant_returned():
    assert rag_eval.recall_at_k([set(), set()], {"doc:a"}, 10) == 0.0


def test_recall_respects_k_cutoff():
    targets = {"doc:a", "doc:b"}
    hits = [set(), set(), {"doc:a"}, {"doc:b"}]
    # k=2 → ilgili sonuçlar kesimin altında kalır
    assert rag_eval.recall_at_k(hits, targets, 2) == 0.0
    assert rag_eval.recall_at_k(hits, targets, 4) == 1.0


def test_recall_does_not_double_count_same_target():
    """Aynı hedefi iki sonuç vurursa recall 1.0'ı AŞMAMALI."""
    targets = {"doc:a"}
    hits = [{"doc:a"}, {"doc:a"}]
    assert rag_eval.recall_at_k(hits, targets, 10) == 1.0


def test_recall_nan_when_no_targets():
    assert math.isnan(rag_eval.recall_at_k([{"doc:a"}], set(), 10))


# ---------------------------------------------------------------------------
# reciprocal rank / MRR
# ---------------------------------------------------------------------------
def test_rr_first_position():
    assert rag_eval.reciprocal_rank([{"doc:a"}, set()]) == 1.0


def test_rr_second_position():
    assert rag_eval.reciprocal_rank([set(), {"doc:a"}]) == 0.5


def test_rr_third_position():
    assert rag_eval.reciprocal_rank([set(), set(), {"doc:a"}]) == pytest.approx(1 / 3)


def test_rr_zero_when_no_hit():
    assert rag_eval.reciprocal_rank([set(), set()]) == 0.0


def test_rr_empty_results():
    assert rag_eval.reciprocal_rank([]) == 0.0


# ---------------------------------------------------------------------------
# nDCG@k
# ---------------------------------------------------------------------------
def test_ndcg_perfect_ranking_is_one():
    targets = {"doc:a", "doc:b"}
    hits = [{"doc:a"}, {"doc:b"}, set()]
    assert rag_eval.ndcg_at_k(hits, targets, 10) == pytest.approx(1.0)


def test_ndcg_zero_when_no_relevant_retrieved():
    assert rag_eval.ndcg_at_k([set(), set()], {"doc:a"}, 10) == 0.0


def test_ndcg_penalizes_lower_rank():
    targets = {"doc:a"}
    top = rag_eval.ndcg_at_k([{"doc:a"}, set(), set()], targets, 10)
    low = rag_eval.ndcg_at_k([set(), set(), {"doc:a"}], targets, 10)
    assert top == pytest.approx(1.0)
    assert 0.0 < low < top


def test_ndcg_known_value_single_target_at_rank_two():
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1 → nDCG = 1/log2(3)
    value = rag_eval.ndcg_at_k([set(), {"doc:a"}], {"doc:a"}, 10)
    assert value == pytest.approx(1.0 / math.log2(3))


def test_ndcg_respects_k_cutoff():
    targets = {"doc:a"}
    assert rag_eval.ndcg_at_k([set(), set(), {"doc:a"}], targets, 2) == 0.0


def test_ndcg_nan_when_no_targets():
    assert math.isnan(rag_eval.ndcg_at_k([{"doc:a"}], set(), 10))


# ---------------------------------------------------------------------------
# percentile — latency raporu buna dayanıyor
# ---------------------------------------------------------------------------
def test_percentile_single_value():
    assert rag_eval.percentile([42.0], 95) == 42.0


def test_percentile_median():
    assert rag_eval.percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_percentile_interpolates():
    # numpy varsayılanı (linear): p50 of [1,2,3,4] = 2.5
    assert rag_eval.percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)


def test_percentile_bounds():
    xs = [5.0, 1.0, 3.0]
    assert rag_eval.percentile(xs, 0) == 1.0
    assert rag_eval.percentile(xs, 100) == 5.0


def test_percentile_unsorted_input():
    assert rag_eval.percentile([3.0, 1.0, 2.0], 50) == 2.0


def test_percentile_empty_is_nan():
    assert math.isnan(rag_eval.percentile([], 95))


# ---------------------------------------------------------------------------
# result_hits — sonucu etiketli hedeflerle eşleştirme
# ---------------------------------------------------------------------------
def test_result_hits_matches_doc_id():
    row = {"doc_id": "d1", "metadata": {}}
    assert rag_eval.result_hits(row, {"doc:d1"}) == {"doc:d1"}


def test_result_hits_matches_url_after_normalization():
    row = {"doc_id": "d1", "metadata": {"url": "https://www.example.com/a/"}}
    assert rag_eval.result_hits(row, {"url:example.com/a"}) == {"url:example.com/a"}


def test_result_hits_empty_when_no_match():
    row = {"doc_id": "other", "metadata": {"url": "https://example.com/z"}}
    assert rag_eval.result_hits(row, {"doc:d1", "url:example.com/a"}) == set()


def test_result_hits_handles_missing_metadata():
    assert rag_eval.result_hits({"doc_id": "d1"}, {"doc:d1"}) == {"doc:d1"}


# ---------------------------------------------------------------------------
# EvalQuery.targets
# ---------------------------------------------------------------------------
def test_targets_combines_and_normalizes():
    query = rag_eval.EvalQuery(
        query="q",
        company_id="c1",
        relevant_doc_ids=("d1",),
        relevant_urls=("https://www.example.com/a/",),
        query_type="exact",
        content_type="schema",
    )
    assert query.targets == {"doc:d1", "url:example.com/a"}


# ---------------------------------------------------------------------------
# check_rerank_integrity — sessiz reranker fallback'ini yakalar
# ---------------------------------------------------------------------------
def _outcome(config, rerank_applied):
    query = rag_eval.EvalQuery("q", "c", ("d1",), (), "exact", "schema")
    return rag_eval.Outcome(
        config=config,
        query=query,
        recall=1.0,
        ndcg=1.0,
        rr=1.0,
        latency_ms=1.0,
        retrieval="hybrid",
        rerank_applied=rerank_applied,
    )


def test_integrity_flags_silent_rerank_fallback():
    problems = rag_eval.check_rerank_integrity(
        [_outcome("hybrid+rerank", False), _outcome("hybrid+rerank", False)]
    )
    assert problems
    assert "rerank_applied=false" in problems[0]


def test_integrity_clean_when_rerank_ran():
    assert rag_eval.check_rerank_integrity([_outcome("hybrid+rerank", True)]) == []


def test_integrity_ignores_non_rerank_configs():
    assert rag_eval.check_rerank_integrity([_outcome("hybrid", False)]) == []
