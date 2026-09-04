from services.dynamic_chunking import recommend_chunking, resolve_dynamic_chunking


def test_recommendation_grows_with_document_length():
    short = recommend_chunking(2_000)
    long = recommend_chunking(80_000)
    assert short["chunkSize"] == 450
    assert long["chunkSize"] == 1200
    assert long["chunkOverlap"] > short["chunkOverlap"]


def test_policy_changes_precision_and_context_window():
    precise = recommend_chunking(20_000, "precise")
    broad = recommend_chunking(20_000, "broad_context")
    assert precise["chunkSize"] < broad["chunkSize"]
    assert precise["chunkOverlap"] < broad["chunkOverlap"]


def test_auto_overrides_stale_fixed_values_but_preserves_other_options():
    resolved = resolve_dynamic_chunking({
        "chunkMode": "auto",
        "chunkPolicy": "balanced",
        "chunkSize": 999,
        "chunkOverlap": 1,
        "minChars": 40,
        "chunkStrategy": "auto",
    }, 2_000)
    assert resolved["chunkSize"] == 450
    assert resolved["chunkOverlap"] == 60
    assert resolved["minChars"] == 40
    assert resolved["chunkStrategy"] == "auto"
    assert resolved["chunkCharacterCount"] == 2_000


def test_manual_and_legacy_options_are_unchanged():
    manual = {"chunkMode": "manual", "chunkSize": 800, "chunkOverlap": 120}
    assert resolve_dynamic_chunking(manual, 100_000) == manual
