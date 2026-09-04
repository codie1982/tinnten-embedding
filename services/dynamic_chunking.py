"""Deterministic per-document chunk sizing used by crawl and reindex flows."""
from __future__ import annotations

from typing import Any, Dict, Mapping


MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 1500
MIN_GAP = 100
POLICIES = {"balanced", "precise", "broad_context"}


def _round_ten(value: float) -> int:
    return int(round(value / 10.0) * 10)


def recommend_chunking(character_count: int = 0, policy: str = "balanced") -> Dict[str, Any]:
    """Return the same length bands exposed by the Space reindex API."""
    chars = max(0, int(character_count or 0))
    selected_policy = policy if policy in POLICIES else "balanced"
    if chars <= 4_000:
        size = 450
    elif chars <= 12_000:
        size = 700
    elif chars <= 40_000:
        size = 950
    elif chars <= 100_000:
        size = 1200
    else:
        size = 1500

    if selected_policy == "precise":
        size = max(MIN_CHUNK_SIZE, _round_ten(size * 0.8))
    elif selected_policy == "broad_context":
        size = min(MAX_CHUNK_SIZE, _round_ten(size * 1.2))
    ratio = 0.16 if selected_policy == "broad_context" else 0.10 if selected_policy == "precise" else 0.13
    overlap = min(size - MIN_GAP, max(0, _round_ten(size * ratio)))
    return {
        "mode": "auto",
        "policy": selected_policy,
        "characterCount": chars,
        "chunkSize": size,
        "chunkOverlap": overlap,
    }


def resolve_dynamic_chunking(options: Mapping[str, Any] | None, character_count: int) -> Dict[str, Any]:
    """Resolve an auto policy while leaving manual/legacy settings unchanged."""
    resolved = dict(options or {})
    mode = str(resolved.get("chunkMode") or resolved.get("mode") or "manual").strip().lower()
    if mode != "auto":
        return resolved
    policy = str(resolved.get("chunkPolicy") or resolved.get("policy") or "balanced").strip().lower()
    decision = recommend_chunking(character_count, policy)
    resolved.update({
        "chunkMode": "auto",
        "chunkPolicy": decision["policy"],
        "chunkCharacterCount": decision["characterCount"],
        "chunkSize": decision["chunkSize"],
        "chunkOverlap": decision["chunkOverlap"],
    })
    resolved.pop("mode", None)
    resolved.pop("policy", None)
    return resolved
