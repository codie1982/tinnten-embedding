#!/usr/bin/env python3
"""
Retrieval eval harness for tinnten-embedding (RAG v2 — Faz 1).

Why this exists: hybrid retrieval, the cross-encoder reranker and structure
chunking are all already implemented but disabled. Turning them on is cheap;
knowing whether they HELPED is not. This harness makes every such change
measurable, and — critically — provable.

What it does:
  1) Preflight via GET /api/v10/config: is hybrid enabled, is a reranker really
     loaded? A run must not silently measure the wrong thing.
  2) Runs a labeled query set against POST /api/v10/content/search under three
     configs — dense, hybrid, hybrid+rerank. All three are request-level flags,
     so one running server serves all of them (no restart, no redeploy).
  3) Asserts `retrieval_meta` proves the intended path actually executed. The
     service falls back to fusion order when the reranker is missing or throws,
     WITHOUT erroring — so "hybrid+rerank" can silently be plain hybrid. We
     check `rerank_applied` rather than trusting the request.
  4) Reports recall@k / nDCG@k / MRR, broken down by query_type x content_type.
     content_type matters: fetcher indexes schema-derived text (a flat newline
     join, no markdown headings) when schema extraction succeeds, so structural
     chunking cannot help those pages — the breakdown keeps that visible.
  5) Optional separate latency run (p50/p95) with warmup excluded.

Quality vs latency are measured in SEPARATE runs on purpose: a 50-100 query
quality set is too small for a trustworthy p95, and the first call pays model
load/cache costs that no user-facing request would.

Exit codes: 0 pass, 1 failed a threshold / capability mismatch, 2 bad input.

Usage examples:
  python scripts/rag_eval.py \
    --base-url http://localhost:5003 \
    --queries scripts/rag_eval_queries.json \
    --k 10

  # only latency, hybrid config, 30 timed runs after 3 warmups
  python scripts/rag_eval.py --latency-runs 30 --latency-config hybrid

  # gate a rollout: fail if hybrid does not beat dense on recall
  python scripts/rag_eval.py --fail-under-recall 0.6 --require-rerank
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import requests


SEARCH_PATH = "/api/v10/content/search"
CONFIG_PATH = "/api/v10/config"

# Request-level config matrix. `hybrid`/`rerank` are honoured per request by the
# service, so all three run against a single server instance.
CONFIGS: Dict[str, Dict[str, Any]] = {
    "dense": {"hybrid": False},
    "hybrid": {"hybrid": True, "rerank": False},
    "hybrid+rerank": {"hybrid": True, "rerank": True},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Pure metric helpers — no I/O, deterministic, unit-tested in
# tests/test_rag_eval_metrics.py. Keep them pure.
# ---------------------------------------------------------------------------
def normalize_url(value: Any) -> str:
    """Canonical URL key: lowercase host, no scheme/fragment, no trailing slash.

    Labels and crawled metadata disagree on trailing slashes and http/https far
    too often to compare raw strings.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    path = (parsed.path or "").rstrip("/")
    key = f"{host}{path}"
    if parsed.query:
        key = f"{key}?{parsed.query}"
    return key


def recall_at_k(hit_targets_in_order: Sequence[Set[str]], targets: Set[str], k: int) -> float:
    """Fraction of labeled targets covered by the top-k results.

    `hit_targets_in_order[i]` is the set of targets result i matches.
    """
    if not targets:
        return float("nan")
    covered: Set[str] = set()
    for hits in hit_targets_in_order[:k]:
        covered |= hits
    return len(covered & targets) / len(targets)


def reciprocal_rank(hit_targets_in_order: Sequence[Set[str]]) -> float:
    """1/rank of the first relevant result; 0.0 if none."""
    for idx, hits in enumerate(hit_targets_in_order, start=1):
        if hits:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(hit_targets_in_order: Sequence[Set[str]], targets: Set[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    IDCG assumes the ideal ranking puts min(k, |targets|) relevant docs first.
    NOTE: label each query ONE way (urls OR doc_ids). Naming the same document
    both ways makes it two targets that a single result can never both "ideally"
    occupy, which depresses nDCG. See RAG_EVAL_RUNBOOK.md.
    """
    if not targets:
        return float("nan")
    dcg = 0.0
    for idx, hits in enumerate(hit_targets_in_order[:k], start=1):
        if hits:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(k, len(targets))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else float("nan")


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (numpy default method)."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(xs[int(rank)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (rank - lo))


def mean(values: Sequence[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return statistics.fmean(clean) if clean else float("nan")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalQuery:
    query: str
    company_id: Optional[str]
    relevant_doc_ids: Tuple[str, ...]
    relevant_urls: Tuple[str, ...]
    query_type: str
    content_type: str

    @property
    def targets(self) -> Set[str]:
        """Labeled targets as canonical keys: `doc:<id>` and `url:<normalized>`."""
        out = {f"doc:{d}" for d in self.relevant_doc_ids if d}
        out |= {f"url:{normalize_url(u)}" for u in self.relevant_urls if normalize_url(u)}
        return out


@dataclass
class Outcome:
    config: str
    query: EvalQuery
    recall: float
    ndcg: float
    rr: float
    latency_ms: float
    retrieval: str = ""
    rerank_applied: bool = False
    error: str = ""


def result_hits(row: Dict[str, Any], targets: Set[str]) -> Set[str]:
    """Which labeled targets this result row satisfies."""
    keys: Set[str] = set()
    doc_id = row.get("doc_id")
    if doc_id:
        keys.add(f"doc:{doc_id}")
    url = (row.get("metadata") or {}).get("url")
    normalized = normalize_url(url)
    if normalized:
        keys.add(f"url:{normalized}")
    return keys & targets


def load_queries(path: Path) -> List[EvalQuery]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("query file must be a JSON array")
    out: List[EvalQuery] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"query[{i}] must be an object")
        text = str(item.get("query") or "").strip()
        if not text:
            raise ValueError(f"query[{i}] missing 'query'")
        doc_ids = tuple(str(x) for x in (item.get("relevant_doc_ids") or []))
        urls = tuple(str(x) for x in (item.get("relevant_urls") or []))
        if not doc_ids and not urls:
            raise ValueError(f"query[{i}] ('{text}') has no relevant_doc_ids or relevant_urls")
        out.append(
            EvalQuery(
                query=text,
                company_id=(str(item["company_id"]) if item.get("company_id") else None),
                relevant_doc_ids=doc_ids,
                relevant_urls=urls,
                query_type=str(item.get("query_type") or "unknown"),
                content_type=str(item.get("content_type") or "unknown"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_capability(session: requests.Session, base_url: str, token: Optional[str]) -> Dict[str, Any]:
    resp = session.get(f"{base_url}{CONFIG_PATH}", headers=_headers(token), timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_search(
    session: requests.Session,
    base_url: str,
    token: Optional[str],
    query: EvalQuery,
    k: int,
    config: Dict[str, Any],
    timeout: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    payload: Dict[str, Any] = {"text": query.query, "k": k, **config}
    if query.company_id:
        payload["filter"] = {"company_id": query.company_id}
    started = perf_counter()
    resp = session.post(
        f"{base_url}{SEARCH_PATH}", json=payload, headers=_headers(token), timeout=timeout
    )
    elapsed_ms = (perf_counter() - started) * 1000.0
    resp.raise_for_status()
    body = resp.json()
    return body.get("results") or [], body.get("retrieval_meta") or {}, elapsed_ms


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    session: requests.Session,
    base_url: str,
    token: Optional[str],
    queries: Sequence[EvalQuery],
    k: int,
    configs: Sequence[str],
    timeout: float,
) -> List[Outcome]:
    outcomes: List[Outcome] = []
    for config_name in configs:
        config = CONFIGS[config_name]
        for query in queries:
            targets = query.targets
            try:
                results, meta, elapsed_ms = run_search(
                    session, base_url, token, query, k, config, timeout
                )
            except Exception as exc:  # noqa: BLE001
                outcomes.append(
                    Outcome(
                        config=config_name,
                        query=query,
                        recall=float("nan"),
                        ndcg=float("nan"),
                        rr=float("nan"),
                        latency_ms=float("nan"),
                        error=str(exc),
                    )
                )
                continue
            hits = [result_hits(row, targets) for row in results]
            outcomes.append(
                Outcome(
                    config=config_name,
                    query=query,
                    recall=recall_at_k(hits, targets, k),
                    ndcg=ndcg_at_k(hits, targets, k),
                    rr=reciprocal_rank(hits),
                    latency_ms=elapsed_ms,
                    retrieval=str(meta.get("retrieval") or ""),
                    rerank_applied=bool(meta.get("rerank_applied")),
                )
            )
    return outcomes


def latency_run(
    session: requests.Session,
    base_url: str,
    token: Optional[str],
    queries: Sequence[EvalQuery],
    k: int,
    config_name: str,
    runs: int,
    warmup: int,
    timeout: float,
) -> List[float]:
    """Timed runs with the first `warmup` discarded.

    Model load and any cache warming land in the warmup calls; including them
    would report a p95 no real user ever experiences.
    """
    config = CONFIGS[config_name]
    samples: List[float] = []
    total = runs + warmup
    for i in range(total):
        query = queries[i % len(queries)]
        try:
            _, _, elapsed_ms = run_search(session, base_url, token, query, k, config, timeout)
        except Exception as exc:  # noqa: BLE001
            log(f"latency run {i + 1}/{total} failed: {exc}")
            continue
        if i >= warmup:
            samples.append(elapsed_ms)
    return samples


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(value: float) -> str:
    return "  n/a" if math.isnan(value) else f"{value:.3f}"


def summarize(outcomes: Sequence[Outcome], k: int) -> Dict[str, Dict[str, float]]:
    by_config: Dict[str, List[Outcome]] = defaultdict(list)
    for outcome in outcomes:
        by_config[outcome.config].append(outcome)
    summary: Dict[str, Dict[str, float]] = {}
    for config_name, rows in by_config.items():
        ok = [r for r in rows if not r.error]
        summary[config_name] = {
            f"recall@{k}": mean([r.recall for r in ok]),
            f"ndcg@{k}": mean([r.ndcg for r in ok]),
            "mrr": mean([r.rr for r in ok]),
            "queries": float(len(rows)),
            "errors": float(len(rows) - len(ok)),
        }
    return summary


def print_summary(outcomes: Sequence[Outcome], k: int) -> None:
    summary = summarize(outcomes, k)
    print("\n=== Overall (higher is better) ===")
    header = f"{'config':<16}{'recall@' + str(k):>12}{'ndcg@' + str(k):>12}{'mrr':>10}{'errors':>9}"
    print(header)
    print("-" * len(header))
    for config_name in CONFIGS:
        if config_name not in summary:
            continue
        row = summary[config_name]
        print(
            f"{config_name:<16}{_fmt(row[f'recall@{k}']):>12}{_fmt(row[f'ndcg@{k}']):>12}"
            f"{_fmt(row['mrr']):>10}{int(row['errors']):>9}"
        )


def print_breakdown(outcomes: Sequence[Outcome], k: int, attr: str) -> None:
    groups: Dict[Tuple[str, str], List[Outcome]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.error:
            continue
        groups[(getattr(outcome.query, attr), outcome.config)].append(outcome)
    if not groups:
        return
    print(f"\n=== Breakdown by {attr} ===")
    header = f"{attr:<18}{'config':<16}{'recall@' + str(k):>12}{'ndcg@' + str(k):>12}{'mrr':>10}{'n':>5}"
    print(header)
    print("-" * len(header))
    for key in sorted({g[0] for g in groups}):
        for config_name in CONFIGS:
            rows = groups.get((key, config_name))
            if not rows:
                continue
            print(
                f"{key:<18}{config_name:<16}"
                f"{_fmt(mean([r.recall for r in rows])):>12}"
                f"{_fmt(mean([r.ndcg for r in rows])):>12}"
                f"{_fmt(mean([r.rr for r in rows])):>10}"
                f"{len(rows):>5}"
            )


def check_rerank_integrity(outcomes: Sequence[Outcome]) -> List[str]:
    """Catch the silent reranker fallback.

    `hybrid+rerank` requests succeed and look normal even when no reranker is
    loaded — the service just returns fusion order. Without this check an eval
    would happily report "hybrid+rerank" numbers that are really plain hybrid.
    """
    problems: List[str] = []
    rows = [o for o in outcomes if o.config == "hybrid+rerank" and not o.error]
    if not rows:
        return problems
    not_applied = [o for o in rows if not o.rerank_applied]
    if not_applied:
        problems.append(
            f"{len(not_applied)}/{len(rows)} 'hybrid+rerank' queries reported "
            f"rerank_applied=false — the reranker did NOT run (silent fallback). "
            f"These numbers are plain hybrid, not hybrid+rerank."
        )
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval eval harness for tinnten-embedding")
    parser.add_argument("--base-url", default=os.getenv("EMBEDDING_BASE_URL", "http://localhost:5003"))
    parser.add_argument("--token", default=os.getenv("EMBEDDING_BEARER_TOKEN"))
    parser.add_argument(
        "--queries",
        default=os.getenv("RAG_EVAL_QUERIES", str(Path(__file__).parent / "rag_eval_queries.json")),
    )
    parser.add_argument("--k", type=int, default=int(os.getenv("RAG_EVAL_K", "10")))
    parser.add_argument(
        "--config",
        action="append",
        choices=list(CONFIGS),
        default=[],
        help="Configs to evaluate (default: all three)",
    )
    parser.add_argument("--timeout", type=float, default=float(os.getenv("RAG_EVAL_TIMEOUT", "60")))
    parser.add_argument("--json-out", default=os.getenv("RAG_EVAL_JSON_OUT"))

    parser.add_argument("--latency-runs", type=int, default=0, help="Timed latency samples (0=skip)")
    parser.add_argument("--latency-warmup", type=int, default=3, help="Discarded warmup calls")
    parser.add_argument("--latency-config", choices=list(CONFIGS), default="hybrid")

    parser.add_argument("--fail-under-recall", type=float, default=None)
    parser.add_argument(
        "--require-rerank",
        action="store_true",
        help="Exit 1 if a 'hybrid+rerank' run did not actually rerank",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    configs = args.config or list(CONFIGS)

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"ERROR: query file not found: {queries_path}", file=sys.stderr)
        return 2
    try:
        queries = load_queries(queries_path)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: invalid query file: {exc}", file=sys.stderr)
        return 2
    if not queries:
        print("ERROR: query file is empty", file=sys.stderr)
        return 2
    if args.k <= 0:
        print("ERROR: --k must be positive", file=sys.stderr)
        return 2

    log(f"Base URL: {base_url}")
    log(f"Queries: {len(queries)} from {queries_path}")
    log(f"Configs: {', '.join(configs)}  k={args.k}")

    session = requests.Session()

    capability: Dict[str, Any] = {}
    if not args.skip_preflight:
        try:
            capability = fetch_capability(session, base_url, args.token)
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: preflight GET {CONFIG_PATH} failed: {exc}\n"
                f"       (auth is on by default — pass --token, or run the server "
                f"locally with REQUIRE_KEYCLOAK_AUTH=false)",
                file=sys.stderr,
            )
            return 2
        log(
            "Capability: hybrid_enabled=%s reranker_configured=%s reranker_loaded=%s"
            % (
                capability.get("hybrid_search_enabled"),
                capability.get("reranker_configured"),
                capability.get("reranker_loaded"),
            )
        )
        if capability.get("reranker_load_error"):
            log(f"WARNING: reranker load error reported: {capability['reranker_load_error']}")
        if "hybrid+rerank" in configs and not capability.get("reranker_configured"):
            log(
                "WARNING: 'hybrid+rerank' requested but RERANKER_MODEL is not set — "
                "the service will silently return fusion order."
            )

    outcomes = evaluate(session, base_url, args.token, queries, args.k, configs, args.timeout)

    print_summary(outcomes, args.k)
    print_breakdown(outcomes, args.k, "query_type")
    print_breakdown(outcomes, args.k, "content_type")

    problems = check_rerank_integrity(outcomes)
    for problem in problems:
        print(f"\nINTEGRITY: {problem}")

    latency_samples: List[float] = []
    if args.latency_runs > 0:
        log(
            f"Latency run: config={args.latency_config} "
            f"runs={args.latency_runs} warmup={args.latency_warmup} (warmup discarded)"
        )
        latency_samples = latency_run(
            session,
            base_url,
            args.token,
            queries,
            args.k,
            args.latency_config,
            args.latency_runs,
            args.latency_warmup,
            args.timeout,
        )
        if latency_samples:
            print(f"\n=== Latency ({args.latency_config}, n={len(latency_samples)}) ===")
            print(f"p50: {percentile(latency_samples, 50):.1f} ms")
            print(f"p95: {percentile(latency_samples, 95):.1f} ms")

    if args.json_out:
        payload = {
            "generated_at": utc_now(),
            "base_url": base_url,
            "k": args.k,
            "capability": capability,
            "summary": summarize(outcomes, args.k),
            "integrity_problems": problems,
            "latency": {
                "config": args.latency_config,
                "samples": len(latency_samples),
                "p50_ms": percentile(latency_samples, 50) if latency_samples else None,
                "p95_ms": percentile(latency_samples, 95) if latency_samples else None,
            },
            "outcomes": [
                {
                    "config": o.config,
                    "query": o.query.query,
                    "query_type": o.query.query_type,
                    "content_type": o.query.content_type,
                    "recall": None if math.isnan(o.recall) else o.recall,
                    "ndcg": None if math.isnan(o.ndcg) else o.ndcg,
                    "rr": None if math.isnan(o.rr) else o.rr,
                    "latency_ms": None if math.isnan(o.latency_ms) else o.latency_ms,
                    "retrieval": o.retrieval,
                    "rerank_applied": o.rerank_applied,
                    "error": o.error,
                }
                for o in outcomes
            ],
        }
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"Wrote {args.json_out}")

    exit_code = 0
    errors = [o for o in outcomes if o.error]
    if errors:
        print(f"\nFAIL: {len(errors)} query runs errored (first: {errors[0].error})")
        exit_code = 1
    if args.require_rerank and problems:
        print("\nFAIL: --require-rerank set but the reranker did not run")
        exit_code = 1
    if args.fail_under_recall is not None:
        summary = summarize(outcomes, args.k)
        for config_name, row in summary.items():
            value = row[f"recall@{args.k}"]
            if not math.isnan(value) and value < args.fail_under_recall:
                print(
                    f"\nFAIL: {config_name} recall@{args.k}={value:.3f} "
                    f"< --fail-under-recall={args.fail_under_recall}"
                )
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
