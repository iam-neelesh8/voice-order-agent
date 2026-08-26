"""The eval harness. One function per stage number.

Nothing downstream starts until the upstream number exists -- otherwise there
is no way to tell which stage caused a regression.
"""

from __future__ import annotations

import time
from typing import Any

from voice_order.evaluation import metrics, queries


def eval_typed_retrieval(
    split: str = "dev",
    query_set: str = "lookup",
    limit: int | None = None,
    category: str | None = None,
    retrievers: str = "lexical",
) -> dict[str, Any]:
    """Stage 2/3 -- recall@k and MRR on typed queries. No audio anywhere.

    The baseline every later stage is measured against. `retrievers` selects
    which components are switched on, so the same harness reports the
    lexical-only baseline and the fused result without a second code path.
    """
    if query_set != "lookup":
        raise NotImplementedError(f"query_set={query_set!r} arrives in stage 3")

    rows = queries.load_lookup_queries()
    if limit:
        rows = rows[:limit]

    from voice_order.retrieval.fusion import Retriever

    retriever = Retriever.load(retrievers=retrievers)

    per_query: list[dict] = []
    latencies: list[float] = []

    for q in rows:
        t0 = time.perf_counter()
        candidates = retriever.search_text(q.question, top_k=20, category=category)
        latencies.append((time.perf_counter() - t0) * 1000)

        row = metrics.score_query(candidates, q.relevant_doc_ids)
        row["category"] = q.category
        row["query_id"] = q.query_id
        per_query.append(row)

    latencies.sort()
    return {
        "query_set": query_set,
        "split": split,
        "retrievers": retrievers,
        "n_queries": len(rows),
        "aggregate": metrics.aggregate(per_query, by="category"),
        "latency_ms": {
            "p50": round(latencies[len(latencies) // 2], 2),
            "p95": round(latencies[int(len(latencies) * 0.95)], 2),
            "mean": round(sum(latencies) / len(latencies), 2),
        },
        "per_query": per_query,
    }


def eval_spoken_retrieval(
    split: str = "dev", condition: str = "phone", nbest: bool = False
) -> dict[str, Any]:
    """Stage 4/5 -- the same metrics, through ASR.

    `condition` is clean | phone | phone_snr20 | ... The drop from
    `eval_typed_retrieval` is the headline problem; n-best fusion and the
    part-number matcher are how much of it comes back.
    """
    raise NotImplementedError("stage 4")


def eval_end_to_end(split: str = "dev") -> dict[str, Any]:
    """Stage 6 -- order accuracy, turns per order, latency per turn."""
    raise NotImplementedError("stage 6")


def eval_human(condition: str = "phone") -> dict[str, Any]:
    """Stage 8 -- the ~100 human recordings.

    Never used for tuning. If synthetic and human disagree, the humans are
    right, and the synthetic pipeline is what needs fixing.
    """
    raise NotImplementedError("stage 8")


def report(result: dict[str, Any]) -> None:
    """Print an eval result. ASCII only, for the Windows console."""
    print()
    print(
        f"{result['query_set']} queries"
        f"  |  retrievers: {result['retrievers']}"
        f"  |  n = {result['n_queries']:,}"
    )
    print()
    print(metrics.format_table(result["aggregate"]))
    lat = result["latency_ms"]
    print()
    print(f"latency per query   p50 {lat['p50']} ms   p95 {lat['p95']} ms   mean {lat['mean']} ms")
